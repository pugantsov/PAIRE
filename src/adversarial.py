from __future__ import annotations

import warnings

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import dill as pickle
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import mode
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", message=r".*'where' used without 'out'.*")

CATEGORICAL_FEATURE_COLS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "race",
    "native-country",
]
NUMERICAL_FEATURE_COLS = [
    "age",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]


def _ensure_2d_instance(x_i):
    """
    Ensure a single instance can be concatenated with a matrix-like background.

    Handles both dense NumPy arrays and sparse rows.
    """
    if sp.issparse(x_i):
        return x_i
    x_i = np.asarray(x_i)
    if x_i.ndim == 1:
        return x_i.reshape(1, -1)
    return x_i


def _stack_background_and_instance(X_background, x_i):
    """
    Stack a background matrix and one target instance while preserving sparse
    behaviour when applicable.
    """
    x_i = _ensure_2d_instance(x_i)
    if sp.issparse(X_background) or sp.issparse(x_i):
        return sp.vstack((X_background, x_i))
    return np.vstack((X_background, x_i))


def _majority_vote(predictions: list[int]) -> int:
    """
    Return the majority-vote prediction.
    """
    result = mode(predictions, keepdims=False).mode
    if np.isscalar(result):
        return int(result)
    return int(np.asarray(result).item())


def process_single_instance(args):
    """
    Process a single attack instance for one quantifier.
    """
    (
        instance_idx,
        x_i,
        quantifier,
        q0,
        X_sigma_groups,
        background_sizes,
        vote_budgets,
        max_vote_budget,
    ) = args

    results = []

    for n in background_sizes:
        preds_i = []

        for b in range(max_vote_budget):
            sigma_union_i = _stack_background_and_instance(
                X_sigma_groups[b][:n], x_i
            )
            q1 = quantifier.quantify(sigma_union_i)

            delta = (q1 * (n + 1)) - (q0[(n, b)] * n)
            preds_i.append(int(np.argmax(delta)))

            if len(preds_i) in vote_budgets:
                results.append((n, len(preds_i), _majority_vote(preds_i)))

    return instance_idx, results


class AttackDataBuilder:
    """
    Prepare attack instances and background pools for the differencing
    attack.
    """

    def __init__(
        self,
        sensitive_col: str,
    ) -> None:
        self.sensitive_col = sensitive_col

    def prepare_attack_data(
        self,
        test_df: pd.DataFrame,
        n_attack_instances: int,
        n_runs: int,
        background_sizes: Sequence[int],
        vote_budgets: Sequence[int],
        base_seed: int = 0,
    ) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
        df = test_df.copy()
        df["id"] = df.index

        I_global = (
            df.groupby(self.sensitive_col, group_keys=False)[df.columns]
            .apply(
                lambda group: group.sample(
                    n=n_attack_instances, random_state=base_seed
                )
            )
            .reset_index(drop=True)
        )

        background_candidates = df[~df["id"].isin(I_global["id"])]

        sigma_runs = []
        max_background_total = max(background_sizes) * max(vote_budgets)

        for run_idx in range(n_runs):
            seed = base_seed + run_idx
            sigma_runs.append(
                background_candidates.sample(
                    n=max_background_total,
                    random_state=seed,
                ).reset_index(drop=True)
            )

        return I_global, sigma_runs


class AdultAttackPreprocessor:
    """
    Transform Adult data using the saved train-fitted preprocessing objects and label encoder.
    """

    def __init__(
        self, data_dir: Path | str, preprocessor_suffix: str = "adult_train"
    ):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory {self.data_dir} does not exist."
            )
        self.preprocessor_suffix = preprocessor_suffix
        self.categorical_feature_cols = CATEGORICAL_FEATURE_COLS
        self.numerical_feature_cols = NUMERICAL_FEATURE_COLS
        self.ohe = joblib.load(
            self.data_dir / f"ohe_{self.preprocessor_suffix}.joblib"
        )
        self.scaler = joblib.load(
            self.data_dir / f"scaler_{self.preprocessor_suffix}.joblib"
        )
        self.label_encoder = joblib.load(
            self.data_dir / f"label_encoder_{self.preprocessor_suffix}.joblib"
        )

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X_cat = self.ohe.transform(
            df[self.categorical_feature_cols].to_numpy()
        )
        X_num = self.scaler.transform(
            df[self.numerical_feature_cols].to_numpy()
        )
        return np.concatenate([X_cat, X_num], axis=1, dtype=float)

    def transform_labels(
        self, df: pd.DataFrame, sensitive_col: str = "sex"
    ) -> np.ndarray:
        return self.label_encoder.transform(
            df[sensitive_col].astype(str).values
        )


class TRECAttackPreprocessor:
    """
    Transform TREC data using the saved train-fitted TF-IDF vectorizer and label
    encoder.
    """

    def __init__(
        self,
        data_dir: Path | str,
        preprocessor_suffix: str = "trec_train",
    ):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory {self.data_dir} does not exist."
            )

        self.preprocessor_suffix = preprocessor_suffix
        self.vectorizer = joblib.load(
            self.data_dir / f"vectorizer_{self.preprocessor_suffix}.joblib"
        )
        self.label_encoder = joblib.load(
            self.data_dir / f"label_encoder_{self.preprocessor_suffix}.joblib"
        )

    def transform_features(self, df: pd.DataFrame):
        return self.vectorizer.transform(df["text"].astype(str).values)

    def transform_labels(
        self,
        df: pd.DataFrame,
        sensitive_col: str = "region",
    ) -> np.ndarray:
        return self.label_encoder.transform(
            df[sensitive_col].astype(str).values
        )


class DifferencingAttackRunner:
    """
    Run the differencing attack against trained quantifiers.
    """

    def __init__(
        self,
        data_dir: Path | str,
        models_dir: Path | str,
        reports_dir: Path | str,
        dataset_name: str,
    ):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)
        if any(
            not (missing := dir_).exists()
            for dir_ in [self.data_dir, self.models_dir, self.reports_dir]
        ):
            raise FileNotFoundError(f"Directory {missing} does not exist.")

        self.dataset_name = dataset_name
        if dataset_name == "adult":
            self.sensitive_col = "sex"

            self.preprocessor = AdultAttackPreprocessor(
                self.data_dir, f"{self.dataset_name}_train"
            )
            self.data_builder = AttackDataBuilder(
                self.sensitive_col,
            )
        elif dataset_name == "trec":
            self.sensitive_col = "region"

            self.preprocessor = TRECAttackPreprocessor(
                self.data_dir, f"{self.dataset_name}_train"
            )
            self.data_builder = AttackDataBuilder(
                self.sensitive_col,
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    def _load_quantifier_paths(
        self,
        quantifiers: Sequence[str] | None = None,
        model_suffix: str | None = None,
    ) -> list[Path]:
        model_paths = sorted(self.models_dir.glob(f"*_{model_suffix}.pkl"))

        if quantifiers is None:
            return model_paths

        allowed = set(quantifiers)
        filtered = [
            path for path in model_paths if path.stem.split("_")[0] in allowed
        ]

        missing = allowed - {path.stem.split("_")[0] for path in filtered}
        if missing:
            raise ValueError(
                f"Missing model files for quantifiers: {sorted(missing)}"
            )

        return filtered

    @staticmethod
    def _split_background_groups(X_sigma, max_vote_budget: int):
        """
        Split one large background matrix into max_vote_budget equally sized groups.
        """
        return [
            X_sigma[
                i
                * X_sigma.shape[0]
                // max_vote_budget : (i + 1)
                * X_sigma.shape[0]
                // max_vote_budget
            ]
            for i in range(max_vote_budget)
        ]

    def run_single(
        self,
        run_idx: int,
        seed: int,
        I_global: pd.DataFrame,
        sigma_global: pd.DataFrame,
        background_sizes: Sequence[int],
        vote_budgets: Sequence[int],
        n_workers: int,
        quantifiers: Sequence[str] | None = None,
        model_suffix: str | None = None,
    ) -> pd.DataFrame:
        """
        Run the attack for one seed/background pool.
        """
        X_I = self.preprocessor.transform_features(I_global)
        X_sigma = self.preprocessor.transform_features(sigma_global)
        y_I = self.preprocessor.transform_labels(
            I_global, sensitive_col=self.sensitive_col
        )

        max_vote_budget = max(vote_budgets)
        X_sigma_groups = self._split_background_groups(
            X_sigma, max_vote_budget
        )

        run_results = []
        model_paths = self._load_quantifier_paths(
            quantifiers=quantifiers,
            model_suffix=model_suffix,
        )

        for model_path in tqdm(
            model_paths,
            desc=f"Run {run_idx + 1} | Quantifiers",
            leave=False,
        ):
            qid = model_path.stem.split("_")[0]

            with model_path.open("rb") as f:
                quantifier = pickle.load(f)

            q0 = {
                (n, b): quantifier.quantify(X_sigma_groups[b][:n])
                for b in range(max_vote_budget)
                for n in background_sizes
            }

            preds = {
                (qid, int(n), int(b)): []
                for n in background_sizes
                for b in vote_budgets
            }

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(
                        process_single_instance,
                        (
                            i,
                            X_I[i],
                            quantifier,
                            q0,
                            X_sigma_groups,
                            list(background_sizes),
                            list(vote_budgets),
                            max_vote_budget,
                        ),
                    ): i
                    for i in range(X_I.shape[0])
                }

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Run {run_idx + 1} | {qid} instances",
                    leave=False,
                ):
                    _, instance_results = future.result()
                    for n, b, pred in instance_results:
                        preds[(qid, int(n), int(b))].append(int(pred))

            for (qid_key, n, b), pred_list in preds.items():
                pred_array = np.array(pred_list, dtype=int)
                macro_f1 = f1_score(y_I, pred_array, average="macro")

                run_results.append(
                    {
                        "dataset": self.dataset_name,
                        "run": run_idx,
                        "seed": seed,
                        "quantifier": qid_key,
                        "n": int(n),
                        "b": int(b),
                        "macro_f1": float(macro_f1),
                    }
                )

        return pd.DataFrame(run_results)

    def run(
        self,
        test_df: pd.DataFrame,
        n_attack_instances: int = 500,
        n_runs: int = 5,
        background_sizes: Sequence[int] = (1, 10, 100),
        vote_budgets: Sequence[int] = (1, 10, 100),
        base_seed: int = 0,
        n_workers: int = 1,
        quantifiers: Sequence[str] | None = None,
        model_suffix: str | None = None,
        save_individual_runs: bool = False,
    ) -> pd.DataFrame:
        I_global, sigma_runs = self.data_builder.prepare_attack_data(
            test_df=test_df,
            n_attack_instances=n_attack_instances,
            n_runs=n_runs,
            background_sizes=background_sizes,
            vote_budgets=vote_budgets,
            base_seed=base_seed,
        )

        all_runs = []
        for run_idx in range(n_runs):
            seed = base_seed + run_idx
            run_df = self.run_single(
                run_idx=run_idx,
                seed=seed,
                I_global=I_global,
                sigma_global=sigma_runs[run_idx],
                background_sizes=background_sizes,
                vote_budgets=vote_budgets,
                n_workers=n_workers,
                quantifiers=quantifiers,
                model_suffix=model_suffix,
            )
            all_runs.append(run_df)

            if save_individual_runs:
                self.reports_dir.mkdir(parents=True, exist_ok=True)
                with (
                    self.reports_dir
                    / f"{self.dataset_name}_adv_run_{run_idx}.pkl"
                ).open("wb") as f:
                    pickle.dump(run_df, f)

        return pd.concat(all_runs, ignore_index=True)

    @staticmethod
    def summarize(results: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize macro-F1 across runs.
        """
        summary = (
            results.groupby(["quantifier", "n", "b"])["macro_f1"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "macro_f1_mean", "std": "macro_f1_std"})
        )
        return summary

    def save_results(
        self, results: pd.DataFrame, output_path: Path | str
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pickle.dump(results, f)

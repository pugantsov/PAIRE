from __future__ import annotations

import math
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import dill as pickle
import joblib
import numpy as np
import pandas as pd
import quapy as qp
from scipy.special import kl_div
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
from whoosh.fields import ID, TEXT, Schema
from whoosh.index import create_in, open_dir
from whoosh.qparser import OrGroup, QueryParser
from whoosh.scoring import BM25F

from src.models import DEFAULT_QUANTIFIERS, get_quantifier_class

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


class QuantifierEvaluator:
    """
    Evaluate trained quantifiers on Estimation Quality metrics.
    """

    ADULT_PROTOCOL_CONFIG = {
        "sample_size": 500,
        "repeats": 10,
    }

    TREC_PROTOCOL_CONFIG = {
        "sample_size": 500,
        "repeats": 100,
    }

    def __init__(
        self,
        dataset: str,
        data_dir: Path | str,
        protocol_config: dict[str, Any] | None = None,
        model_suffix: str = "adult",
    ):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory {self.data_dir} does not exist."
            )
        if dataset not in {"adult", "trec"}:
            raise ValueError("dataset must be one of {'adult', 'trec'}")

        self.dataset = dataset
        self.categorical_feature_cols = CATEGORICAL_FEATURE_COLS
        self.numerical_feature_cols = NUMERICAL_FEATURE_COLS
        self.model_suffix = model_suffix

        if protocol_config is not None:
            self.protocol_config = protocol_config
        elif dataset == "adult":
            self.protocol_config = self.ADULT_PROTOCOL_CONFIG
        else:
            self.protocol_config = self.TREC_PROTOCOL_CONFIG

    def load_adult_preprocessors(
        self,
        suffix: str = "adult_train",
    ) -> tuple[OneHotEncoder, StandardScaler, LabelEncoder]:
        ohe = joblib.load(self.data_dir / f"ohe_{suffix}.joblib")
        scaler = joblib.load(self.data_dir / f"scaler_{suffix}.joblib")
        label_encoder = joblib.load(
            self.data_dir / f"label_encoder_{suffix}.joblib"
        )
        return ohe, scaler, label_encoder

    def load_trec_preprocessors(
        self,
        suffix: str = "trec_train",
    ) -> tuple[TfidfVectorizer, LabelEncoder, np.ndarray]:
        vectorizer = joblib.load(self.data_dir / f"vectorizer_{suffix}.joblib")
        label_encoder = joblib.load(
            self.data_dir / f"label_encoder_{suffix}.joblib"
        )
        labels = np.load(
            self.data_dir / f"labels_{suffix}.npy", allow_pickle=True
        )
        return vectorizer, label_encoder, labels

    def transform_adult_test_dataframe(
        self,
        test_df: pd.DataFrame,
        ohe: OneHotEncoder,
        scaler: StandardScaler,
        label_encoder: LabelEncoder,
        target_col: str = "sex",
    ) -> qp.data.LabelledCollection:
        X = np.concatenate(
            [
                ohe.transform(
                    test_df[self.categorical_feature_cols].to_numpy()
                ),
                scaler.transform(
                    test_df[self.numerical_feature_cols].to_numpy()
                ),
            ],
            axis=1,
            dtype=float,
        )
        y = label_encoder.transform(test_df[target_col].astype(str).to_numpy())
        return qp.data.LabelledCollection(X, y.tolist())

    def evaluate_adult_models(
        self,
        test_df: pd.DataFrame,
        models_dir: Path | str,
        reports_dir: Path | str,
        preprocessor_suffix: str = "adult_train",
        target_col: str = "sex",
    ) -> None:
        models_dir = Path(models_dir)
        if not models_dir.exists():
            raise FileNotFoundError(
                f"Models directory {models_dir} does not exist."
            )

        reports_dir = Path(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)

        ohe, scaler, label_encoder = self.load_adult_preprocessors(
            suffix=preprocessor_suffix
        )
        dataset = self.transform_adult_test_dataframe(
            test_df=test_df,
            ohe=ohe,
            scaler=scaler,
            label_encoder=label_encoder,
            target_col=target_col,
        )

        qp.environ["SAMPLE_SIZE"] = len(dataset)
        protocol = qp.protocol.APP(dataset, **self.protocol_config)

        model_paths = sorted(models_dir.glob(f"*_{self.model_suffix}.pkl"))
        for model_path in (model_iter := tqdm(model_paths, desc="Models")):
            qid = model_path.stem.split("_")[0]
            model_iter.set_description(f"Evaluating {qid}")

            with model_path.open("rb") as f:
                model = pickle.load(f)

            report = qp.evaluation.evaluation_report(
                model,
                protocol,
                error_metrics=["ae", "rae"],
                aggr_speedup="force",
            )
            report.loc[:, "quantifier"] = qid
            report.loc[:, "protocol"] = "APP"
            report.loc[:, "bias"] = report.apply(
                lambda row: np.array(row["estim-prev"])
                - np.array(row["true-prev"]),
                axis=1,
            )

            output_path = reports_dir / f"{qid}_{self.model_suffix}.pkl"
            with output_path.open("wb") as f:
                pickle.dump(report, f)

    def _process_single_trec_query(
        self,
        query_path: Path,
        model_path: Path,
        qid: str,
        labels: np.ndarray,
        vectorizer: TfidfVectorizer,
        reports_dir: Path,
        text_col: str = "text",
        target_col: str = "region",
    ) -> tuple[Path | None, str | None]:
        """
        Evaluate one TREC query file against one trained quantifier and save it.
        """
        query_id = query_path.stem.split("_")[-1]
        query_df = pd.read_json(query_path, lines=True)

        if len(query_df) < self.protocol_config["sample_size"]:
            return (
                None,
                f"Skipping query {query_id}: fewer than {self.protocol_config['sample_size']} documents.",
            )

        qp.environ["SAMPLE_SIZE"] = len(query_df)

        X = vectorizer.transform(query_df[text_col].astype(str).to_numpy())
        y = query_df[target_col].astype(str).to_numpy()

        D = qp.data.LabelledCollection(X, y.tolist(), classes=labels)

        with model_path.open("rb") as f:
            model = pickle.load(f)

        protocol = qp.protocol.NPP(D, **self.protocol_config)
        report = qp.evaluation.evaluation_report(
            model,
            protocol,
            error_metrics=["ae", "rae"],
            aggr_speedup="force",
        )

        report.loc[:, "query_set"] = query_id
        report.loc[:, "quantifier"] = qid
        report.loc[:, "protocol"] = "NPP"
        report.loc[:, "bias"] = report.apply(
            lambda row: np.array(row["estim-prev"])
            - np.array(row["true-prev"]),
            axis=1,
        )

        output_path = (
            reports_dir / f"{qid}_{self.model_suffix}_q{query_id}.pkl"
        )
        with output_path.open("wb") as f:
            pickle.dump(report, f)

    def evaluate_trec_models(
        self,
        models_dir: Path | str,
        reports_dir: Path | str,
        queries_pattern: str = "trec_test_query_*.jsonl",
        preprocessor_suffix: str = "trec_train",
        n_workers: int = 1,
        text_col: str = "text",
        target_col: str = "region",
    ) -> None:
        models_dir = Path(models_dir)
        if not models_dir.exists():
            raise FileNotFoundError(
                f"Models directory {models_dir} does not exist."
            )

        reports_dir = Path(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)

        vectorizer, _, labels = self.load_trec_preprocessors(
            suffix=preprocessor_suffix
        )

        model_paths = sorted(models_dir.glob(f"*_{self.model_suffix}.pkl"))
        query_paths = sorted(self.data_dir.glob(queries_pattern))

        if not query_paths:
            raise FileNotFoundError(
                f"No query files found matching pattern {queries_pattern} in {self.data_dir}"
            )

        print(
            f"Found {len(model_paths)} models and {len(query_paths)} queries"
        )

        for model_path in tqdm(model_paths, desc="Models"):
            qid = model_path.stem.split("_")[0]

            if n_workers <= 1:
                for query_path in tqdm(
                    query_paths,
                    desc=f"Processing {qid}",
                    leave=False,
                ):
                    _, error = self._process_single_trec_query(
                        query_path=query_path,
                        model_path=model_path,
                        qid=qid,
                        labels=labels,
                        vectorizer=vectorizer,
                        reports_dir=reports_dir,
                        text_col=text_col,
                        target_col=target_col,
                    )
                    if error:
                        print(error)
            else:
                process_func = partial(
                    self._process_single_trec_query,
                    model_path=model_path,
                    qid=qid,
                    labels=labels,
                    vectorizer=vectorizer,
                    reports_dir=reports_dir,
                    text_col=text_col,
                    target_col=target_col,
                )

                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = {
                        executor.submit(process_func, query_path): query_path
                        for query_path in query_paths
                    }

                    for future in tqdm(
                        as_completed(futures),
                        total=len(query_paths),
                        desc=f"Processing {qid}",
                        leave=False,
                    ):
                        query_path = futures[future]
                        try:
                            _, error = future.result()
                            if error:
                                print(error)
                        except Exception as e:
                            print(f"Error processing {query_path}: {e}")

    def evaluate_models(
        self,
        models_dir: Path | str,
        reports_dir: Path | str,
        test_df: pd.DataFrame | None = None,
        preprocessor_suffix: str | None = None,
        n_workers: int = 1,
        queries_pattern: str = "trec_test_query_*.jsonl",
    ) -> None:
        if self.dataset == "adult":
            if test_df is None:
                raise ValueError(
                    "test_df must be provided for dataset='adult'"
                )
            self.evaluate_adult_models(
                test_df=test_df,
                models_dir=models_dir,
                reports_dir=reports_dir,
                preprocessor_suffix=preprocessor_suffix or "adult_train",
            )
        else:
            self.evaluate_trec_models(
                models_dir=models_dir,
                reports_dir=reports_dir,
                queries_pattern=queries_pattern,
                preprocessor_suffix=preprocessor_suffix or "trec_train",
                n_workers=n_workers,
            )

    @staticmethod
    def load_reports(
        reports_dir: Path | str,
        model_suffix: str,
    ) -> pd.DataFrame:
        reports_dir = Path(reports_dir)
        if not reports_dir.exists():
            raise FileNotFoundError(
                f"Reports directory {reports_dir} does not exist."
            )

        report_paths = sorted(reports_dir.glob(f"*_{model_suffix}*.pkl"))
        dfs = []
        for path in report_paths:
            with path.open("rb") as f:
                dfs.append(pd.read_pickle(f))

        if not dfs:
            raise FileNotFoundError(
                f"No report files found for suffix '{model_suffix}' in {reports_dir}"
            )

        return pd.concat(dfs, ignore_index=True)

    @staticmethod
    def filter_non_degenerate_prevalences(
        reports: pd.DataFrame,
        prevalence_col: str = "true-prev",
    ) -> pd.DataFrame:
        """
        Remove samples with any class at prevalence 0 or 1 to stabilise RAE calculation.
        """
        reports = reports.copy()
        return reports[
            reports[prevalence_col].apply(
                lambda x: not (
                    (np.array(x) == 0.0) | (np.array(x) == 1.0)
                ).any()
            )
        ]

    @staticmethod
    def generate_summary_table(
        reports: pd.DataFrame,
        model_order: list[str] | None = None,
        metrics: list[str] | None = None,
    ) -> pd.DataFrame:
        metrics = metrics or ["ae", "rae"]

        table = reports.groupby(["quantifier"])[metrics].agg(["mean", "std"])
        table.columns = [
            f"M{col.upper()}_{stat}" for col, stat in table.columns
        ]

        if model_order is not None:
            missing = set(model_order) - set(table.index)
            if missing:
                raise ValueError(
                    f"The following models in model_order are missing from the data: {missing}"
                )
            table = table.reindex(model_order)

        return table


class AdultFairnessEvaluator:
    """
    Evaluate quantifier-based fairness estimation on the Adult dataset.
    """

    def __init__(
        self,
        quantifiers: list[str] | None = None,
        alpha: float = 0.05,
        sensitive_cardinality: int = 2,
        random_state: int = 0,
    ):
        self.quantifiers = quantifiers or DEFAULT_QUANTIFIERS
        self.alpha = alpha
        self.sensitive_cardinality = sensitive_cardinality
        self.random_state = random_state
        self.categorical_feature_cols = CATEGORICAL_FEATURE_COLS
        self.numerical_feature_cols = NUMERICAL_FEATURE_COLS

    def manual_joint_sampling(
        self,
        dataset: qp.data.LabelledCollection,
        n_prevalences: int = 11,
        max_prev: float = 0.1,
        sample_size: int = 5000,
        repeats: int = 10,
        random_state: int = 0,
    ):
        """
        Manual sampling protocol over a binary control label.
        """
        rng = np.random.default_rng(self.random_state)

        y = np.asarray(dataset.y)
        X = np.asarray(dataset.X)

        idx_neg = np.where(y == 0)[0]
        idx_pos = np.where(y == 1)[0]

        prev_grid = np.linspace(0.0, max_prev, n_prevalences)

        for p_pos in prev_grid:
            n_pos = int(round(sample_size * p_pos))
            n_neg = sample_size - n_pos

            if len(idx_pos) == 0 or len(idx_neg) == 0:
                continue

            for _ in range(repeats):
                sample_pos = rng.choice(
                    idx_pos,
                    size=n_pos,
                    replace=len(idx_pos) < n_pos,
                )
                sample_neg = rng.choice(
                    idx_neg,
                    size=n_neg,
                    replace=len(idx_neg) < n_neg,
                )

                indices = np.concatenate([sample_neg, sample_pos])
                rng.shuffle(indices)

                sample_X = X[indices]
                prev = np.array(
                    [n_neg / sample_size, n_pos / sample_size],
                    dtype=float,
                )

                yield indices, (sample_X, prev)

    def _fit_preprocessors(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
    ) -> tuple[OneHotEncoder, StandardScaler]:
        ohe = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
            dtype=float,
        )
        scaler = StandardScaler()

        ohe.fit(
            np.concatenate(
                (
                    data1[self.categorical_feature_cols].values,
                    data2[self.categorical_feature_cols].values,
                )
            )
        )
        scaler.fit(
            np.concatenate(
                (
                    data1[self.numerical_feature_cols].values,
                    data2[self.numerical_feature_cols].values,
                )
            )
        )

        return ohe, scaler

    def _transform(
        self,
        df: pd.DataFrame,
        ohe: OneHotEncoder,
        scaler: StandardScaler,
    ) -> np.ndarray:
        return np.concatenate(
            [
                ohe.transform(df[self.categorical_feature_cols].values),
                scaler.transform(df[self.numerical_feature_cols].values),
            ],
            axis=1,
            dtype=float,
        )

    @staticmethod
    def _build_base_classifier():
        return LogisticRegression(
            solver="lbfgs",
            l1_ratio=0,
            C=1.0,
            max_iter=1000,
            random_state=0,
        )

    def prepare_datasets(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        data3: pd.DataFrame,
    ) -> dict[str, Any]:
        ohe, scaler = self._fit_preprocessors(data1, data2)

        X1 = self._transform(data1, ohe=ohe, scaler=scaler)
        X2 = self._transform(data2, ohe=ohe, scaler=scaler)
        X3 = self._transform(data3, ohe=ohe, scaler=scaler)

        Y1 = pd.factorize(data1["class"], sort=True)[0]
        Y2 = pd.factorize(data2["class"], sort=True)[0]
        Y3 = pd.factorize(data3["class"], sort=True)[0]

        S2_sex = pd.factorize(data2["sex"], sort=True)[0]
        S3_sex = pd.factorize(data3["sex"], sort=True)[0]

        S3_joint = (
            ((data3["sex"] == "Female") & (data3["class"] == ">50K"))
            .astype(int)
            .values
        )

        D1 = qp.data.LabelledCollection(X1, Y1)
        D2 = qp.data.LabelledCollection(X2, S2_sex)
        D3 = qp.data.LabelledCollection(X3, S3_joint)

        return {
            "D1": D1,
            "D2": D2,
            "D3": D3,
            "Y2": Y2,
            "Y3": Y3,
            "S2_sex": S2_sex,
            "S3_sex": S3_sex,
        }

    def evaluate(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        data3: pd.DataFrame,
        classifier_name: str = "lr",
        n_prevalences: int = 11,
        max_prev: float = 0.1,
        sample_size: int = 5000,
        repeats: int = 10,
    ) -> pd.DataFrame:
        prepared = self.prepare_datasets(data1, data2, data3)

        D1 = prepared["D1"]
        D2 = prepared["D2"]
        D3 = prepared["D3"]
        Y2 = prepared["Y2"]
        Y3 = prepared["Y3"]
        S2_sex = prepared["S2_sex"]
        S3_sex = prepared["S3_sex"]

        h = self._build_base_classifier().fit(D1.X, D1.y)

        D2_pred = h.predict(D2.X)

        D2_neg_idx, D2_pos_idx = (np.where(D2_pred == y)[0] for y in [0, 1])
        D2_neg = qp.data.LabelledCollection(D2.X[D2_neg_idx], D2.y[D2_neg_idx])
        D2_pos = qp.data.LabelledCollection(D2.X[D2_pos_idx], D2.y[D2_pos_idx])

        idx_D2_ypos = np.where(Y2 == 1)[0]
        idx_D2_ypos_hatpos = np.where((Y2 == 1) & (D2_pred == 1))[0]

        D2_ypos = qp.data.LabelledCollection(
            D2.X[idx_D2_ypos],
            S2_sex[idx_D2_ypos],
        )
        D2_ypos_hatpos = qp.data.LabelledCollection(
            D2.X[idx_D2_ypos_hatpos],
            S2_sex[idx_D2_ypos_hatpos],
        )

        rows = []

        for qid in (loop := tqdm(self.quantifiers)):
            loop.set_description(qid)
            q_class = get_quantifier_class(qid)

            def build_sensitive_classifier():
                return LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000,
                )

            q_neg = q_class(build_sensitive_classifier()).fit(*D2_neg.Xy)
            q_pos = q_class(build_sensitive_classifier()).fit(*D2_pos.Xy)

            prev_D2_pos = D2_pos.prevalence()
            prev_D2_neg = D2_neg.prevalence()

            q_ypos = q_class(build_sensitive_classifier()).fit(*D2_ypos.Xy)
            q_ypos_hatpos = q_class(build_sensitive_classifier()).fit(
                *D2_ypos_hatpos.Xy
            )

            prev_D2_ypos = D2_ypos.prevalence()
            prev_D2_ypos_hatpos = D2_ypos_hatpos.prevalence()

            protocol = self.manual_joint_sampling(
                D3,
                n_prevalences=n_prevalences,
                max_prev=max_prev,
                sample_size=sample_size,
                repeats=repeats,
            )

            for indices, (sample, prev) in protocol:
                D3_pred = h.predict(sample)
                D3_neg_idx, D3_pos_idx = (
                    np.where(D3_pred == y)[0] for y in [0, 1]
                )

                if len(D3_pos_idx) == 0 or len(D3_neg_idx) == 0:
                    continue

                p_d3_pos = len(D3_pos_idx) / len(sample)
                p_d3_neg = len(D3_neg_idx) / len(sample)

                q_pos_pred = q_pos.quantify(sample[D3_pos_idx])
                q_neg_pred = q_neg.quantify(sample[D3_neg_idx])

                q_pos_pred = (
                    q_pos_pred * len(D3_pos_idx)
                    + prev_D2_pos * self.alpha * self.sensitive_cardinality
                ) / (len(D3_pos_idx) + self.alpha * self.sensitive_cardinality)
                q_neg_pred = (
                    q_neg_pred * len(D3_neg_idx)
                    + prev_D2_neg * self.alpha * self.sensitive_cardinality
                ) / (len(D3_neg_idx) + self.alpha * self.sensitive_cardinality)

                mu_true = []
                for s in [0, 1]:
                    mask = S3_sex[indices] == s
                    mu_true.append(
                        (D3_pred[mask] == 1).mean() if mask.sum() > 0 else 0.0
                    )

                mu_est = []
                for s in [0, 1]:
                    denom = q_pos_pred[s] * p_d3_pos + q_neg_pred[s] * p_d3_neg
                    mu_s = (
                        q_pos_pred[s] * (p_d3_pos / denom)
                        if denom > 0
                        else 0.0
                    )
                    mu_est.append(mu_s)

                dd_true = mu_true[1] - mu_true[0]
                dd_est = mu_est[1] - mu_est[0]
                dd_err = dd_est - dd_true

                dd_mcfe = (
                    abs(mu_est[0] - mu_true[0]) + abs(mu_est[1] - mu_true[1])
                ) / 2

                y_true = Y3[indices]
                s_true = S3_sex[indices]

                idx_ypos = np.where(y_true == 1)[0]
                idx_hatpos_ypos = np.where((y_true == 1) & (D3_pred == 1))[0]

                if len(idx_ypos) == 0 or len(idx_hatpos_ypos) == 0:
                    continue

                p_hatpos_given_ypos = len(idx_hatpos_ypos) / len(idx_ypos)

                pS_given_ypos = q_ypos.quantify(sample[idx_ypos])
                pS_given_hatpos_ypos = q_ypos_hatpos.quantify(
                    sample[idx_hatpos_ypos]
                )

                pS_given_ypos = (
                    pS_given_ypos * len(idx_ypos)
                    + prev_D2_ypos * self.alpha * self.sensitive_cardinality
                ) / (len(idx_ypos) + self.alpha * self.sensitive_cardinality)
                pS_given_hatpos_ypos = (
                    pS_given_hatpos_ypos * len(idx_hatpos_ypos)
                    + prev_D2_ypos_hatpos
                    * self.alpha
                    * self.sensitive_cardinality
                ) / (
                    len(idx_hatpos_ypos)
                    + self.alpha * self.sensitive_cardinality
                )

                tpr_est = []
                for s in [0, 1]:
                    denom = pS_given_ypos[s]
                    val = (
                        pS_given_hatpos_ypos[s] * p_hatpos_given_ypos / denom
                        if denom > 0
                        else 0.0
                    )
                    tpr_est.append(float(np.clip(val, 0.0, 1.0)))

                tpr_true = []
                for s in [0, 1]:
                    mask = (s_true == s) & (y_true == 1)
                    tpr_true.append(
                        (D3_pred[mask] == 1).mean() if mask.sum() > 0 else 0.0
                    )

                tpr_mcfe = (
                    abs(tpr_est[0] - tpr_true[0])
                    + abs(tpr_est[1] - tpr_true[1])
                ) / 2

                tpr_delta_true = tpr_true[1] - tpr_true[0]
                tpr_delta_est = tpr_est[1] - tpr_est[0]
                tpr_delta_err = tpr_delta_est - tpr_delta_true

                rows.append(
                    {
                        "true-prev": prev,
                        "quantifier": qid,
                        "classifier": classifier_name,
                        "dd_true": dd_true,
                        "dd_est": dd_est,
                        "dd_e": dd_err,
                        "dd_mcfe": dd_mcfe,
                        "tpr_delta_true": tpr_delta_true,
                        "tpr_delta_est": tpr_delta_est,
                        "tpr_delta_e": tpr_delta_err,
                        "tpr_mcfe": tpr_mcfe,
                    }
                )

        return pd.DataFrame(rows)

    @staticmethod
    def save_report(report: pd.DataFrame, output_path: Path | str) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pickle.dump(report, f)


class TRECFairnessCorpusBuilder:
    """
    Build the corpus artefacts used for TREC fairness experiments.
    """

    def __init__(
        self,
        max_chars: int = 10000,
        top_n_terms: int = 100,
        random_state: int = 0,
    ) -> None:
        self.max_chars = max_chars
        self.top_n_terms = top_n_terms
        self.random_state = random_state

    def clean_text(self, s: str) -> str:
        s = "" if s is None else str(s)
        s = s.replace("\t", " ").replace("\r", " ").replace("\n", " ")
        s = s[: self.max_chars]
        return s.strip()

    def vectorize(self, texts: Iterable[str]) -> TfidfVectorizer:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2),
        )
        vectorizer.fit(list(texts))
        return vectorizer

    def build_train_nonrelevant_split(
        self,
        train_df: pd.DataFrame,
        test_size: float = 0.9,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_df = train_df.copy()
        train_df.loc[:, "text"] = train_df["text"].apply(self.clean_text)
        train_df = train_df.drop_duplicates(subset=["id"])

        fair_train, nonrel_corpus = train_test_split(
            train_df,
            test_size=test_size,
            random_state=self.random_state,
            stratify=train_df["region"],
        )
        return fair_train, nonrel_corpus

    def build_query_table(
        self,
        query_paths: Iterable[Path],
        vectorizer: TfidfVectorizer,
        skip_query_ids: set[str] | None = None,
    ) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
        skip_query_ids = skip_query_ids or set()

        query_strings = {}
        rel_docs = []

        for query_path in tqdm(
            list(query_paths), desc="Building fairness queries"
        ):
            query_id = query_path.stem.split("_")[-1]
            if query_id in skip_query_ids:
                continue

            query_df = pd.read_json(query_path, lines=True)
            query_df = query_df.drop_duplicates(subset=["id"])
            query_df.loc[:, "text"] = query_df["text"].apply(self.clean_text)
            rel_docs.append(query_df)

            tfidf_matrix = vectorizer.transform(query_df["text"].values)
            feature_names = vectorizer.get_feature_names_out()

            term_scores = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
            top_idx = term_scores.argsort()[::-1][: self.top_n_terms]
            query_strings[int(query_id)] = " ".join(
                feature_names[i] for i in top_idx
            )

        query_df = pd.DataFrame.from_dict(
            query_strings,
            orient="index",
            columns=["query"],
        )
        query_df["qid"] = query_df.index.astype(str)
        query_df = query_df.reset_index(drop=True)

        return query_df, rel_docs

    @staticmethod
    def build_docs_table(
        rel_docs: list[pd.DataFrame],
        nonrel_corpus: pd.DataFrame,
    ) -> pd.DataFrame:
        return pd.concat(rel_docs + [nonrel_corpus], ignore_index=True)

    def build_bm25_ranked_lists(
        self,
        docs: pd.DataFrame,
        queries: pd.DataFrame,
        index_dir: Path | str,
    ) -> dict[str, list[tuple[int, str]]]:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        schema = Schema(
            doc_id=ID(stored=True, unique=True),
            content=TEXT(stored=False),
            region=ID(stored=True),
        )

        ix = create_in(str(index_dir), schema)
        writer = ix.writer(limitmb=1024, procs=2)

        for row in tqdm(
            docs.itertuples(index=False),
            total=len(docs),
            desc="Indexing documents",
        ):
            writer.add_document(
                doc_id=str(row.id),
                content=row.text,
                region=str(row.region),
            )
        writer.commit()

        ix = open_dir(str(index_dir))
        searcher = ix.searcher(weighting=BM25F(B=0.75, K1=1.2))
        parser = QueryParser("content", schema=ix.schema, group=OrGroup)

        ranked_lists: dict[str, list[tuple[int, str]]] = {}
        for row in tqdm(
            queries.itertuples(index=False),
            total=len(queries),
            desc="Ranking corpus for fairness queries",
        ):
            query = parser.parse(row.query)
            results = searcher.search(query, limit=10000)
            ranked_lists[str(row.qid)] = [
                (int(hit["doc_id"]), hit["region"]) for hit in results
            ]

        searcher.close()
        return ranked_lists

    @staticmethod
    def save_pickle(obj: Any, output_path: Path | str) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def save_dataframe(df: pd.DataFrame, output_path: Path | str) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)


class TRECFairnessEvaluator:
    """
    Evaluate quantifier-based fairness estimation on ranked TREC outputs.
    """

    DEFAULT_CUTOFFS = [50, 100, 500, 1000]

    def __init__(
        self,
        quantifiers: list[str] | None = None,
        cutoffs: list[int] | None = None,
        eps: float = 1e-9,
    ) -> None:
        self.quantifiers = quantifiers or DEFAULT_QUANTIFIERS
        self.cutoffs = cutoffs or self.DEFAULT_CUTOFFS
        self.eps = eps

    def compute_local_target_distributions(
        self,
        docs: pd.DataFrame,
        labels: np.ndarray,
    ) -> dict[str, np.ndarray]:
        labels_map = {lbl: i for i, lbl in enumerate(labels)}
        target_distributions = {}

        if "query_set" not in docs.columns:
            raise ValueError(
                "docs dataframe must contain a 'query_set' column for local p*."
            )

        for query_id, query_group in tqdm(
            docs.groupby("query_set"),
            desc="Computing local target distributions",
        ):
            region_counts = Counter(query_group["region"])
            p_star = np.zeros(len(labels), dtype=float)

            for region, count in region_counts.items():
                p_star[labels_map[region]] = count

            p_star = (p_star + self.eps) / (
                p_star.sum() + self.eps * len(labels)
            )
            target_distributions[str(query_id)] = p_star

        return target_distributions

    def evaluate_ranked_lists(
        self,
        ranked_lists: dict[str, list[tuple]],
        docs: pd.DataFrame,
        vectorizer: TfidfVectorizer,
        models_dir: Path | str,
        labels: np.ndarray,
        experiment_name: str = "BM25",
        model_pattern: str = "trec_fair_*.pkl",
    ) -> pd.DataFrame:
        models_dir = Path(models_dir)
        labels_map = {lbl: i for i, lbl in enumerate(labels)}

        id_to_text = {
            int(row.id): row.text for row in docs.itertuples(index=False)
        }
        target_distributions = self.compute_local_target_distributions(
            docs, labels
        )

        rows = []
        model_paths = sorted(models_dir.glob(model_pattern))

        for q_path in (
            q_loop := tqdm(model_paths, desc="Fairness quantifiers")
        ):
            qid = q_path.stem.split("_")[-1]
            qid = {"CC": "TE", "PCC": "WE"}.get(qid, qid)
            q_loop.set_description(f"({qid})")

            with q_path.open("rb") as f:
                q = pickle.load(f)

            for query_id, ranked_list in tqdm(
                ranked_lists.items(), leave=False
            ):
                p_star = target_distributions[str(query_id)]

                for k in self.cutoffs:
                    if len(ranked_list) < k:
                        continue

                    top_k = ranked_list[:k]

                    if len(top_k[0]) == 2:
                        counts = Counter(region for _, region in top_k)
                        texts = [id_to_text[doc_id] for doc_id, _ in top_k]
                    else:
                        counts = Counter(region for _, region, _ in top_k)
                        texts = [id_to_text[doc_id] for doc_id, _, _ in top_k]

                    p = np.zeros(len(labels), dtype=float)
                    for region, count in counts.items():
                        p[labels_map[region]] = count
                    p = (p + self.eps) / (p.sum() + self.eps * len(labels))

                    X = vectorizer.transform(texts)
                    p_hat = q.quantify(X)
                    p_hat = (p_hat + self.eps) / (
                        p_hat.sum() + self.eps * len(labels)
                    )

                    D_true = float(np.sum(kl_div(p, p_star)))
                    D_hat = float(np.sum(kl_div(p_hat, p_star)))

                    rows.append(
                        {
                            "qid": qid,
                            "query_id": str(query_id),
                            "k": k,
                            "D_KL_true": D_true,
                            "D_KL_hat": D_hat,
                            "w_k": 1 / math.log2(k),
                            "exp": experiment_name,
                        }
                    )

        return pd.DataFrame(rows)

    @staticmethod
    def aggregate_rkl(results_df: pd.DataFrame) -> pd.DataFrame:
        def agg_rkl(g: pd.DataFrame) -> pd.Series:
            Zg = g["w_k"].sum()
            rkl_true = (g["w_k"] * g["D_KL_true"]).sum() / Zg
            rkl_hat = (g["w_k"] * g["D_KL_hat"]).sum() / Zg
            return pd.Series(
                {
                    "rKL_true": rkl_true,
                    "rKL_hat": rkl_hat,
                    "MRFE": abs(rkl_hat - rkl_true),
                }
            )

        return (
            results_df.groupby(["qid", "query_id"], as_index=False)
            .apply(agg_rkl)
            .reset_index(drop=True)
        )

    @staticmethod
    def generate_summary_table(rkl_df: pd.DataFrame) -> pd.DataFrame:
        summary = (
            rkl_df.groupby(["qid"], as_index=False)["MRFE"]
            .agg(["mean", "std"])
            .reset_index()
        )
        return summary.round(3).set_index("qid")

    @staticmethod
    def save_report(report: pd.DataFrame, output_path: Path | str) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(output_path, index=False)

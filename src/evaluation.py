from __future__ import annotations

import itertools
import warnings
from pathlib import Path
from typing import Any, Iterable

import dill as pickle
import joblib
import numpy as np
import pandas as pd
import quapy as qp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm.auto import tqdm

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

    DEFAULT_PROTOCOL_CONFIG = {
        "sample_size": 500,
        "repeats": 10,
    }

    def __init__(
        self,
        data_dir: Path | str,
        protocol_config: dict[str, Any] | None = None,
        target_col: str = "sex",
        model_suffix: str = "adult",
    ):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory {self.data_dir} does not exist."
            )
        self.categorical_feature_cols = CATEGORICAL_FEATURE_COLS
        self.numerical_feature_cols = NUMERICAL_FEATURE_COLS
        self.protocol_config = protocol_config or self.DEFAULT_PROTOCOL_CONFIG
        self.target_col = target_col
        self.model_suffix = model_suffix

    @property
    def feature_cols(self) -> list[str]:
        return self.categorical_feature_cols + self.numerical_feature_cols

    def load_preprocessors(
        self, suffix: str = "adult"
    ) -> tuple[OneHotEncoder, StandardScaler]:
        ohe = joblib.load(self.data_dir / f"ohe_{suffix}.joblib")
        scaler = joblib.load(self.data_dir / f"scaler_{suffix}.joblib")
        return ohe, scaler

    def transform_test_dataframe(
        self, test_df: pd.DataFrame, ohe: OneHotEncoder, scaler: StandardScaler
    ) -> qp.data.LabelledCollection:
        X = np.concatenate(
            [
                ohe.transform(test_df[self.categorical_feature_cols]),
                scaler.transform(test_df[self.numerical_feature_cols]),
            ],
            axis=1,
            dtype=float,
        )
        y = test_df[self.target_col].to_list()
        return qp.data.LabelledCollection(X, y)

    def evaluate_models(
        self,
        test_df: pd.DataFrame,
        models_dir: Path | str,
        reports_dir: Path | str,
        preprocessor_suffix: str = "adult",
    ) -> None:
        models_dir = Path(models_dir)
        if not models_dir.exists():
            raise FileNotFoundError(
                f"Models directory {models_dir} does not exist."
            )
        reports_dir = Path(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)

        ohe, scaler = self.load_preprocessors(preprocessor_suffix)

        dataset = self.transform_test_dataframe(
            test_df, ohe=ohe, scaler=scaler
        )
        qp.environ["SAMPLE_SIZE"] = len(dataset)
        protocol = qp.protocol.APP(dataset, **self.protocol_config)

        model_paths = list(models_dir.glob(f"*_{self.model_suffix}.pkl"))
        for model_path in (model_iter := tqdm(model_paths)):
            qid = model_path.stem.split("_")[0]
            model_iter.set_description(qid)

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

            output_path = reports_dir / f"{qid}_{self.model_suffix}.csv"
            with output_path.open("wb") as f:
                pickle.dump(report, f)

    @staticmethod
    def load_reports(
        reports_dir: Path | str, model_suffix: str = "adult"
    ) -> pd.DataFrame:
        reports_dir = Path(reports_dir)
        if not reports_dir.exists():
            raise FileNotFoundError(
                f"Reports directory {reports_dir} does not exist."
            )
        report_paths = list(reports_dir.glob(f"*_{model_suffix}.csv"))

        dfs = []
        for path in report_paths:
            with Path(path).open("rb") as f:
                dfs.append(pd.read_pickle(f))
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


def manual_joint_sampling(
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
    rng = np.random.default_rng(random_state)

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
                idx_pos, size=n_pos, replace=len(idx_pos) < n_pos
            )
            sample_neg = rng.choice(
                idx_neg, size=n_neg, replace=len(idx_neg) < n_neg
            )

            indices = np.concatenate([sample_neg, sample_pos])
            rng.shuffle(indices)

            sample_X = X[indices]
            prev = np.array(
                [n_neg / sample_size, n_pos / sample_size], dtype=float
            )

            yield indices, (sample_X, prev)


class FairnessEvaluator:
    """
    Evaluate quantifier-based fairness estimation.
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

    def _fit_preprocessors(
        self, data1: pd.DataFrame, data2: pd.DataFrame
    ) -> tuple[OneHotEncoder, StandardScaler]:
        ohe = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore", dtype=float
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
        self, df: pd.DataFrame, ohe: OneHotEncoder, scaler: StandardScaler
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
    def _build_base_classifier(classifier_name: str):
        if classifier_name != "lr":
            raise ValueError(
                "Only 'lr' is included in the public codebase version."
            )

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

        h = self._build_base_classifier(classifier_name)
        h = h.fit(D1.X, D1.y)

        D2_pred = h.predict(D2.X)

        D2_neg_idx, D2_pos_idx = (np.where(D2_pred == y)[0] for y in [0, 1])
        D2_neg = qp.data.LabelledCollection(D2.X[D2_neg_idx], D2.y[D2_neg_idx])
        D2_pos = qp.data.LabelledCollection(D2.X[D2_pos_idx], D2.y[D2_pos_idx])

        idx_D2_ypos = np.where(Y2 == 1)[0]
        idx_D2_ypos_hatpos = np.where((Y2 == 1) & (D2_pred == 1))[0]

        D2_ypos = qp.data.LabelledCollection(
            D2.X[idx_D2_ypos], S2_sex[idx_D2_ypos]
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

            protocol = manual_joint_sampling(
                D3,
                n_prevalences=n_prevalences,
                max_prev=max_prev,
                sample_size=sample_size,
                repeats=repeats,
                random_state=self.random_state,
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

                if len(idx_ypos) == 0:
                    continue
                if len(idx_hatpos_ypos) == 0:
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

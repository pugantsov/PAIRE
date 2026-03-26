from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Iterable

import dill as pickle
import joblib
import numpy as np
import pandas as pd
import quapy as qp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", message=r".*'where' used without 'out'.*")

DEFAULT_QUANTIFIERS = ["CC", "PCC", "PACC", "EMQ", "KDEyML"]


def to_serializable(obj: Any) -> Any:
    """
    Recursively convert objects containing NumPy types into JSON-serializable
    Python-native types.
    """
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def get_quantifier_class(qid: str):
    """
    Resolve a QuaPy quantifier class by identifier.
    """
    if qid.startswith("KDEy"):
        return getattr(qp.method._kdey, qid)
    try:
        q_class = getattr(qp.method.aggregative, qid)
    except AttributeError:
        raise ValueError(f"Quantifier {qid} not found") from None
    return q_class


class AdultPreprocessor:
    """
    Preprocessing pipeline for the Adult dataset used in the paper.
    """

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

    def __init__(self):
        self.categorical_feature_cols = self.CATEGORICAL_FEATURE_COLS
        self.numerical_feature_cols = self.NUMERICAL_FEATURE_COLS
        self.ohe = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, dtype=float
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    @property
    def feature_cols(self) -> list[str]:
        return self.categorical_feature_cols + self.numerical_feature_cols

    def fit_transform_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit preprocessing components on a dataframe and return transformed features.
        """
        X_cat = self.ohe.fit_transform(
            df[self.categorical_feature_cols].to_numpy()
        )
        X_num = self.scaler.fit_transform(
            df[self.numerical_feature_cols].to_numpy()
        )
        return np.concatenate([X_cat, X_num], axis=1, dtype=float)

    def transform_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform a dataframe using already-fitted preprocessing components.
        """
        X_cat = self.ohe.transform(
            df[self.categorical_feature_cols].to_numpy()
        )
        X_num = self.scaler.transform(
            df[self.numerical_feature_cols].to_numpy()
        )
        return np.concatenate([X_cat, X_num], axis=1, dtype=float)

    def fit_transform_array_split(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessing components on a training set and transform both training and test sets.
        """
        n_cat = len(self.categorical_feature_cols)

        X_train_cat = self.ohe.fit_transform(X_train[:, :n_cat])
        X_train_num = self.scaler.fit_transform(X_train[:, n_cat:])
        X_test_cat = self.ohe.transform(X_test[:, :n_cat])
        X_test_num = self.scaler.transform(X_test[:, n_cat:])

        X_train_out = np.concatenate(
            [X_train_cat, X_train_num], axis=1, dtype=float
        )
        X_test_out = np.concatenate(
            [X_test_cat, X_test_num], axis=1, dtype=float
        )

        return X_train_out, X_test_out

    def fit_label_encoder(
        self,
        y: pd.Series | np.ndarray | list[str],
    ) -> np.ndarray:
        """
        Fit the label encoder on label values and return encoded labels.
        """
        y = np.asarray(y).astype(str)
        return self.label_encoder.fit_transform(y)

    def transform_labels(
        self,
        y: pd.Series | np.ndarray | list[str],
    ) -> np.ndarray:
        """
        Transform label values using the fitted label encoder.
        """
        y = np.asarray(y).astype(str)
        return self.label_encoder.transform(y)

    def save(
        self, output_dir: Path | str, suffix: str = "adult_train"
    ) -> None:
        """
        Persist fitted preprocessing objects.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.ohe, output_dir / f"ohe_{suffix}.joblib")
        joblib.dump(self.scaler, output_dir / f"scaler_{suffix}.joblib")
        joblib.dump(
            self.label_encoder, output_dir / f"label_encoder_{suffix}.joblib"
        )


class QuantifierHyperparameterTuner:
    """
    Hyperparameter tuner for QuaPy quantifiers using GridSearchQ.
    """

    DEFAULT_CLASSIFIER_GRID = {
        "classifier__C": np.logspace(-3, 3, 7),
        "classifier__max_iter": [1000],
        "classifier__solver": ["saga"],
        "classifier__l1_ratio": [0],
    }

    DEFAULT_PROTOCOL_CONFIG = {
        "sample_size": 500,
        "repeats": 10,
    }

    def __init__(
        self,
        quantifiers: list[str] | None = None,
        classifier_grid: dict[str, Any] | None = None,
        protocol_config: dict[str, Any] | None = None,
        random_state: int = 0,
        target_col: str = "sex",
    ) -> None:
        self.quantifiers = quantifiers or DEFAULT_QUANTIFIERS
        self.classifier_grid = classifier_grid or self.DEFAULT_CLASSIFIER_GRID
        self.protocol_config = protocol_config or self.DEFAULT_PROTOCOL_CONFIG
        self.random_state = random_state
        self.target_col = target_col

    def _build_param_grid(self, qid: str) -> dict[str, Any]:
        """
        Build a parameter grid for a given quantifier.
        """
        if qid.startswith("KDEy"):
            return self.classifier_grid | {
                "bandwidth": np.linspace(0.01, 0.2, 20),
                "random_state": [self.random_state],
            }
        return self.classifier_grid

    def prepare_collections(
        self, train_df: pd.DataFrame, preprocessor: AdultPreprocessor
    ) -> tuple[qp.data.LabelledCollection, qp.data.LabelledCollection]:
        """
        Create the D1/D2 stratified split used for the UCI Adult dataset.
        """
        dataset = qp.data.LabelledCollection(
            train_df[preprocessor.feature_cols],
            train_df[self.target_col].to_list(),
        )
        d1, d2 = dataset.split_stratified(
            train_prop=0.7, random_state=self.random_state
        )
        X_d1, X_d2 = preprocessor.fit_transform_array_split(d1.X, d2.X)
        d1 = qp.data.LabelledCollection(X_d1, d1.y)
        d2 = qp.data.LabelledCollection(X_d2, d2.y)
        return d1, d2

    def tune(
        self, train_df: pd.DataFrame, preprocessor: AdultPreprocessor
    ) -> dict[str, dict[str, Any]]:
        """
        Run hyperparameter tuning for all configured quantifiers.
        """
        d1, d2 = self.prepare_collections(train_df, preprocessor)

        qp.environ["SAMPLE_SIZE"] = len(d2)
        protocol = qp.protocol.APP(d2, **self.protocol_config)

        best_parameters: dict[str, dict[str, Any]] = {}

        for qid in (loop := tqdm(self.quantifiers[::-1])):
            loop.set_description(qid)

            q_class = get_quantifier_class(qid)
            model_selection = qp.model_selection.GridSearchQ(
                model=q_class(LogisticRegression()),
                param_grid=self._build_param_grid(qid),
                protocol=protocol,
                error="mae",
                refit=False,
                verbose=True,
                n_jobs=-1,
            )
            model_selection.fit(*d1.Xy)
            best_parameters[qid] = model_selection.best_params_
        return best_parameters

    def save_best_parameters(
        self,
        best_parameters: dict[str, dict[str, Any]],
        output_path: Path | str,
    ) -> None:
        """
        Save tuned parameters to JSON.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            json.dump(to_serializable(best_parameters), f, indent=4)


class QuantifierTrainer:
    """
    Train and persist quantification models using tuned hyperparameters.
    """

    def __init__(
        self,
        quantifiers: Iterable[str] | None = None,
        target_col: str = "sex",
    ) -> None:
        self.quantifiers = list(quantifiers or DEFAULT_QUANTIFIERS)
        self.target_col = target_col

    @staticmethod
    def load_parameters(
        params_path: Path | str,
    ) -> dict[str, dict[str, Any]]:
        """
        Load tuned hyperparameters from JSON.
        """
        with Path(params_path).open("r") as f:
            return json.load(f)

    @staticmethod
    def _split_quantifier_and_classifier_params(
        params: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Split classifier__* keys from quantifier-specific parameters.
        """
        params_q = params.copy()
        params_clf = {
            k.removeprefix("classifier__"): params_q.pop(k)
            for k in list(params_q)
            if k.startswith("classifier__")
        }
        return params_q, params_clf

    def prepare_training_collection(
        self,
        train_df: pd.DataFrame,
        preprocessor: AdultPreprocessor,
    ) -> qp.data.LabelledCollection:
        """
        Preprocess the full training dataframe and return a labelled collection.
        """
        X = preprocessor.fit_transform_dataframe(train_df)
        y = preprocessor.fit_label_encoder(train_df[self.target_col].tolist())
        return qp.data.LabelledCollection(X, y.tolist())

    def train_and_save(
        self,
        train_df: pd.DataFrame,
        params: dict[str, dict[str, Any]],
        preprocessor: AdultPreprocessor,
        data_dir: Path | str,
        model_dir: Path | str,
        model_suffix: str = "adult",
    ) -> None:
        """
        Train all models and persist them to disk.
        """
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(
                f"Data directory {data_dir} does not exist."
            )
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        dataset = self.prepare_training_collection(train_df, preprocessor)
        preprocessor.save(data_dir, suffix=f"{model_suffix}_train")

        for qid, q_params in (loop := tqdm(params.items())):
            loop.set_description(qid)

            params_q, params_clf = (
                self._split_quantifier_and_classifier_params(q_params)
            )
            q_class = get_quantifier_class(qid)

            quantifier = q_class(LogisticRegression(**params_clf), **params_q)
            quantifier.fit(*dataset.Xy)

            with (model_dir / f"{qid}_{model_suffix}.pkl").open("wb") as f:
                pickle.dump(quantifier, f)

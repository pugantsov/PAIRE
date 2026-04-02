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
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

import src.utils as utils

warnings.filterwarnings("ignore", message=r".*'where' used without 'out'.*")

DEFAULT_QUANTIFIERS = ["CC", "PCC", "PACC", "EMQ", "KDEyML"]


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
        X_cat = self.ohe.fit_transform(
            df[self.categorical_feature_cols].to_numpy()
        )
        X_num = self.scaler.fit_transform(
            df[self.numerical_feature_cols].to_numpy()
        )
        return np.concatenate([X_cat, X_num], axis=1, dtype=float)

    def transform_dataframe(self, df: pd.DataFrame) -> np.ndarray:
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
        y = np.asarray(y).astype(str)
        return self.label_encoder.fit_transform(y)

    def transform_labels(
        self,
        y: pd.Series | np.ndarray | list[str],
    ) -> np.ndarray:
        y = np.asarray(y).astype(str)
        return self.label_encoder.transform(y)

    def prepare_tuning_collections(
        self,
        train_df: pd.DataFrame,
        random_state: int = 0,
    ) -> tuple[qp.data.LabelledCollection, qp.data.LabelledCollection]:
        dataset = qp.data.LabelledCollection(
            train_df[self.feature_cols],
            train_df[self.target_col].tolist(),
        )
        d1, d2 = dataset.split_stratified(
            train_prop=0.7, random_state=random_state
        )
        X_d1, X_d2 = self.fit_transform_array_split(d1.X, d2.X)
        y_d1 = self.fit_label_encoder(d1.y)
        y_d2 = self.transform_labels(d2.y)

        d1 = qp.data.LabelledCollection(X_d1, y_d1.tolist())
        d2 = qp.data.LabelledCollection(X_d2, y_d2.tolist())
        return d1, d2

    def prepare_training_collection(
        self,
        train_df: pd.DataFrame,
    ) -> qp.data.LabelledCollection:
        X = self.fit_transform_dataframe(train_df)
        y = self.fit_label_encoder(train_df[self.target_col].tolist())
        return qp.data.LabelledCollection(X, y.tolist())

    def save(
        self, output_dir: Path | str, suffix: str = "adult_train"
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.ohe, output_dir / f"ohe_{suffix}.joblib")
        joblib.dump(self.scaler, output_dir / f"scaler_{suffix}.joblib")
        joblib.dump(
            self.label_encoder, output_dir / f"label_encoder_{suffix}.joblib"
        )


class TRECPreprocessor:
    """
    Preprocessing pipeline for the TREC dataset used in the paper.
    """

    TEXT_COL = "text"
    TARGET_COL = "region"

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: tuple[int, int] = (1, 2),
        lowercase: bool = True,
        stop_words: str | None = "english",
    ):
        self.text_col = self.TEXT_COL
        self.target_col = self.TARGET_COL
        self.vectorizer = TfidfVectorizer(
            lowercase=lowercase,
            stop_words=stop_words,
            max_features=max_features,
            ngram_range=ngram_range,
        )
        self.label_encoder = LabelEncoder()
        self.classes_ = None

    def fit_transform_dataframe(self, df: pd.DataFrame):
        return self.vectorizer.fit_transform(
            df[self.text_col].astype(str).to_numpy()
        )

    def transform_dataframe(self, df: pd.DataFrame):
        return self.vectorizer.transform(
            df[self.text_col].astype(str).to_numpy()
        )

    def fit_label_encoder(
        self,
        y: pd.Series | np.ndarray | list[str],
    ) -> np.ndarray:
        y = np.asarray(y).astype(str)
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        return y_encoded

    def transform_labels(
        self,
        y: pd.Series | np.ndarray | list[str],
    ) -> np.ndarray:
        y = np.asarray(y).astype(str)
        return self.label_encoder.transform(y)

    def prepare_tuning_collections(
        self,
        train_df: pd.DataFrame,
        random_state: int = 0,
    ) -> tuple[qp.data.LabelledCollection, qp.data.LabelledCollection]:
        dataset = qp.data.LabelledCollection(
            train_df[self.text_col].astype(str).to_numpy(),
            train_df[self.target_col].tolist(),
        )
        d1, d2 = dataset.split_stratified(
            train_prop=0.7, random_state=random_state
        )

        X_d1 = self.vectorizer.fit_transform(d1.X)
        X_d2 = self.vectorizer.transform(d2.X)

        y_d1 = self.fit_label_encoder(d1.y)
        y_d2 = self.transform_labels(d2.y)

        d1 = qp.data.LabelledCollection(
            X_d1, y_d1.tolist(), classes=np.arange(len(self.classes_))
        )
        d2 = qp.data.LabelledCollection(
            X_d2, y_d2.tolist(), classes=np.arange(len(self.classes_))
        )
        return d1, d2

    def prepare_training_collection(
        self,
        train_df: pd.DataFrame,
    ) -> qp.data.LabelledCollection:
        X = self.fit_transform_dataframe(train_df)
        y = self.fit_label_encoder(train_df[self.target_col].tolist())
        return qp.data.LabelledCollection(
            X, y.tolist(), classes=np.arange(len(self.classes_))
        )

    def save(
        self,
        output_dir: Path | str,
        suffix: str = "trec_train",
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            self.vectorizer, output_dir / f"vectorizer_{suffix}.joblib"
        )
        joblib.dump(
            self.label_encoder,
            output_dir / f"label_encoder_{suffix}.joblib",
        )
        if self.classes_ is not None:
            np.save(output_dir / f"labels_{suffix}.npy", self.classes_)


class QuantifierHyperparameterTuner:
    """
    Hyperparameter tuner for QuaPy quantifiers using GridSearchQ.
    """

    DEFAULT_CLASSIFIER_GRID = {
        "classifier__C": np.logspace(-3, 3, 7),
        "classifier__solver": ["saga"],
        "classifier__l1_ratio": [0],
    }

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
        quantifiers: list[str] | None = None,
        classifier_grid: dict[str, Any] | None = None,
        protocol_config: dict[str, Any] | None = None,
        random_state: int = 0,
    ) -> None:
        if dataset not in {"adult", "trec"}:
            raise ValueError("dataset must be one of {'adult', 'trec'}")
        self.dataset = dataset
        self.quantifiers = quantifiers or DEFAULT_QUANTIFIERS
        self.classifier_grid = classifier_grid or self.DEFAULT_CLASSIFIER_GRID
        self.random_state = random_state

        if protocol_config is not None:
            self.protocol_config = protocol_config
        elif dataset == "adult":
            self.protocol_config = self.ADULT_PROTOCOL_CONFIG
        else:
            self.protocol_config = self.TREC_PROTOCOL_CONFIG

    def _build_param_grid(self, qid: str) -> dict[str, Any]:
        if qid.startswith("KDEy"):
            return self.classifier_grid | {
                "bandwidth": np.linspace(0.01, 0.2, 20),
                "random_state": [self.random_state],
            }
        return self.classifier_grid

    def tune(
        self,
        train_df: pd.DataFrame,
        preprocessor: AdultPreprocessor | TRECPreprocessor,
        protocol: qp.protocol.APP | qp.protocol.NPP,
    ) -> dict[str, dict[str, Any]]:
        d1, d2 = preprocessor.prepare_tuning_collections(
            train_df, random_state=self.random_state
        )

        qp.environ["SAMPLE_SIZE"] = len(d2)
        protocol = protocol(d2, **self.protocol_config)

        best_parameters: dict[str, dict[str, Any]] = {}

        for qid in (loop := tqdm(self.quantifiers)):
            loop.set_description(qid)

            q_class = utils.get_quantifier_class(qid)
            model_selection = qp.model_selection.GridSearchQ(
                model=q_class(LogisticRegression()),
                param_grid=self._build_param_grid(qid),
                protocol=protocol,
                error="mrae",
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
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            json.dump(utils.to_serializable(best_parameters), f, indent=4)


class QuantifierTrainer:
    """
    Train and persist quantification models using tuned hyperparameters.
    """

    def __init__(
        self,
        quantifiers: Iterable[str] | None = None,
    ) -> None:
        self.quantifiers = list(quantifiers or DEFAULT_QUANTIFIERS)

    @staticmethod
    def load_parameters(
        params_path: Path | str,
    ) -> dict[str, dict[str, Any]]:
        with Path(params_path).open("r") as f:
            return json.load(f)

    @staticmethod
    def _split_quantifier_and_classifier_params(
        params: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        params_q = params.copy()
        params_clf = {
            k.removeprefix("classifier__"): params_q.pop(k)
            for k in list(params_q)
            if k.startswith("classifier__")
        }
        return params_q, params_clf

    def train_and_save(
        self,
        train_df: pd.DataFrame,
        params: dict[str, dict[str, Any]],
        preprocessor: AdultPreprocessor | TRECPreprocessor,
        data_dir: Path | str,
        model_dir: Path | str,
        model_suffix: str,
    ) -> None:
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(
                f"Data directory {data_dir} does not exist."
            )
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        dataset = preprocessor.prepare_training_collection(train_df)
        preprocessor.save(data_dir, suffix=f"{model_suffix}_train")

        for qid, q_params in (loop := tqdm(params.items())):
            loop.set_description(qid)

            params_q, params_clf = (
                self._split_quantifier_and_classifier_params(q_params)
            )
            q_class = utils.get_quantifier_class(qid)

            quantifier = q_class(LogisticRegression(**params_clf), **params_q)
            quantifier.fit(*dataset.Xy)

            with (model_dir / f"{qid}_{model_suffix}.pkl").open("wb") as f:
                pickle.dump(quantifier, f)

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.models import (
    AdultPreprocessor,
    TRECPreprocessor,
    QuantifierHyperparameterTuner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune hyperparameters for quantification models."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["adult", "trec"],
        required=True,
        help="Dataset to use.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing the dataset.",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default=None,
        help="Training CSV filename inside --data-dir.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Path to the JSON file where tuned parameters will be saved.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed used for split and/or model-based random state.",
    )
    return parser.parse_args()


def get_train_file(dataset: str, train_file: str | None) -> str:
    if train_file is not None:
        return train_file
    return {
        "adult": "adult_train.csv",
        "trec": "trec_train.jsonl",
    }[dataset]


def load_training_data(dataset: str, train_path: Path) -> pd.DataFrame:
    if dataset == "adult":
        return pd.read_csv(train_path)
    if dataset == "trec":
        return pd.read_json(train_path, lines=True)
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_preprocessor(dataset: str):
    if dataset == "adult":
        return AdultPreprocessor()
    if dataset == "trec":
        return TRECPreprocessor()
    raise ValueError(f"Unsupported dataset: {dataset}")


def main() -> None:
    args = parse_args()

    train_file = get_train_file(args.dataset, args.train_file)
    train = load_training_data(args.dataset, args.data_dir / train_file)

    preprocessor = build_preprocessor(args.dataset)
    tuner = QuantifierHyperparameterTuner(random_state=args.random_state)

    best_parameters = tuner.tune(train_df=train, preprocessor=preprocessor)
    tuner.save_best_parameters(best_parameters, args.output_file)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=r".*'where' used without 'out'.*")

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
    return parser.parse_args()


def load_training_data(dataset: str, data_dir: Path) -> pd.DataFrame:
    if dataset == "adult":
        return pd.concat(
            [
                pd.read_csv(data_dir / train_path)
                for train_path in ["adult_D1.csv", "adult_D2.csv"]
            ]
        )
    if dataset == "trec":
        return pd.read_json(data_dir / "trec_train.jsonl", lines=True)
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_preprocessor(dataset: str):
    if dataset == "adult":
        return AdultPreprocessor()
    if dataset == "trec":
        return TRECPreprocessor()
    raise ValueError(f"Unsupported dataset: {dataset}")


def create_dirs(project_root: Path) -> dict[str, Path]:
    dirs = {}
    for subfolder in ["data", "models", "reports"]:
        (project_root / subfolder).mkdir(parents=True, exist_ok=True)
        dirs[subfolder] = project_root / subfolder
    return dirs


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    dirs = create_dirs(project_root)

    train = load_training_data(args.dataset, dirs["data"])

    preprocessor = build_preprocessor(args.dataset)
    tuner = QuantifierHyperparameterTuner(args.dataset)

    best_parameters = tuner.tune(train_df=train, preprocessor=preprocessor)
    tuner.save_best_parameters(
        best_parameters, dirs["models"] / f"params_{args.dataset}.json"
    )


if __name__ == "__main__":
    main()

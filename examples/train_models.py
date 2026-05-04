from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models import (
    AdultPreprocessor,
    TRECPreprocessor,
    QuantifierTrainer,
)

warnings.filterwarnings("ignore", message=r".*'where' used without 'out'.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train quantification models."
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


def run(
    dataset: str,
    data_dir: Path | str,
    models_dir: Path | str,
    parameters: dict[str, dict[str, Any]],
    quantifiers: list[str] | None = None,
    random_seed: int = 0,
    save_preprocessor: bool = True,
) -> None:
    data_dir = Path(data_dir)
    models_dir = Path(models_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")
    models_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(random_seed)

    train_df = load_training_data(dataset, data_dir)
    preprocessor = build_preprocessor(dataset)

    quantifiers = list(quantifiers) if quantifiers else list(parameters.keys())
    missing = [qid for qid in quantifiers if qid not in parameters]
    if missing:
        raise ValueError(f"Missing parameters for quantifiers: {missing}")

    trainer = QuantifierTrainer(quantifiers=quantifiers)
    trainer.train_and_save(
        train_df=train_df,
        params=parameters,
        preprocessor=preprocessor,
        data_dir=data_dir,
        model_dir=models_dir,
        model_suffix=dataset,
        save_preprocessor=save_preprocessor,
    )


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    dirs = create_dirs(project_root)

    parameters = QuantifierTrainer.load_parameters(
        dirs["models"] / f"params_{args.dataset}.json"
    )

    run(
        dataset=args.dataset,
        data_dir=dirs["data"],
        models_dir=dirs["models"],
        parameters=parameters,
    )


if __name__ == "__main__":
    main()

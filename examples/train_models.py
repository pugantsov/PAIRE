from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.models import AdultPreprocessor, QuantifierTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune hyperparameters for Adult quantification models."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing the Adult dataset.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Directory where trained quantifiers will be saved.",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="adult_train.csv",
        help="Training CSV filename inside --data-dir.",
    )
    parser.add_argument(
        "--params-file",
        type=Path,
        default=None,
        help="Path to the JSON file where tuned parameters will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train = pd.read_csv(args.data_dir / args.train_file)

    preprocessor = AdultPreprocessor()
    trainer = QuantifierTrainer()

    params = trainer.load_parameters(args.params_file)
    trainer.train_and_save(
        train_df=train,
        params=params,
        preprocessor=preprocessor,
        data_dir=args.data_dir,
        model_dir=args.models_dir,
        model_suffix="adult",
    )


if __name__ == "__main__":
    main()

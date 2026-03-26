from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.models import AdultPreprocessor, QuantifierHyperparameterTuner


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
        "--train-file",
        type=str,
        default="adult_train.csv",
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
        help="Random seed used for the D1/D2 split and model-based random state.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train = pd.read_csv(args.data_dir / args.train_file)

    preprocessor = AdultPreprocessor()
    tuner = QuantifierHyperparameterTuner(
        random_state=args.random_state,
        quantifiers=["CC", "PCC", "PACC", "EMQ", "KDEyML"],
    )

    best_parameters = tuner.tune(train_df=train, preprocessor=preprocessor)
    tuner.save_best_parameters(best_parameters, args.output_file)


if __name__ == "__main__":
    main()

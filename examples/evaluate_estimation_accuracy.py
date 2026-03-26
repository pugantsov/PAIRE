from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation import QuantifierEvaluator
from src.models import DEFAULT_QUANTIFIERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained quantifiers on estimation accuracy."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing the test CSV.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Directory containing trained quantifier models.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Directory where per-model evaluation reports will be saved.",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="adult_test.csv",
        help="Test CSV filename inside --data-dir.",
    )
    parser.add_argument(
        "--preprocessor-suffix",
        type=str,
        default="adult_train",
        help="Filename suffix used for saved preprocessors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    test = pd.read_csv(args.data_dir / args.test_file)

    evaluator = QuantifierEvaluator(
        data_dir=args.data_dir, model_suffix="adult"
    )
    evaluator.evaluate_models(
        test_df=test,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        preprocessor_suffix=args.preprocessor_suffix,
    )

    reports = evaluator.load_reports(args.reports_dir, model_suffix="adult")
    reports = evaluator.filter_non_degenerate_prevalences(reports)

    summary = evaluator.generate_summary_table(
        reports,
        model_order=DEFAULT_QUANTIFIERS[:-1],
        metrics=["ae", "rae"],
    )
    print(summary)


if __name__ == "__main__":
    main()

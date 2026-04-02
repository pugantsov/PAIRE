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
        "--dataset",
        type=str,
        choices=["adult", "trec"],
        required=True,
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing the dataset files and saved preprocessing artefacts.",
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
        default=None,
        help="Adult test CSV filename inside --data-dir. Ignored for TREC unless query pattern usage downstream is overridden.",
    )
    parser.add_argument(
        "--queries-pattern",
        type=str,
        default="trec_test_query_*.jsonl",
        help="Glob pattern for TREC query files inside --data-dir.",
    )
    parser.add_argument(
        "--preprocessor-suffix",
        type=str,
        default=None,
        help="Filename suffix used for saved preprocessors.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of worker processes for TREC query evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    evaluator = QuantifierEvaluator(
        data_dir=args.data_dir,
        dataset=args.dataset,
        model_suffix=args.dataset,
    )

    if args.dataset == "adult":
        test_file = args.test_file or "adult_test.csv"
        test = pd.read_csv(args.data_dir / test_file)

        evaluator.evaluate_models(
            test_df=test,
            models_dir=args.models_dir,
            reports_dir=args.reports_dir,
            preprocessor_suffix=args.preprocessor_suffix or "adult_train",
        )

    else:
        evaluator.evaluate_models(
            models_dir=args.models_dir,
            reports_dir=args.reports_dir,
            preprocessor_suffix=args.preprocessor_suffix or "trec_train",
            n_workers=args.n_workers,
            queries_pattern=args.queries_pattern,
        )

    reports = evaluator.load_reports(
        args.reports_dir,
        model_suffix=args.dataset,
    )
    reports = evaluator.filter_non_degenerate_prevalences(reports)

    summary = evaluator.generate_summary_table(
        reports,
        model_order=DEFAULT_QUANTIFIERS,
        metrics=["ae", "rae"],
    )
    print(summary)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=r".*'where' used without 'out'.*")

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
        "--n-workers",
        type=int,
        default=1,
        help="Number of worker processes for TREC query evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    dirs = {
        folder: project_root / folder
        for folder in ["data", "models", "reports"]
    }

    evaluator = QuantifierEvaluator(
        data_dir=dirs["data"],
        dataset=args.dataset,
        model_suffix=args.dataset,
    )

    if args.dataset == "adult":
        test = pd.read_csv(dirs["data"] / "adult_D3.csv")

        evaluator.evaluate_models(
            test_df=test,
            models_dir=dirs["models"],
            reports_dir=dirs["reports"],
            preprocessor_suffix="adult_train",
        )

    else:
        evaluator.evaluate_models(
            models_dir=dirs["models"],
            reports_dir=dirs["reports"],
            preprocessor_suffix="trec_train",
            n_workers=args.n_workers,
            queries_pattern="trec_test_query_*.jsonl",
        )

    reports = evaluator.load_reports(
        dirs["reports"],
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

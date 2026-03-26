from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation import FairnessEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fairness estimation using quantification."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing the test CSV.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Directory where per-model evaluation reports will be saved.",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="adult",
        help="Dataset ID used for saved preprocessors and evaluation reports.",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="lr",
        choices=["lr"],
        help="Downstream classifier used for partitioning.",
    )
    parser.add_argument(
        "--n-prevalences",
        type=int,
        default=11,
        help="Number of prevalence points in the manual sampling protocol.",
    )
    parser.add_argument(
        "--max-prev",
        type=float,
        default=0.1,
        help="Maximum prevalence of the positive control label in D3 sampling.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Sample size for the manual sampling protocol.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of repetitions per prevalence point.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data1 = pd.read_csv(args.data_dir / f"{args.dataset_id}_D1.csv")
    data2 = pd.read_csv(args.data_dir / f"{args.dataset_id}_D2.csv")
    data3 = pd.read_csv(args.data_dir / f"{args.dataset_id}_D3.csv")

    evaluator = FairnessEvaluator()

    report = evaluator.evaluate(
        data1=data1,
        data2=data2,
        data3=data3,
        classifier_name=args.classifier,
        n_prevalences=args.n_prevalences,
        max_prev=args.max_prev,
        sample_size=args.sample_size,
        repeats=args.repeats,
    )

    evaluator.save_report(
        report, args.reports_dir / f"{args.dataset_id}_fairness.pkl"
    )
    print(report.head())


if __name__ == "__main__":
    main()

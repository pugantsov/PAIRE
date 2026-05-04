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


def run(
    dataset: str,
    data_dir: Path | str,
    models_dir: Path | str,
    reports_dir: Path | str,
    sample_size: int | None = None,
    repeats: int | None = None,
    quantifiers: list[str] | None = None,
    n_workers: int = 1,
    print_summary: bool = True,
) -> pd.DataFrame:
    """
    Programmatic entry point for the estimation-accuracy evaluation.

    Used by both the CLI in this script and by the reproducibility
    pipeline in `reproduce/`.
    """
    data_dir = Path(data_dir)
    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory {data_dir} does not exist."
        )
    if not models_dir.exists():
        raise FileNotFoundError(
            f"Models directory {models_dir} does not exist."
        )
    reports_dir.mkdir(parents=True, exist_ok=True)

    protocol_config: dict | None = None
    if sample_size is not None and repeats is not None:
        protocol_config = {
            "sample_size": sample_size,
            "repeats": repeats,
        }

    evaluator = QuantifierEvaluator(
        data_dir=data_dir,
        dataset=dataset,
        model_suffix=dataset,
        protocol_config=protocol_config,
    )

    if dataset == "adult":
        test = pd.read_csv(data_dir / "adult_D3.csv")
        evaluator.evaluate_models(
            test_df=test,
            models_dir=models_dir,
            reports_dir=reports_dir,
            preprocessor_suffix="adult_train",
            quantifiers=list(quantifiers) if quantifiers else None,
        )
    else:
        evaluator.evaluate_models(
            models_dir=models_dir,
            reports_dir=reports_dir,
            preprocessor_suffix="trec_train",
            n_workers=n_workers,
            queries_pattern="trec_test_query_*.jsonl",
            quantifiers=list(quantifiers) if quantifiers else None,
        )

    reports = evaluator.load_reports(reports_dir, model_suffix=dataset)
    reports = evaluator.filter_non_degenerate_prevalences(reports)

    model_order = list(quantifiers) if quantifiers else DEFAULT_QUANTIFIERS
    summary = evaluator.generate_summary_table(
        reports,
        model_order=model_order,
        metrics=["ae", "rae"],
    )

    if print_summary:
        print(summary)

    return summary


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    dirs = {
        folder: project_root / folder
        for folder in ["data", "models", "reports"]
    }

    run(
        dataset=args.dataset,
        data_dir=dirs["data"],
        models_dir=dirs["models"],
        reports_dir=dirs["reports"],
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()

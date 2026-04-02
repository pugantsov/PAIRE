from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import pandas as pd

from src.adversarial import DifferencingAttackRunner


def parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_str_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the differencing attack against trained quantifiers."
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
        required=True,
        help="Directory containing saved preprocessors and test data.",
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
        help="Directory for optional per-run outputs.",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="adult_test.csv",
        help="Adult test file. Defaults to adult_test.csv for Adult. Ignored for TREC.",
    )
    parser.add_argument(
        "--quantifiers",
        type=parse_str_list,
        default=parse_str_list("CC,PCC,PACC,EMQ,KDEyML"),
        help="Comma-separated list of quantifiers to evaluate.",
    )
    parser.add_argument(
        "--n-attack-instances",
        type=int,
        default=500,
        help="Number of attack instances sampled per sensitive group.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Number of independent runs / background pools.",
    )
    parser.add_argument(
        "--background-sizes",
        type=parse_int_list,
        default=[1, 10, 100],
        help="Comma-separated background sizes n, e.g. 1,10,100.",
    )
    parser.add_argument(
        "--vote-budgets",
        type=parse_int_list,
        default=[1, 10, 100],
        help="Comma-separated repetition budgets B, e.g. 1,10,100.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="Base random seed.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=max(mp.cpu_count() - 1, 1),
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--save-individual-runs",
        action="store_true",
        help="Also save one pickle per run in --reports-dir.",
    )
    return parser.parse_args()


def load_test_data(args: argparse.Namespace) -> pd.DataFrame:
    if args.dataset == "adult":
        test_file = args.test_file or "adult_test.csv"
        return pd.read_csv(args.data_dir / test_file)

    query_paths = sorted(args.data_dir.glob("trec_test_query_*.jsonl"))
    if not query_paths:
        raise FileNotFoundError(
            f"No TREC query files found matching 'trec_test_query_*.jsonl' in {args.data_dir}"
        )

    dfs = []
    for path in query_paths:
        df = pd.read_json(path, lines=True)
        if "query_set" not in df.columns:
            query_id = path.stem.split("_")[-1]
            df["query_set"] = query_id
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def main() -> None:
    args = parse_args()

    test_df = load_test_data(args)

    runner = DifferencingAttackRunner(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        dataset_name=args.dataset,
    )

    results = runner.run(
        test_df=test_df,
        n_attack_instances=args.n_attack_instances,
        n_runs=args.n_runs,
        background_sizes=args.background_sizes,
        vote_budgets=args.vote_budgets,
        base_seed=args.base_seed,
        n_workers=args.n_workers,
        quantifiers=args.quantifiers,
        model_suffix=args.dataset,
        save_individual_runs=args.save_individual_runs,
    )

    runner.save_results(
        results,
        args.reports_dir / f"{args.dataset}_adversarial.pkl",
    )

    summary = runner.summarize(results)
    print(summary)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import warnings
import multiprocessing as mp
from pathlib import Path

warnings.filterwarnings("ignore", message=r".*'where' used without 'out'.*")

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


def load_test_data(
    args: argparse.Namespace, dirs: dict[str, Path]
) -> pd.DataFrame:
    if args.dataset == "adult":
        return pd.read_csv(dirs["data"] / "adult_D3.csv")

    query_paths = sorted(dirs["data"].glob("trec_test_query_*.jsonl"))
    if not query_paths:
        raise FileNotFoundError(
            f"No TREC query files found matching 'trec_test_query_*.jsonl' in {dirs["data"]}"
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

    project_root = Path(__file__).resolve().parents[1]
    dirs = {
        folder: project_root / folder
        for folder in ["data", "models", "reports"]
    }

    test_df = load_test_data(args, dirs)

    runner = DifferencingAttackRunner(
        data_dir=dirs["data"],
        models_dir=dirs["models"],
        reports_dir=dirs["reports"],
        dataset_name=args.dataset,
    )

    results = runner.run(
        test_df=test_df,
        n_attack_instances=args.n_attack_instances,
        n_runs=args.n_runs,
        background_sizes=args.background_sizes,
        vote_budgets=args.vote_budgets,
        n_workers=args.n_workers,
        quantifiers=args.quantifiers,
        model_suffix=args.dataset,
        save_individual_runs=args.save_individual_runs,
    )

    runner.save_results(
        results,
        dirs["reports"] / f"{args.dataset}_adversarial.pkl",
    )

    summary = runner.summarize(results)
    print(summary)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import warnings
import multiprocessing as mp
from pathlib import Path
from typing import Sequence

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


def load_test_data(dataset: str, data_dir: Path) -> pd.DataFrame:
    if dataset == "adult":
        return pd.read_csv(data_dir / "adult_D3.csv")

    query_paths = sorted(data_dir.glob("trec_test_query_*.jsonl"))
    if not query_paths:
        raise FileNotFoundError(
            f"No TREC query files found matching 'trec_test_query_*.jsonl' "
            f"in {data_dir}"
        )

    dfs = []
    for path in query_paths:
        df = pd.read_json(path, lines=True)
        if "query_set" not in df.columns:
            query_id = path.stem.split("_")[-1]
            df["query_set"] = query_id
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def run(
    dataset: str,
    data_dir: Path | str,
    models_dir: Path | str,
    reports_dir: Path | str,
    quantifiers: Sequence[str] | None = None,
    n_attack_instances: int = 500,
    n_runs: int = 5,
    background_sizes: Sequence[int] = (1, 10, 100),
    vote_budgets: Sequence[int] = (1, 10, 100),
    base_seed: int = 0,
    n_workers: int = 1,
    save_individual_runs: bool = False,
    print_summary: bool = True,
) -> tuple[pd.DataFrame, Path]:
    """
    Programmatic entry point for the differencing-attack evaluation.

    Returns the full results DataFrame and the path of the saved pickle.
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

    test_df = load_test_data(dataset, data_dir)

    runner = DifferencingAttackRunner(
        data_dir=data_dir,
        models_dir=models_dir,
        reports_dir=reports_dir,
        dataset_name=dataset,
    )

    results = runner.run(
        test_df=test_df,
        n_attack_instances=n_attack_instances,
        n_runs=n_runs,
        background_sizes=list(background_sizes),
        vote_budgets=list(vote_budgets),
        base_seed=base_seed,
        n_workers=n_workers,
        quantifiers=list(quantifiers) if quantifiers else None,
        model_suffix=dataset,
        save_individual_runs=save_individual_runs,
    )

    output_path = reports_dir / f"{dataset}_adversarial.pkl"
    runner.save_results(results, output_path)

    if print_summary:
        print(runner.summarize(results))

    return results, output_path


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
        quantifiers=args.quantifiers,
        n_attack_instances=args.n_attack_instances,
        n_runs=args.n_runs,
        background_sizes=args.background_sizes,
        vote_budgets=args.vote_budgets,
        n_workers=args.n_workers,
        save_individual_runs=args.save_individual_runs,
    )


if __name__ == "__main__":
    main()

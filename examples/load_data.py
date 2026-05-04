from __future__ import annotations

import argparse
from pathlib import Path

from src.data import AdultDatasetLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare dataset files")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["adult", "trec"],
        required=True,
        help="Dataset to prepare.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / args.dataset
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "adult":
        loader = AdultDatasetLoader(data_dir=data_dir)
        loader.write_split_csvs()
        print(f"Adult split CSVs written to {data_dir}")

    elif args.dataset == "trec":
        print(
            f"Nothing to build for TREC. Place the unpacked TREC files in {data_dir}"
        )


if __name__ == "__main__":
    main()

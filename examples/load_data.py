from __future__ import annotations

import argparse
from pathlib import Path

from src.data import AdultDatasetLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch dataset and materialise the indexed CSV splits."
    )

    project_root = Path(__file__).resolve().parents[1]
    default_data_dir = project_root / "data"

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="Directory containing the dataset files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    loader = AdultDatasetLoader(data_dir=args.data_dir)
    loader.write_split_csvs()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from src.data import AdultDatasetLoader


def create_dirs(project_root: Path) -> dict[str, Path]:
    dirs = {}
    for subfolder in ["data", "models", "reports"]:
        (project_root / subfolder).mkdir(parents=True, exist_ok=True)
        dirs[subfolder] = project_root / subfolder
    return dirs


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dirs = create_dirs(project_root)
    loader = AdultDatasetLoader(data_dir=dirs["data"])
    loader.write_split_csvs()


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ADULT_OPENML_NAME = "adult"
ADULT_OPENML_VERSION = 2
ADULT_SPLIT_NAMES = ("D1", "D2", "D3")


class AdultDatasetLoader:
    """
    Fetch UCI Adult from OpenML and split.
    """

    def __init__(
        self,
        data_dir: Path | str,
        dataset_name: str = ADULT_OPENML_NAME,
        dataset_version: int = ADULT_OPENML_VERSION,
    ) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory {self.data_dir} does not exist."
            )
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version

    def load_full_dataframe(self) -> pd.DataFrame:
        """
        Load the complete Adult dataframe.
        """
        from sklearn.datasets import fetch_openml

        dataset = fetch_openml(
            name=self.dataset_name,
            version=self.dataset_version,
            as_frame=True,
        )
        target_name = dataset.target.name or "class"
        dataframe = pd.concat(
            [dataset.data, dataset.target.rename(target_name)], axis=1
        )
        return dataframe.dropna().reset_index(drop=True)

    def load_split_indices(self, split_name: str) -> np.ndarray:
        """
        Load the row indices for one split.
        """
        indices_path = self.data_dir / f"adult_{split_name}.indices"
        if not indices_path.exists():
            raise FileNotFoundError(f"Missing indices file: {indices_path}")
        return np.loadtxt(indices_path, dtype=int, ndmin=1)

    def build_splits(
        self, dataframe: pd.DataFrame | None = None
    ) -> dict[str, pd.DataFrame]:
        """
        Slice the full Adult dataframe into indexed splits.
        """
        dataframe = (
            self.load_full_dataframe() if dataframe is None else dataframe
        )
        splits: dict[str, pd.DataFrame] = {}

        for split_name in ADULT_SPLIT_NAMES:
            indices = self.load_split_indices(split_name)
            if len(indices) == 0:
                raise ValueError(f"Indices for split {split_name} are empty.")
            if indices.min() < 0 or indices.max() >= len(dataframe):
                raise IndexError(
                    f"Indices for split {split_name} are out of bounds for "
                    f"a dataframe with {len(dataframe)} rows."
                )
            splits[split_name] = dataframe.iloc[indices].reset_index(drop=True)

        return splits

    def write_split_csvs(self, dataframe: pd.DataFrame | None = None) -> None:
        """
        Write `adult_D1.csv`, `adult_D2.csv`, and `adult_D3.csv` to `data_dir`.
        """
        splits = self.build_splits(dataframe=dataframe)

        for split_name, split_df in splits.items():
            split_df.to_csv(
                self.data_dir / f"adult_{split_name}.csv", index=False
            )

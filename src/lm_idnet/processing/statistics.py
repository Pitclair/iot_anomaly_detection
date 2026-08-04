"""Descriptive statistics for processed packet windows."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from lm_idnet.exceptions import DataValidationError

from .categories import validate_categories
from .storage import load_processed_dataset


class Statistics:
    def __init__(self, processed_dir, categories, dates=None):
        self.processed_dir = Path(processed_dir)
        self.categories = validate_categories(categories)
        self.capture_ids = set(dates or [])

    def _dataset_paths(self) -> list[Path]:
        paths = sorted(self.processed_dir.glob("*.json"))
        if not paths and self.processed_dir.is_dir():
            for subdirectory in self.processed_dir.iterdir():
                if subdirectory.is_dir():
                    paths.extend(sorted(subdirectory.glob("*.json")))

        if not self.capture_ids:
            return paths
        return [path for path in paths if path.stem in self.capture_ids]

    @staticmethod
    def mean(matrix):
        return np.mean(matrix, axis=0) if matrix.size > 0 else np.array([])

    @staticmethod
    def variance(matrix):
        if matrix.size == 0:
            return np.array([])
        if matrix.shape[0] < 2:
            return np.full(matrix.shape[1], np.nan)
        return np.var(matrix, axis=0, ddof=1)

    @staticmethod
    def dispersion(variance, mean):
        with np.errstate(divide="ignore", invalid="ignore"):
            dispersion = np.true_divide(variance, mean)
            dispersion[~np.isfinite(dispersion)] = 0
        return dispersion

    @staticmethod
    def _optional_float(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    def process_file(self, file_path: str | Path) -> dict[str, Any]:
        dataset = load_processed_dataset(file_path)
        if not dataset.windows:
            raise DataValidationError(f"no windows found in {file_path}")
        if any(window.categories != self.categories for window in dataset.windows):
            raise DataValidationError(
                f"processed categories do not match configuration: {file_path}"
            )

        observed_windows = [
            window for window in dataset.windows if window.state != "missing"
        ]
        if not observed_windows:
            raise DataValidationError(f"no observed windows found in {file_path}")

        matrix = np.asarray(
            [window.counts for window in observed_windows],
            dtype=int,
        )
        means = self.mean(matrix)
        variances = self.variance(matrix)
        dispersions = self.dispersion(variances, means)
        silent_count = int((matrix.sum(axis=1) == 0).sum())

        category_statistics = []
        for index, category in enumerate(self.categories):
            category_statistics.append(
                {
                    "category": category,
                    "mean": float(means[index]),
                    "variance": self._optional_float(variances[index]),
                    "dispersion_index": float(dispersions[index]),
                }
            )

        return {
            "capture_id": dataset.metadata.capture_id,
            "partition": dataset.metadata.partition,
            "total_instances": int(matrix.shape[0]),
            "silent_windows": {
                "count": silent_count,
                "percent": 100.0 * silent_count / matrix.shape[0],
            },
            "categories": category_statistics,
        }

    def build_report(self) -> dict[str, Any]:
        paths = self._dataset_paths()
        if not paths:
            raise DataValidationError(
                f"no processed JSON files found in {self.processed_dir}"
            )
        return {
            "report_type": "capture_statistics",
            "captures": [self.process_file(path) for path in paths],
        }

    def write_report(self, output_path: str | Path) -> dict[str, Any]:
        report = self.build_report()
        report_path = Path(output_path)
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        except OSError as error:
            raise DataValidationError(
                f"cannot save statistics report: {report_path}"
            ) from error
        return report

"""Descriptive statistics for processed packet windows."""

import logging
from pathlib import Path

import numpy as np
from tabulate import tabulate

from lm_idnet.exceptions import DataValidationError

from .categories import validate_categories
from .storage import load_processed_dataset

logger = logging.getLogger(__name__)


class Statistics:
    def __init__(self, json_dir, categories, dates=None):
        self.json_dir = Path(json_dir)
        self.categories = validate_categories(categories)
        self.dates = dates if dates is not None else []
        self.json_paths = self._collect_json_files()

    def _collect_json_files(self):
        json_files = sorted(self.json_dir.glob("*.json"))
        if not json_files:
            for subdir in self.json_dir.iterdir():
                if subdir.is_dir():
                    json_files.extend(sorted(subdir.glob("*.json")))

        if not self.dates:
            return [str(path) for path in json_files]
        return [
            str(path)
            for path in json_files
            if any(date in str(path) for date in self.dates)
        ]

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

    def process_all(self):
        if not self.json_paths:
            logger.warning("No processed JSON files found in %s", self.json_dir)
            return
        for file_path in self.json_paths:
            self.process_file(file_path)

    def process_file(self, file_path):
        try:
            dataset = load_processed_dataset(file_path)
        except DataValidationError as error:
            print(f"[ERROR] Invalid processed dataset {file_path}: {error}")
            return
        if not dataset.windows:
            print(f"[ERROR] No windows found in {file_path}.")
            return
        if dataset.windows[0].categories != self.categories:
            raise ValueError(
                f"processed categories {dataset.windows[0].categories} do not match "
                f"configured categories {self.categories}"
            )
        if any(window.categories != self.categories for window in dataset.windows):
            raise ValueError("processed windows contain inconsistent category orders")

        observed_windows = [
            window for window in dataset.windows if window.state != "missing"
        ]
        if not observed_windows:
            print(f"[ERROR] No observed windows found in {file_path}.")
            return
        matrix = np.asarray(
            [window.counts for window in observed_windows],
            dtype=int,
        )
        means = self.mean(matrix)
        variances = self.variance(matrix)
        dispersions = self.dispersion(variances, means)
        silent_rows = (matrix.sum(axis=1) == 0).sum()
        silent_pct = 100 * silent_rows / matrix.shape[0] if matrix.shape[0] > 0 else 0
        table = [
            [proto, f"{means[i]:.2f}", f"{variances[i]:.2f}", f"{dispersions[i]:.2f}"]
            for i, proto in enumerate(self.categories)
        ]
        headers = ["Protocol", "Mean", "Variance", "Dispersion Index"]
        report = tabulate(table, headers, tablefmt="github")
        print(f"# Overdispersion Report for {file_path}\n")
        print(f"**Total Instances (R):** {matrix.shape[0]}")
        print(f"**Silent Windows:** {silent_rows} ({silent_pct:.1f}%)\n")
        print(report)

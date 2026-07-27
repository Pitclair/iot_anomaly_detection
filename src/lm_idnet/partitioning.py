"""Purpose-scoped access to frozen temporal capture partitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from lm_idnet.config import AppConfig
from lm_idnet.exceptions import DataValidationError

PartitionName = Literal["fit", "calibration", "development_test", "final_test"]


@dataclass(frozen=True)
class PartitionSelection:
    """A named immutable selection of whole capture identifiers."""

    name: PartitionName
    capture_ids: tuple[str, ...]


def all_capture_ids(config: AppConfig) -> tuple[str, ...]:
    """Return all captures for ingestion-only operations."""
    return config.ingest.partitions.all_capture_ids()


def partition_name_for_capture(
    config: AppConfig,
    capture_id: str,
) -> PartitionName:
    """Resolve one whole capture to exactly one configured partition."""
    partitions = config.ingest.partitions
    for name in ("fit", "calibration", "development_test", "final_test"):
        if capture_id in getattr(partitions, name):
            return name
    raise DataValidationError(f"capture is not assigned to a partition: {capture_id}")


def fit_partition_for_training(config: AppConfig) -> PartitionSelection:
    """Expose only fit captures to model-training code."""
    return PartitionSelection("fit", config.ingest.partitions.fit)


def calibration_partition_for_threshold(config: AppConfig) -> PartitionSelection:
    """Expose only calibration captures to threshold-fitting code."""
    return PartitionSelection("calibration", config.ingest.partitions.calibration)


def development_partition_for_evaluation(config: AppConfig) -> PartitionSelection:
    return PartitionSelection(
        "development_test",
        config.ingest.partitions.development_test,
    )


def final_partition_for_locked_evaluation(config: AppConfig) -> PartitionSelection:
    """Expose final-test IDs only at the explicitly locked evaluation boundary."""
    return PartitionSelection("final_test", config.ingest.partitions.final_test)

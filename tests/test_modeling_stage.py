"""Tests for the deliberately incomplete model-initialization stage."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from lm_idnet.exceptions import DataValidationError
from lm_idnet.models.modeling_stage import (
    create_initial_alpha,
    initialize_modeling_stage,
    load_training_matrix,
)
from lm_idnet.processing.schemas import Metadata, ProcessedDataset, WindowRecord
from lm_idnet.processing.storage import save_processed_dataset

pytestmark = pytest.mark.unit


def write_fit_datasets(config, processed_root, *, wrong_partition=False) -> None:
    categories = config.ingest.categories
    processed_dir = processed_root / config.ingest.dataset_folder
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)

    for index, capture_id in enumerate(config.ingest.partitions.fit):
        windows = (
            WindowRecord(
                start_utc=start,
                end_utc=start + timedelta(minutes=10),
                categories=categories,
                counts=(index + 1, 2, 3, 4),
                state="observed",
            ),
            WindowRecord(
                start_utc=start + timedelta(minutes=10),
                end_utc=start + timedelta(minutes=20),
                categories=categories,
                counts=(0, 0, 0, 0),
                state="observed-silent",
            ),
            WindowRecord(
                start_utc=start + timedelta(minutes=20),
                end_utc=start + timedelta(minutes=30),
                categories=categories,
                counts=None,
                state="missing",
            ),
        )
        partition = "calibration" if wrong_partition and index == 0 else "fit"
        dataset = ProcessedDataset(
            metadata=Metadata(
                device_id=config.ingest.dataset_folder,
                capture_id=capture_id,
                partition=partition,
                date=capture_id,
                file_source=f"{capture_id}.pcap",
            ),
            windows=windows,
        )
        save_processed_dataset(dataset, processed_dir / f"{capture_id}.json")


def test_load_training_matrix_uses_only_observed_fit_windows(
    tmp_path,
    config_factory,
) -> None:
    config = config_factory(ingest={"processed_root": tmp_path})
    write_fit_datasets(config, tmp_path)

    training = load_training_matrix(config)

    capture_count = len(config.ingest.partitions.fit)
    assert training.counts.shape == (capture_count * 2, 4)
    assert training.counts.dtype == np.int64
    assert training.capture_ids == config.ingest.partitions.fit
    assert training.silent_window_count == capture_count
    assert training.missing_window_count == capture_count


def test_initialization_reports_that_no_model_was_fitted(
    tmp_path,
    config_factory,
) -> None:
    config = config_factory(ingest={"processed_root": tmp_path})
    write_fit_datasets(config, tmp_path)

    result = initialize_modeling_stage(config)

    assert result["matrix_shape"] == [12, 4]
    assert result["initial_alpha"] == [1.0, 1.0, 1.0, 1.0]
    assert result["initial_concentration"] == 4.0
    assert result["initial_psi"] == 0.25
    assert result["log_likelihood_backend"] == "scipy"
    assert np.isfinite(result["initial_log_likelihood"])
    assert result["fitting_performed"] is False
    assert result["model_saved"] is False


def test_training_rejects_dataset_with_wrong_partition(
    tmp_path,
    config_factory,
) -> None:
    config = config_factory(ingest={"processed_root": tmp_path})
    write_fit_datasets(config, tmp_path, wrong_partition=True)

    with pytest.raises(DataValidationError, match="not marked as fit"):
        load_training_matrix(config)


def test_initial_alpha_requires_multiple_categories() -> None:
    with pytest.raises(ValueError, match="at least two"):
        create_initial_alpha(1)

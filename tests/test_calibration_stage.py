"""Tests for calibration-window scoring and threshold persistence."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from lm_idnet.algorithms.dirichlet import DirichletFit
from lm_idnet.artifacts import artifact_fingerprint, load_artifact
from lm_idnet.exceptions import DataValidationError
from lm_idnet.models.calibration_stage import (
    calibrate_threshold,
    load_calibration_windows,
)
from lm_idnet.models.modeling_stage import save_model
from lm_idnet.processing.schemas import Metadata, ProcessedDataset, WindowRecord
from lm_idnet.processing.storage import save_processed_dataset

pytestmark = pytest.mark.unit


def write_calibration_datasets(config, *, wrong_partition: bool = False) -> None:
    """Write one observed, silent, and missing window per capture."""
    processed_dir = (
        config.ingest.processed_root / config.ingest.dataset_folder
    )
    categories = config.ingest.categories
    first_start = datetime(2020, 10, 14, tzinfo=timezone.utc)

    for index, capture_id in enumerate(config.ingest.partitions.calibration):
        start = first_start + timedelta(days=index)
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
        partition = (
            "fit" if wrong_partition and index == 0 else "calibration"
        )
        dataset = ProcessedDataset(
            metadata=Metadata(
                device_id=config.ingest.device_name,
                capture_id=capture_id,
                partition=partition,
                date=capture_id,
                file_source=f"{capture_id}.pcap",
            ),
            windows=windows,
        )
        save_processed_dataset(
            dataset,
            processed_dir / f"{capture_id}.json",
        )


def write_model(config) -> None:
    alpha = np.asarray([1.0, 2.0, 3.0, 4.0])
    fit = DirichletFit(
        initial_alpha=alpha.copy(),
        alpha=alpha,
        concentration=10.0,
        psi=0.1,
        iterations=1,
        converged=True,
        initial_log_likelihood=-10.0,
        final_log_likelihood=-9.0,
    )
    save_model(
        path=config.outputs.model_path,
        categories=config.ingest.categories,
        fit=fit,
        training_capture_ids=config.ingest.partitions.fit,
        log_likelihood_backend="scipy",
        tolerance=config.estimator.tolerance_delta,
        max_iterations=config.estimator.max_iterations,
        duration_seconds=0.1,
    )


def test_load_calibration_windows_keeps_observed_and_silent_windows(
    tmp_path,
    config_factory,
) -> None:
    config = config_factory(ingest={"processed_root": tmp_path})
    write_calibration_datasets(config)
    windows = load_calibration_windows(config, config.ingest.categories)

    assert len(windows) == len(config.ingest.partitions.calibration) * 2
    assert {window.state for window in windows} == {
        "observed",
        "observed-silent",
    }


def test_load_calibration_windows_rejects_wrong_partition(
    tmp_path,
    config_factory,
) -> None:
    config = config_factory(ingest={"processed_root": tmp_path})
    write_calibration_datasets(config, wrong_partition=True)
    with pytest.raises(DataValidationError, match="wrong partition"):
        load_calibration_windows(config, config.ingest.categories)


@pytest.mark.parametrize("score_type", ["raw", "normalized"])
def test_calibrate_threshold_scores_windows_and_saves_artifact(
    tmp_path,
    config_factory,
    score_type: str,
) -> None:
    model_path = tmp_path / "model.json"
    threshold_path = tmp_path / "threshold.json"
    config = config_factory(
        ingest={"processed_root": tmp_path},
        calibration={
            "quantile": 0.25,
            "minimum_samples": 4,
            "score_type": score_type,
        },
        outputs={
            "model_path": model_path,
            "threshold_path": threshold_path,
        },
    )
    write_model(config)
    write_calibration_datasets(config)

    result = calibrate_threshold(config)
    threshold = load_artifact(
        threshold_path,
        expected_type="threshold",
    )

    assert result["window_count"] == 4
    assert result["score_type"] == threshold.score_type == score_type
    assert result["threshold"] == pytest.approx(threshold.threshold)
    assert threshold.model_fingerprint == artifact_fingerprint(
        load_artifact(model_path, expected_type="model")
    )
    assert threshold.calibration_capture_ids == (
        config.ingest.partitions.calibration
    )
    assert threshold.calibration_window_count == 4
    assert threshold.quantile == 0.25
    assert threshold.quantile_method == "linear"
    assert threshold.score_minimum <= threshold.score_median
    assert threshold.score_median <= threshold.score_maximum
    assert threshold.score_iqr > 0


def test_calibrate_threshold_requires_minimum_samples(
    tmp_path,
    config_factory,
) -> None:
    config = config_factory(
        ingest={"processed_root": tmp_path},
        calibration={"minimum_samples": 5},
        outputs={
            "model_path": tmp_path / "model.json",
            "threshold_path": tmp_path / "threshold.json",
        },
    )
    write_model(config)
    write_calibration_datasets(config)

    with pytest.raises(DataValidationError, match="at least 5 windows"):
        calibrate_threshold(config)

    assert not config.outputs.threshold_path.exists()

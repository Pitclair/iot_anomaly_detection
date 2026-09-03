"""Tests for Dirichlet-multinomial model training."""

from datetime import datetime, timedelta, timezone
import logging

import numpy as np
import pytest

from lm_idnet.artifacts import load_artifact
from lm_idnet.exceptions import ConvergenceError, DataValidationError
from lm_idnet.models import modeling_stage
from lm_idnet.models.modeling_stage import (
    load_training_matrix,
    train_model,
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
                device_id=config.ingest.device_id,
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


def test_training_fits_and_saves_model(
    tmp_path,
    config_factory,
    caplog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.json"
    config = config_factory(
        precision_digits=8,
        ingest={"processed_root": tmp_path},
        estimator={"tolerance_delta": 1e-4, "max_iterations": 2000},
        outputs={"model_path": model_path},
    )
    write_fit_datasets(config, tmp_path)
    precision_values = []
    original_create_estimator = modeling_stage.create_estimator

    def create_estimator_with_recording(estimator_config, precision_digits=6):
        precision_values.append(precision_digits)
        return original_create_estimator(estimator_config, precision_digits)

    monkeypatch.setattr(
        modeling_stage,
        "create_estimator",
        create_estimator_with_recording,
    )

    with caplog.at_level(logging.INFO):
        result = train_model(config)
    saved = load_artifact(model_path, expected_type="model").model_dump(mode="json")

    assert result["stage"] == "dirichlet_multinomial_training"
    assert precision_values == [8]
    assert result["matrix_shape"] == [12, 4]
    assert result["initial_alpha"] == pytest.approx([2.8, 1.6, 2.4, 3.2])
    assert (np.asarray(result["alpha"]) > 0).all()
    assert result["log_likelihood_backend"] == "lm"
    assert result["final_log_likelihood"] > result["initial_log_likelihood"]
    assert result["model_path"] == str(model_path)
    assert saved["alpha"] == pytest.approx(result["alpha"])
    assert saved["categories"] == list(config.ingest.categories)
    assert saved["concentration"] == pytest.approx(result["concentration"])
    assert saved["mean_probabilities"] == pytest.approx(
        np.asarray(result["alpha"]) / result["concentration"]
    )
    assert saved["psi"] == pytest.approx(result["psi"])
    assert saved["training_capture_ids"] == list(config.ingest.partitions.fit)
    assert saved["log_likelihood_backend"] == "lm"
    assert saved["fit_diagnostics"] == {
        "initial_alpha": pytest.approx(result["initial_alpha"]),
        "iterations": result["iterations"],
        "converged": True,
        "tolerance": config.estimator.tolerance_delta,
        "max_iterations": config.estimator.max_iterations,
        "initial_log_likelihood": result["initial_log_likelihood"],
        "final_log_likelihood": result["final_log_likelihood"],
        "duration_seconds": pytest.approx(result["duration_seconds"]),
    }
    assert "Model converged" in caplog.text
    assert "Saved model" in caplog.text


def test_training_does_not_save_unconverged_model(
    tmp_path,
    config_factory,
) -> None:
    model_path = tmp_path / "model.json"
    config = config_factory(
        ingest={"processed_root": tmp_path},
        estimator={"max_iterations": 1},
        outputs={"model_path": model_path},
    )
    write_fit_datasets(config, tmp_path)

    with pytest.raises(ConvergenceError, match="did not converge"):
        train_model(config)

    assert not model_path.exists()


def test_training_rejects_dataset_with_wrong_partition(
    tmp_path,
    config_factory,
) -> None:
    config = config_factory(ingest={"processed_root": tmp_path})
    write_fit_datasets(config, tmp_path, wrong_partition=True)

    with pytest.raises(DataValidationError, match="not marked as fit"):
        load_training_matrix(config)

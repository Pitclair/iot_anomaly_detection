"""Tests for development-window anomaly scoring."""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from lm_idnet.algorithms.dirichlet import DirichletFit
from lm_idnet.artifacts import load_model_for_scoring
from lm_idnet.exceptions import ArtifactCompatibilityError
from lm_idnet.models.calibration_stage import save_threshold
from lm_idnet.models.modeling_stage import save_model
from lm_idnet.models.scoring_stage import score_windows
from lm_idnet.processing.schemas import Metadata, ProcessedDataset, WindowRecord
from lm_idnet.processing.storage import save_processed_dataset

pytestmark = pytest.mark.unit


def write_model(config) -> dict:
    alpha = np.asarray([1.0, 2.0, 3.0, 4.0])
    save_model(
        config.outputs.model_path,
        config.ingest.categories,
        DirichletFit(alpha.copy(), alpha, 10.0, 0.1, 1, True, -10.0, -9.0),
        config.ingest.partitions.fit,
        "scipy",
        config.estimator.tolerance_delta,
        config.estimator.max_iterations,
        0.1,
    )
    return load_model_for_scoring(config.outputs.model_path)


def write_threshold(config, model_checksum: str, score_type: str) -> None:
    start = datetime(2020, 10, 14, tzinfo=timezone.utc)
    save_threshold(
        config.outputs.threshold_path,
        {
            "artifact_type": "threshold",
            "schema_version": "1.2.0",
            "model_checksum": model_checksum,
            "score_type": score_type,
            "quantile": 0.01,
            "threshold": 0.0,
            "calibration_capture_ids": config.ingest.partitions.calibration,
            "calibration_window_count": 1,
            "calibration_start_utc": start,
            "calibration_end_utc": start + timedelta(minutes=10),
            "quantile_method": "linear",
            "score_minimum": -1.0,
            "score_median": -0.5,
            "score_maximum": 0.0,
        },
    )


def write_development_captures(config) -> None:
    output_dir = config.ingest.processed_root / config.ingest.dataset_folder
    for index, capture_id in enumerate(config.ingest.partitions.development_test):
        start = datetime(2020, 10, 16 + index, tzinfo=timezone.utc)
        windows = (
            WindowRecord(
                start_utc=start,
                end_utc=start + timedelta(minutes=10),
                categories=config.ingest.categories,
                counts=(1, 2, 3, 4),
                state="observed",
            ),
            WindowRecord(
                start_utc=start + timedelta(minutes=10),
                end_utc=start + timedelta(minutes=20),
                categories=config.ingest.categories,
                counts=(0, 0, 0, 0),
                state="observed-silent",
            ),
            WindowRecord(
                start_utc=start + timedelta(minutes=20),
                end_utc=start + timedelta(minutes=30),
                categories=config.ingest.categories,
                counts=None,
                state="missing",
            ),
        )
        save_processed_dataset(
            ProcessedDataset(
                metadata=Metadata(
                    device_id=config.ingest.dataset_folder,
                    capture_id=capture_id,
                    partition="development_test",
                    date=capture_id,
                    file_source=f"{capture_id}.pcap",
                ),
                windows=windows,
            ),
            output_dir / f"{capture_id}.json",
        )


def test_score_windows_writes_observed_and_silent_results(
    tmp_path,
    config_factory,
) -> None:
    config = config_factory(
        ingest={"processed_root": tmp_path},
        outputs={
            "model_path": tmp_path / "model.json",
            "threshold_path": tmp_path / "threshold.json",
            "events_path": tmp_path / "events.jsonl",
        },
    )
    model = write_model(config)
    write_threshold(config, model["checksum"], "raw_log_probability")
    write_development_captures(config)

    summary = score_windows(config)
    results = [
        json.loads(line)
        for line in config.outputs.events_path.read_text(encoding="utf-8").splitlines()
    ]

    assert summary["partition"] == "development_test"
    assert summary["window_count"] == len(results) == 4
    assert summary["anomaly_count"] == 2
    assert all(
        set(result)
        == {
            "capture_id",
            "window_start_utc",
            "window_end_utc",
            "counts",
            "score",
            "threshold",
            "is_anomaly",
            "model_checksum",
            "threshold_checksum",
        }
        for result in results
    )
    assert {result["capture_id"] for result in results} == set(
        config.ingest.partitions.development_test
    )
    assert all(result["counts"] is not None for result in results)
    assert all(
        result["is_anomaly"] == (result["score"] < result["threshold"])
        for result in results
    )
    assert all(result["model_checksum"] == model["checksum"] for result in results)
    assert all(
        result["threshold_checksum"] == summary["threshold_checksum"]
        for result in results
    )
    assert all(
        result["is_anomaly"] is False
        for result in results
        if sum(result["counts"]) == 0
    )


@pytest.mark.parametrize(
    ("model_checksum", "score_type", "message"),
    [
        ("0" * 64, "raw_log_probability", "model checksum"),
        (None, "normalized_log_probability", "score type"),
    ],
)
def test_score_windows_rejects_incompatible_threshold(
    tmp_path,
    config_factory,
    model_checksum,
    score_type,
    message,
) -> None:
    config = config_factory(
        outputs={
            "model_path": tmp_path / "model.json",
            "threshold_path": tmp_path / "threshold.json",
            "events_path": tmp_path / "events.jsonl",
        }
    )
    model = write_model(config)
    write_threshold(config, model_checksum or model["checksum"], score_type)

    with pytest.raises(ArtifactCompatibilityError, match=message):
        score_windows(config)

    assert not config.outputs.events_path.exists()

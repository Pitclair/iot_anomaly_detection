import json
from copy import deepcopy
from typing import Any

import pytest

from lm_idnet.artifacts import Artifact, load_artifact, validate_artifact
from lm_idnet.exceptions import ArtifactCompatibilityError

pytestmark = pytest.mark.unit

HASH = "0" * 64


def artifact_examples() -> dict[str, dict[str, Any]]:
    return {
        "model": {
            "artifact_type": "model",
            "categories": ["tcp", "udp", "ssdp", "arp"],
            "alpha": [1.0, 2.0, 3.0, 4.0],
            "concentration": 10.0,
            "mean_probabilities": [0.1, 0.2, 0.3, 0.4],
            "psi": 0.1,
            "training_capture_ids": ["capture-001", "capture-002"],
            "log_likelihood_backend": "scipy",
            "fit_diagnostics": {
                "initial_alpha": [2.5, 2.5, 2.5, 2.5],
                "iterations": 25,
                "converged": True,
                "tolerance": 1e-6,
                "max_iterations": 100,
                "initial_log_likelihood": -50.0,
                "final_log_likelihood": -40.0,
                "duration_seconds": 0.25,
            },
        },
        "threshold": {
            "artifact_type": "threshold",
            "score_type": "raw",
            "quantile": 0.01,
            "threshold": -42.5,
            "calibration_capture_ids": ["capture-003", "capture-004"],
            "calibration_window_count": 288,
            "calibration_start_utc": "2020-01-03T00:00:00Z",
            "calibration_end_utc": "2020-01-05T00:00:00Z",
            "quantile_method": "linear",
            "score_minimum": -50.0,
            "score_median": -10.0,
            "score_maximum": -5.0,
        },
        "anomaly_event": {
            "artifact_type": "anomaly_event",
            "event_id": "event-001",
            "window_id": "window-001",
            "score": -50.0,
            "threshold": -42.5,
            "decision": True,
        },
        "experiment_manifest": {
            "artifact_type": "experiment_manifest",
            "run_id": "run-001",
            "command": "lm-idnet train",
            "code_commit": "abc123",
            "configuration_hash": HASH,
            "dataset_hash": HASH,
            "seed": 1001,
        },
    }


@pytest.mark.parametrize("artifact_type", artifact_examples())
def test_artifact_round_trip(artifact_type: str) -> None:
    artifact = validate_artifact(
        artifact_examples()[artifact_type],
        expected_type=artifact_type,
    )
    serialized = json.loads(artifact.model_dump_json())
    reparsed = validate_artifact(serialized, expected_type=artifact_type)

    assert isinstance(reparsed, Artifact.__args__)
    assert reparsed == artifact


def test_artifact_loads_from_disk(tmp_path) -> None:
    path = tmp_path / "model.json"
    path.write_text(json.dumps(artifact_examples()["model"]), encoding="utf-8")

    loaded = load_artifact(path, expected_type="model")

    assert loaded.alpha == (1.0, 2.0, 3.0, 4.0)


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    [
        ("concentration", 11.0, "concentration must equal"),
        ("mean_probabilities", [0.2, 0.2, 0.3, 0.3], "alpha / concentration"),
        ("psi", 0.2, "psi must be the reciprocal"),
    ],
)
def test_model_rejects_inconsistent_derived_parameters(
    field: str,
    invalid_value: object,
    message: str,
) -> None:
    model = deepcopy(artifact_examples()["model"])
    model[field] = invalid_value

    with pytest.raises(ArtifactCompatibilityError, match=message):
        validate_artifact(model, expected_type="model")


def test_model_rejects_partial_training_provenance() -> None:
    model = deepcopy(artifact_examples()["model"])
    model["fit_diagnostics"] = None

    with pytest.raises(
        ArtifactCompatibilityError,
        match="must be provided together",
    ):
        validate_artifact(model, expected_type="model")


def test_wrong_artifact_type_is_rejected() -> None:
    with pytest.raises(ArtifactCompatibilityError, match="expected threshold"):
        validate_artifact(
            artifact_examples()["model"],
            expected_type="threshold",
        )


@pytest.mark.parametrize("removed_field", ["schema_version", "checksum"])
def test_removed_metadata_is_not_accepted(removed_field: str) -> None:
    model = artifact_examples()["model"]
    model[removed_field] = "unused"

    with pytest.raises(ArtifactCompatibilityError, match="Extra inputs"):
        validate_artifact(model, expected_type="model")

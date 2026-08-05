import json
from copy import deepcopy
from typing import Any

import pytest

from lm_idnet.artifact_schemas import (
    CURRENT_SCHEMA_VERSION,
    LEGACY_SCHEMA_VERSION,
    PREVIOUS_SCHEMA_VERSION,
    Artifact,
    validate_versioned_artifact,
)
from lm_idnet.artifacts import artifact_checksum, load_typed_artifact
from lm_idnet.exceptions import ArtifactCompatibilityError

pytestmark = pytest.mark.unit

HASH = "0" * 64


def artifact_examples() -> dict[str, dict[str, Any]]:
    examples: dict[str, dict[str, Any]] = {
        "model": {
            "model_version": "model-001",
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
            "model_checksum": HASH,
            "score_type": "raw_log_probability",
            "quantile": 0.01,
            "threshold": -42.5,
        },
        "anomaly_event": {
            "event_id": "event-001",
            "window_id": "window-001",
            "score": -50.0,
            "threshold": -42.5,
            "decision": True,
            "model_version": "model-001",
        },
        "experiment_manifest": {
            "run_id": "run-001",
            "command": "lm-idnet train",
            "code_commit": "abc123",
            "configuration_hash": HASH,
            "dataset_hash": HASH,
            "seed": 1001,
        },
    }
    for artifact_type, artifact in examples.items():
        artifact["artifact_type"] = artifact_type
        artifact["schema_version"] = CURRENT_SCHEMA_VERSION
        artifact["checksum"] = artifact_checksum(artifact)
    return examples


@pytest.mark.parametrize("artifact_type", artifact_examples())
def test_current_schema_round_trip(artifact_type: str) -> None:
    raw = artifact_examples()[artifact_type]

    parsed = validate_versioned_artifact(raw, expected_type=artifact_type)
    serialized = json.loads(parsed.model_dump_json())
    reparsed = validate_versioned_artifact(
        serialized,
        expected_type=artifact_type,
    )

    assert isinstance(reparsed, type(parsed))
    assert reparsed == parsed


@pytest.mark.parametrize("artifact_type", artifact_examples())
def test_previous_supported_schema_migrates(artifact_type: str) -> None:
    previous = deepcopy(artifact_examples()[artifact_type])
    previous["schema_version"] = PREVIOUS_SCHEMA_VERSION
    previous["checksum"] = artifact_checksum(previous)

    migrated = validate_versioned_artifact(
        previous,
        expected_type=artifact_type,
    )

    assert isinstance(migrated, Artifact.__args__)
    assert migrated.schema_version == CURRENT_SCHEMA_VERSION
    assert migrated.artifact_type == artifact_type
    assert migrated.checksum == artifact_checksum(
        migrated.model_dump(mode="json")
    )


@pytest.mark.parametrize("artifact_type", artifact_examples())
def test_previous_artifact_loads_from_disk_through_migration(
    tmp_path: Any,
    artifact_type: str,
) -> None:
    previous = deepcopy(artifact_examples()[artifact_type])
    previous["schema_version"] = PREVIOUS_SCHEMA_VERSION
    previous["checksum"] = artifact_checksum(previous)
    path = tmp_path / f"{artifact_type}.json"
    path.write_text(json.dumps(previous), encoding="utf-8")

    loaded = load_typed_artifact(path, expected_type=artifact_type)

    assert loaded.schema_version == CURRENT_SCHEMA_VERSION
    assert loaded.artifact_type == artifact_type


@pytest.mark.parametrize("artifact_type", artifact_examples())
def test_legacy_schema_migrates(artifact_type: str) -> None:
    legacy = deepcopy(artifact_examples()[artifact_type])
    legacy["schema_version"] = LEGACY_SCHEMA_VERSION
    legacy.pop("artifact_type")
    legacy["checksum"] = artifact_checksum(legacy)

    migrated = validate_versioned_artifact(
        legacy,
        expected_type=artifact_type,
    )

    assert migrated.schema_version == CURRENT_SCHEMA_VERSION
    assert migrated.artifact_type == artifact_type


def test_minimal_previous_model_migrates_without_inventing_provenance() -> None:
    previous = {
        "artifact_type": "model",
        "schema_version": PREVIOUS_SCHEMA_VERSION,
        "model_version": "legacy-model",
        "categories": ["tcp", "udp", "ssdp", "arp"],
        "alpha": [1.0, 2.0, 3.0, 4.0],
        "checksum": HASH,
    }

    migrated = validate_versioned_artifact(previous, expected_type="model")

    assert migrated.concentration == pytest.approx(10.0)
    assert migrated.mean_probabilities == pytest.approx((0.1, 0.2, 0.3, 0.4))
    assert migrated.psi == pytest.approx(0.1)
    assert migrated.training_capture_ids == ()
    assert migrated.log_likelihood_backend is None
    assert migrated.fit_diagnostics is None


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
        validate_versioned_artifact(model, expected_type="model")


def test_model_rejects_partial_training_provenance() -> None:
    model = deepcopy(artifact_examples()["model"])
    model["fit_diagnostics"] = None

    with pytest.raises(
        ArtifactCompatibilityError,
        match="must be provided together",
    ):
        validate_versioned_artifact(model, expected_type="model")


@pytest.mark.parametrize("artifact_type", artifact_examples())
def test_unknown_future_major_is_rejected(artifact_type: str) -> None:
    future = deepcopy(artifact_examples()[artifact_type])
    future["schema_version"] = "2.0.0"

    with pytest.raises(
        ArtifactCompatibilityError,
        match="unsupported schema major version 2",
    ):
        validate_versioned_artifact(future, expected_type=artifact_type)


def test_wrong_artifact_type_is_rejected() -> None:
    raw = artifact_examples()["model"]

    with pytest.raises(ArtifactCompatibilityError, match="expected threshold"):
        validate_versioned_artifact(raw, expected_type="threshold")


@pytest.mark.parametrize("version", ["1", "1.0", "v1.0.0", "latest", 1])
def test_non_semantic_versions_are_rejected(version: object) -> None:
    raw = deepcopy(artifact_examples()["model"])
    raw["schema_version"] = version

    with pytest.raises(
        ArtifactCompatibilityError,
        match="invalid semantic schema version",
    ):
        validate_versioned_artifact(raw, expected_type="model")

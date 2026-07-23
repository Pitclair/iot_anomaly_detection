import json
from pathlib import Path

import pytest

from lm_idnet import cli
from lm_idnet.artifacts import artifact_checksum, load_model_for_scoring
from lm_idnet.config import load_config
from lm_idnet.exceptions import (
    ArtifactCompatibilityError,
    ArtifactIntegrityError,
    ConfigurationError,
    ConvergenceError,
    DataValidationError,
    IngestionError,
    LMIDNetError,
    NumericalPrecisionError,
    PolicyRejectionError,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "error",
    [
        ConfigurationError("invalid quantile"),
        IngestionError("capture is truncated"),
        DataValidationError("negative packet count"),
        NumericalPrecisionError("requested precision is unavailable"),
        ConvergenceError("maximum iterations exhausted"),
        ArtifactCompatibilityError("unsupported model schema"),
        ArtifactIntegrityError("model checksum mismatch"),
        PolicyRejectionError("candidate exceeds drift policy"),
    ],
)
def test_domain_exception_maps_to_cli_exit_code_and_message(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: LMIDNetError,
) -> None:
    def fail(_argv: object = None) -> None:
        raise error

    monkeypatch.setattr(cli, "run_command", fail)

    assert cli.main([]) == error.exit_code
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"ERROR [{error.error_code}]: {error}\n"


def test_missing_config_uses_configuration_failure(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.json"

    with pytest.raises(ConfigurationError, match="config not found"):
        load_config(missing)


def test_invalid_json_uses_configuration_failure(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="not readable JSON"):
        load_config(invalid)


def test_corrupt_model_checksum_fails_closed(tmp_path: Path) -> None:
    model_path = tmp_path / "model.json"
    model = {
        "schema_version": "1.0.0",
        "model_version": "test-model",
        "alpha": [1.0, 2.0, 3.0, 4.0],
    }
    model["checksum"] = artifact_checksum(model)
    model["alpha"][0] = 999.0
    model_path.write_text(json.dumps(model), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="checksum mismatch"):
        load_model_for_scoring(model_path)


def test_valid_model_checksum_is_accepted(tmp_path: Path) -> None:
    model_path = tmp_path / "model.json"
    model = {
        "schema_version": "1.0.0",
        "model_version": "test-model",
        "alpha": [1.0, 2.0, 3.0, 4.0],
    }
    model["checksum"] = artifact_checksum(model)
    model_path.write_text(json.dumps(model), encoding="utf-8")

    assert load_model_for_scoring(model_path) == model

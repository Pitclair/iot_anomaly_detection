import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from lm_idnet.cli import COMMANDS

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "config.json"


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "lm_idnet.cli", *arguments],
        capture_output=True,
        check=False,
        text=True,
    )


def write_empty_pcap(path: Path) -> None:
    path.write_bytes(struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))


def test_root_help_lists_every_stable_command() -> None:
    result = run_cli("--help")

    assert result.returncode == 0
    for command in COMMANDS:
        assert command in result.stdout


@pytest.mark.parametrize("command", COMMANDS)
def test_subcommand_help(command: str) -> None:
    result = run_cli(command, "--help")

    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--dry-run" in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize("command", COMMANDS)
def test_subcommand_rejects_missing_config(command: str) -> None:
    result = run_cli(command)

    assert result.returncode == 2
    assert "--config" in result.stderr


@pytest.mark.parametrize("command", COMMANDS)
def test_subcommand_rejects_invalid_config(
    command: str,
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")

    result = run_cli(command, "--config", str(invalid), "--dry-run")

    assert result.returncode == 2
    assert "configuration_error" in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("command", COMMANDS)
def test_subcommand_minimal_dry_run_succeeds(command: str) -> None:
    result = run_cli(
        command,
        "--config",
        str(CONFIG),
        "--dry-run",
    )

    assert result.returncode == 0
    assert json.loads(result.stdout) == {
        "command": command,
        "config": str(CONFIG),
        "status": "ready",
    }
    assert result.stderr == ""


def test_json_error_format_is_machine_readable(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    result = run_cli(
        "--error-format",
        "json",
        "train",
        "--config",
        str(missing),
    )

    assert result.returncode == 2
    assert json.loads(result.stderr) == {
        "error": {
            "code": "configuration_error",
            "message": f"config not found: {missing}",
        }
    }
    assert result.stdout == ""


@pytest.mark.parametrize(
    "command",
    ("train", "calibrate", "score", "evaluate", "adapt", "benchmark"),
)
def test_unimplemented_stage_fails_instead_of_claiming_success(
    command: str,
) -> None:
    result = run_cli(command, "--config", str(CONFIG))

    assert result.returncode == 9
    assert "command_unavailable_error" in result.stderr
    assert '"status": "completed"' not in result.stdout


def test_preprocess_inventory_only_writes_accepted_report(tmp_path: Path) -> None:
    raw_directory = tmp_path / "raw" / "camera"
    raw_directory.mkdir(parents=True)
    capture_ids = ["camera-2020-10-08", "camera-2020-10-09"]
    for capture_id in capture_ids:
        write_empty_pcap(raw_directory / f"{capture_id}.pcap")

    config_data = json.loads(CONFIG.read_text(encoding="utf-8"))
    config_data["ingest"].update(
        {
            "raw_root": str(tmp_path / "raw"),
            "dataset_folder": "camera",
            "training_dates": [capture_ids[0]],
            "testing_dates": [capture_ids[1]],
            "allowed_duplicate_captures": [
                {
                    "capture_ids": capture_ids,
                    "reason": "Empty integration-test fixtures",
                }
            ],
        }
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_data), encoding="utf-8")
    output_path = tmp_path / "capture_inventory.json"

    result = run_cli(
        "preprocess",
        "--config",
        str(config_path),
        "--inventory-only",
        "--inventory-output",
        str(output_path),
    )

    assert result.returncode == 0
    assert json.loads(result.stdout)["inventory"] == str(output_path)
    assert json.loads(output_path.read_text(encoding="utf-8"))["accepted"] is True


def test_preprocess_inventory_only_fails_when_required_capture_is_missing(
    tmp_path: Path,
) -> None:
    config_data = json.loads(CONFIG.read_text(encoding="utf-8"))
    config_data["ingest"].update(
        {
            "raw_root": str(tmp_path / "raw"),
            "dataset_folder": "camera",
            "training_dates": ["camera-2020-10-08"],
            "testing_dates": ["camera-2020-10-09"],
            "allowed_duplicate_captures": [],
        }
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_data), encoding="utf-8")
    output_path = tmp_path / "capture_inventory.json"

    result = run_cli(
        "preprocess",
        "--config",
        str(config_path),
        "--inventory-only",
        "--inventory-output",
        str(output_path),
    )

    assert result.returncode == 4
    assert "2 required capture(s) missing" in result.stderr
    assert json.loads(output_path.read_text(encoding="utf-8"))["missing_count"] == 2

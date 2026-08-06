import json
import logging
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from lm_idnet.cli import COMMANDS, LOG_DATE_FORMAT, LOG_FORMAT, UtcLogFormatter

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "config.json"


def test_log_formatter_uses_explicit_iso_utc_timestamp() -> None:
    formatter = UtcLogFormatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    record = logging.LogRecord(
        name="lm_idnet.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="message",
        args=(),
        exc_info=None,
    )
    record.created = 0.123
    record.msecs = 123

    assert (
        formatter.format(record)
        == "1970-01-01T00:00:00.123Z INFO lm_idnet.test: message"
    )


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["LM_IDNET_LOG_PATH"] = os.devnull
    return subprocess.run(
        [sys.executable, "-m", "lm_idnet.cli", *arguments],
        capture_output=True,
        check=False,
        text=True,
        env=environment,
    )


def write_empty_pcap(path: Path) -> None:
    path.write_bytes(struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))


def set_test_partitions(config_data: dict, capture_ids: list[str]) -> None:
    if len(capture_ids) != 4:
        raise ValueError("CLI tests require one capture per partition")
    config_data["ingest"]["partitions"] = {
        "fit": [capture_ids[0]],
        "calibration": [capture_ids[1]],
        "development_test": [capture_ids[2]],
        "final_test": [capture_ids[3]],
    }


def test_root_help_lists_every_stable_command() -> None:
    result = run_cli("--help")

    assert result.returncode == 0
    for command in COMMANDS:
        assert command in result.stdout
    assert "--verbose" in result.stdout


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
    ("calibrate", "score", "evaluate", "adapt", "benchmark"),
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
    capture_ids = [f"camera-2020-10-{day:02d}" for day in range(8, 12)]
    for capture_id in capture_ids:
        write_empty_pcap(raw_directory / f"{capture_id}.pcap")

    config_data = json.loads(CONFIG.read_text(encoding="utf-8"))
    set_test_partitions(config_data, capture_ids)
    config_data["ingest"].update(
        {
            "raw_root": str(tmp_path / "raw"),
            "dataset_folder": "camera",
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
    capture_ids = [f"camera-2020-10-{day:02d}" for day in range(8, 12)]
    set_test_partitions(config_data, capture_ids)
    config_data["ingest"].update(
        {
            "raw_root": str(tmp_path / "raw"),
            "dataset_folder": "camera",
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
    assert "4 required capture(s) missing" in result.stderr
    assert json.loads(output_path.read_text(encoding="utf-8"))["missing_count"] == 4


def test_preprocess_validation_only_parses_all_sources(tmp_path: Path) -> None:
    raw_directory = tmp_path / "raw" / "camera"
    raw_directory.mkdir(parents=True)
    capture_ids = [f"camera-2020-10-{day:02d}" for day in range(8, 12)]
    for capture_id in capture_ids:
        write_empty_pcap(raw_directory / f"{capture_id}.pcap")

    config_data = json.loads(CONFIG.read_text(encoding="utf-8"))
    set_test_partitions(config_data, capture_ids)
    config_data["ingest"].update(
        {
            "raw_root": str(tmp_path / "raw"),
            "dataset_folder": "camera",
            "allowed_duplicate_captures": [],
        }
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_data), encoding="utf-8")
    output_path = tmp_path / "capture_validation.json"

    result = run_cli(
        "preprocess",
        "--config",
        str(config_path),
        "--validate-captures-only",
        "--validation-output",
        str(output_path),
    )

    assert result.returncode == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["accepted"] is True
    assert report["summary"]["parsed_to_eof_count"] == 4


def test_preprocess_validation_only_rejects_truncated_source(tmp_path: Path) -> None:
    raw_directory = tmp_path / "raw" / "camera"
    raw_directory.mkdir(parents=True)
    capture_ids = [f"camera-2020-10-{day:02d}" for day in range(8, 12)]
    for capture_id in capture_ids[:3]:
        write_empty_pcap(raw_directory / f"{capture_id}.pcap")
    fixture_hex = (
        ROOT / "tests" / "data" / "quarantine" / "truncated_capture.hex"
    ).read_text(encoding="utf-8").strip()
    (raw_directory / f"{capture_ids[3]}.pcap").write_bytes(bytes.fromhex(fixture_hex))

    config_data = json.loads(CONFIG.read_text(encoding="utf-8"))
    set_test_partitions(config_data, capture_ids)
    config_data["ingest"].update(
        {
            "raw_root": str(tmp_path / "raw"),
            "dataset_folder": "camera",
            "allowed_duplicate_captures": [],
        }
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_data), encoding="utf-8")
    output_path = tmp_path / "capture_validation.json"

    result = run_cli(
        "preprocess",
        "--config",
        str(config_path),
        "--validate-captures-only",
        "--validation-output",
        str(output_path),
    )

    assert result.returncode == 4
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["accepted"] is False
    assert report["captures"][3]["fatal_error"]["code"] == "truncated_packet_data"


def test_preprocess_capture_modes_are_mutually_exclusive() -> None:
    result = run_cli(
        "preprocess",
        "--config",
        str(CONFIG),
        "--inventory-only",
        "--validate-captures-only",
    )

    assert result.returncode == 2
    assert "not allowed with argument" in result.stderr


def test_preprocess_fingerprint_only_writes_reproducible_manifest(
    tmp_path: Path,
) -> None:
    raw_directory = tmp_path / "raw" / "camera"
    raw_directory.mkdir(parents=True)
    capture_ids = [f"camera-2020-10-{day:02d}" for day in range(8, 12)]
    for index, capture_id in enumerate(capture_ids):
        (raw_directory / f"{capture_id}.pcap").write_bytes(
            f"capture-{index}".encode()
        )

    config_data = json.loads(CONFIG.read_text(encoding="utf-8"))
    set_test_partitions(config_data, capture_ids)
    config_data["ingest"].update(
        {
            "raw_root": str(tmp_path / "raw"),
            "dataset_folder": "camera",
            "allowed_duplicate_captures": [],
        }
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_data), encoding="utf-8")
    first_path = tmp_path / "first_fingerprint.json"
    second_path = tmp_path / "second_fingerprint.json"

    first = run_cli(
        "preprocess",
        "--config",
        str(config_path),
        "--fingerprint-only",
        "--fingerprint-output",
        str(first_path),
    )
    second = run_cli(
        "preprocess",
        "--config",
        str(config_path),
        "--fingerprint-only",
        "--fingerprint-output",
        str(second_path),
    )

    assert first.returncode == second.returncode == 0
    assert first_path.read_bytes() == second_path.read_bytes()
    assert json.loads(first.stdout)["dataset_version"] == json.loads(second.stdout)[
        "dataset_version"
    ]

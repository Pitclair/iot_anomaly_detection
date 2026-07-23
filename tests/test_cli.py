import json
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

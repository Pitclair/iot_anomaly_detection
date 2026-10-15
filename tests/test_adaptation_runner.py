"""Check experiment selection without invoking Docker or changing real results."""

from pathlib import Path
import subprocess

import pytest

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("arguments", [[], ["conservative"], ["legacy"], ["invalid"], ["legacy", "extra"]])
def test_adaptation_runner_selects_policy(tmp_path, arguments):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    runner = scripts / "run_adaptation_experiments.sh"
    runner.write_text((ROOT / "scripts" / runner.name).read_text())
    stub = scripts / "run.sh"
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$*"\n')
    stub.chmod(0o755)
    for path in (ROOT / "configs" / "experiments").rglob("*.json"):
        destination = tmp_path / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("{}")

    result = subprocess.run(
        ["sh", str(runner), *arguments],
        capture_output=True, text=True, check=False,
    )

    if arguments == ["invalid"] or len(arguments) > 1:
        assert result.returncode == 2
        assert "Usage:" in result.stderr
        assert not result.stdout
        return
    assert result.returncode == 0, result.stderr
    commands = result.stdout.splitlines()
    assert len(commands) == 12
    for adapt, evaluate in zip(commands[::2], commands[1::2]):
        assert adapt.startswith("adapt --config ")
        assert evaluate == adapt.replace("adapt --config ", "evaluate --config ", 1)
        assert ("/legacy/" in adapt) == (arguments == ["legacy"])

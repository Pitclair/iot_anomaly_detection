import subprocess
import sys

import pytest

pytestmark = pytest.mark.integration


def test_smoke_main():
    cmd = [sys.executable, "-m", "lm_idnet.cli", "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "IoT Anomaly Detection Baseline" in res.stdout

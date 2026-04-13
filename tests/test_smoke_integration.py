import subprocess
import sys
from pathlib import Path


def test_smoke_main():
    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(root / 'main.py'), 'model', '--config', str(root / 'configs' / 'config.json')]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert 'Model saved' in res.stdout


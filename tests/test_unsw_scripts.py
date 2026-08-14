import subprocess
from pathlib import Path

from lm_idnet.config import load_config


def test_samsung_camera_config() -> None:
    config = load_config("configs/unsw_samsung_camera.json")

    assert config.ingest.device_mac == "00:16:6c:ab:6b:88"
    assert len(config.ingest.partitions.all_capture_ids()) == 12


def test_unsw_downloader_rejects_unknown_profile() -> None:
    script = Path("unsw_iot_attack_pcaps/download_pcaps.sh")
    result = subprocess.run(
        [script, "unknown"], capture_output=True, text=True, check=False
    )

    assert result.returncode == 2
    assert "[chromecast|samsung-camera]" in result.stderr

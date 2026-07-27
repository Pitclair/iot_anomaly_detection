from copy import deepcopy
from pathlib import Path
from typing import Callable

import pytest

from lm_idnet.config import AppConfig
from lm_idnet.dataset_fingerprint import build_dataset_manifest
from lm_idnet.exceptions import DataValidationError

pytestmark = pytest.mark.unit


def fingerprint_config(
    config_factory: Callable[..., AppConfig],
    raw_root: Path,
) -> AppConfig:
    capture_ids = [f"camera-2020-10-{day:02d}" for day in range(8, 13)]
    return config_factory(
        ingest={
            "raw_root": str(raw_root),
            "dataset_folder": "camera",
            "partitions": {
                "fit": capture_ids[:2],
                "calibration": [capture_ids[2]],
                "development_test": [capture_ids[3]],
                "final_test": [capture_ids[4]],
            },
            "allowed_duplicate_captures": [],
        }
    )


@pytest.fixture
def fingerprint_inputs(
    tmp_path: Path,
    config_factory: Callable[..., AppConfig],
) -> tuple[AppConfig, Path]:
    config = fingerprint_config(config_factory, tmp_path / "raw")
    raw_directory = config.ingest.raw_root / config.ingest.dataset_folder
    raw_directory.mkdir(parents=True)
    for index, capture_id in enumerate(config.ingest.partitions.all_capture_ids()):
        (raw_directory / f"{capture_id}.pcap").write_bytes(
            f"synthetic-capture-{index}".encode()
        )
    return config, raw_directory


def test_unchanged_inputs_reproduce_identical_manifest(
    fingerprint_inputs: tuple[AppConfig, Path],
) -> None:
    config, _raw_directory = fingerprint_inputs

    first = build_dataset_manifest(config)
    second = build_dataset_manifest(config)

    assert first == second
    assert first["dataset_version"].startswith("sha256:")


def test_changing_file_bytes_changes_fingerprint(
    fingerprint_inputs: tuple[AppConfig, Path],
) -> None:
    config, raw_directory = fingerprint_inputs
    before = build_dataset_manifest(config)["dataset_version"]
    first_capture = config.ingest.partitions.fit[0]
    (raw_directory / f"{first_capture}.pcap").write_bytes(b"changed bytes")

    after = build_dataset_manifest(config)["dataset_version"]

    assert after != before


def test_changing_category_order_changes_fingerprint(
    fingerprint_inputs: tuple[AppConfig, Path],
) -> None:
    config, _raw_directory = fingerprint_inputs
    changed_data = config.model_dump(mode="json")
    changed_data["ingest"]["categories"] = ["udp", "tcp", "ssdp", "arp"]
    changed = AppConfig.model_validate(changed_data)

    assert (
        build_dataset_manifest(config)["dataset_version"]
        != build_dataset_manifest(changed)["dataset_version"]
    )


def test_changing_window_length_changes_fingerprint(
    fingerprint_inputs: tuple[AppConfig, Path],
) -> None:
    config, _raw_directory = fingerprint_inputs
    changed_data = config.model_dump(mode="json")
    changed_data["ingest"]["window_minutes"] = 5
    changed = AppConfig.model_validate(changed_data)

    assert (
        build_dataset_manifest(config)["dataset_version"]
        != build_dataset_manifest(changed)["dataset_version"]
    )


def test_changing_date_assignment_changes_fingerprint(
    fingerprint_inputs: tuple[AppConfig, Path],
) -> None:
    config, _raw_directory = fingerprint_inputs
    changed_data = deepcopy(config.model_dump(mode="json"))
    moved_capture = changed_data["ingest"]["partitions"]["fit"].pop()
    changed_data["ingest"]["partitions"]["calibration"].insert(0, moved_capture)
    changed = AppConfig.model_validate(changed_data)

    assert (
        build_dataset_manifest(config)["dataset_version"]
        != build_dataset_manifest(changed)["dataset_version"]
    )


def test_missing_source_is_rejected(
    tmp_path: Path,
    config_factory: Callable[..., AppConfig],
) -> None:
    config = fingerprint_config(config_factory, tmp_path / "missing")

    with pytest.raises(DataValidationError, match="dataset source is missing"):
        build_dataset_manifest(config)

import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from lm_idnet.config import AppConfig, load_config

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.json"
SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "normalized_config.json"


@pytest.fixture
def valid_data() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_load_known_valid_configuration() -> None:
    config = load_config(CONFIG_PATH)

    assert config.ingest.categories == ("tcp", "udp", "ssdp", "arp")
    assert config.estimator.categories_k == 4
    assert len(config.ingest.partitions.fit) == 6


def test_normalized_configuration_matches_snapshot() -> None:
    normalized = load_config(CONFIG_PATH).model_dump_json(indent=2)
    expected = SNAPSHOT_PATH.read_text(encoding="utf-8").rstrip()

    assert normalized == expected


def test_missing_required_field_is_rejected(valid_data: dict) -> None:
    del valid_data["seeds"]

    with pytest.raises(ValidationError, match="seeds"):
        AppConfig.model_validate(valid_data)


def test_extra_field_is_rejected(valid_data: dict) -> None:
    valid_data["estimator"]["tolernace"] = 0.1

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AppConfig.model_validate(valid_data)


@pytest.mark.parametrize(
    ("partition", "dates"),
    [
        ("fit", []),
        ("fit", ["camera-2020-10-08", "camera-2020-10-08"]),
        ("calibration", [""]),
    ],
)
def test_invalid_date_lists_are_rejected(
    valid_data: dict, partition: str, dates: list[str]
) -> None:
    valid_data["ingest"]["partitions"][partition] = dates

    with pytest.raises(ValidationError):
        AppConfig.model_validate(valid_data)


def test_overlapping_partitions_are_rejected(valid_data: dict) -> None:
    valid_data["ingest"]["partitions"]["calibration"][0] = valid_data[
        "ingest"
    ]["partitions"]["fit"][0]

    with pytest.raises(ValidationError, match="appears in both"):
        AppConfig.model_validate(valid_data)


def test_out_of_order_capture_ids_are_rejected(valid_data: dict) -> None:
    valid_data["ingest"]["partitions"]["fit"][0:2] = reversed(
        valid_data["ingest"]["partitions"]["fit"][0:2]
    )

    with pytest.raises(ValidationError, match="chronological order"):
        AppConfig.model_validate(valid_data)


def test_partition_boundaries_must_move_forward(valid_data: dict) -> None:
    valid_data["ingest"]["partitions"]["calibration"] = [
        "camera-2020-01-01"
    ]

    with pytest.raises(ValidationError, match="fit must end before calibration"):
        AppConfig.model_validate(valid_data)


def test_duplicate_normalized_categories_are_rejected(valid_data: dict) -> None:
    valid_data["ingest"]["categories"] = ["TCP", " tcp ", "UDP", "ARP"]

    with pytest.raises(ValidationError, match="unique after normalization"):
        AppConfig.model_validate(valid_data)


@pytest.mark.parametrize("window_minutes", [0, -10])
def test_non_positive_window_is_rejected(
    valid_data: dict, window_minutes: int
) -> None:
    valid_data["ingest"]["window_minutes"] = window_minutes

    with pytest.raises(ValidationError, match="must be positive"):
        AppConfig.model_validate(valid_data)


@pytest.mark.parametrize("window_minutes", [10, 7])
def test_positive_window_length_is_accepted(
    valid_data: dict,
    window_minutes: int,
) -> None:
    valid_data["ingest"]["window_minutes"] = window_minutes

    config = AppConfig.model_validate(valid_data)

    assert config.ingest.window_minutes == window_minutes


@pytest.mark.parametrize("window_minutes", [True, 2.5, "10"])
def test_non_integer_window_length_is_rejected(
    valid_data: dict,
    window_minutes: object,
) -> None:
    valid_data["ingest"]["window_minutes"] = window_minutes

    with pytest.raises(ValidationError, match="must be an integer"):
        AppConfig.model_validate(valid_data)


@pytest.mark.parametrize("quantile", [-0.1, 0, 1, 1.1])
def test_invalid_quantile_is_rejected(valid_data: dict, quantile: float) -> None:
    valid_data["calibration"]["quantile"] = quantile

    with pytest.raises(ValidationError):
        AppConfig.model_validate(valid_data)


def test_category_count_mismatch_is_rejected(valid_data: dict) -> None:
    changed = deepcopy(valid_data)
    changed["estimator"]["categories_k"] = 3

    with pytest.raises(ValidationError, match="must equal"):
        AppConfig.model_validate(changed)

import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from lm_idnet.config import AppConfig, load_config

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "d_link_day_cam5.json"
SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "normalized_config.json"


@pytest.fixture
def valid_data() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_load_known_valid_configuration() -> None:
    config = load_config(CONFIG_PATH)

    assert config.ingest.device_id == "D-LinkDayCam5"
    assert config.ingest.device_mac == "b0:c5:54:42:8f:88"
    assert config.ingest.categories == ("tcp", "udp", "ssdp", "arp")
    assert config.estimator.categories_k == 4
    assert config.estimator.initial_alpha_concentration == 10.0
    assert config.estimator.log_likelihood_backend == "lm"
    assert config.calibration.score_type == "raw"
    assert config.adaptation.policy == "conservative"
    assert len(config.ingest.partitions.fit) == 6


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("device_id", " "),
        ("device_mac", "not-a-mac"),
    ],
)
def test_invalid_device_identity_is_rejected(
    valid_data: dict,
    field: str,
    value: object,
) -> None:
    valid_data["ingest"][field] = value

    with pytest.raises(ValidationError):
        AppConfig.model_validate(valid_data)


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


def test_calibration_score_type_defaults_to_raw_and_rejects_unknown(
    valid_data: dict,
) -> None:
    del valid_data["calibration"]["score_type"]
    assert AppConfig.model_validate(valid_data).calibration.score_type == "raw"

    valid_data["calibration"]["score_type"] = "other"
    with pytest.raises(ValidationError, match="score_type"):
        AppConfig.model_validate(valid_data)


def test_adaptation_mode_and_buffer_are_validated(valid_data: dict) -> None:
    valid_data["adaptation"]["mode"] = "other"
    with pytest.raises(ValidationError, match="adaptation.mode"):
        AppConfig.model_validate(valid_data)

    valid_data["adaptation"].update(
        {"mode": "adaptive_threshold", "buffer_size": 29}
    )
    with pytest.raises(ValidationError, match="minimum_samples"):
        AppConfig.model_validate(valid_data)

    valid_data["adaptation"].update(
        {
            "mode": "periodic_refit",
            "buffer_size": 30,
            "minimum_refit_windows": 31,
        }
    )
    with pytest.raises(ValidationError, match="minimum_refit_windows"):
        AppConfig.model_validate(valid_data)


def test_adaptation_policy_is_explicit_and_validated(valid_data: dict) -> None:
    valid_data["adaptation"]["policy"] = "legacy"
    assert AppConfig.model_validate(valid_data).adaptation.policy == "legacy"
    valid_data["adaptation"]["policy"] = "other"
    with pytest.raises(ValidationError, match="adaptation.policy"):
        AppConfig.model_validate(valid_data)


@pytest.mark.parametrize("device", ["unsw_chromecast", "unsw_samsung_camera"])
@pytest.mark.parametrize("variant", ["adaptive_threshold", "periodic_refit", "periodic_refit_288_240"])
def test_legacy_experiment_configs_keep_distinct_outputs(device, variant):
    path = ROOT / "configs" / "experiments" / f"{device}_{variant}.json"
    conservative = load_config(path)
    legacy = load_config(path.parent / "legacy" / path.name)

    assert legacy.adaptation.policy == "legacy"
    for field in ("ingest", "seeds", "estimator", "calibration", "forecast"):
        assert getattr(legacy, field) == getattr(conservative, field)
    for field in ("mode", "buffer_size", "update_every", "minimum_refit_windows", "safe_margin"):
        assert getattr(legacy.adaptation, field) == getattr(conservative.adaptation, field)
    assert legacy.outputs.model_path == conservative.outputs.model_path
    assert legacy.outputs.threshold_path == conservative.outputs.threshold_path
    suffix = Path(legacy.ingest.dataset_folder) / variant
    assert legacy.outputs.events_path == Path("artifacts/experiments/legacy") / suffix / "events.json"
    assert legacy.outputs.reports_dir == Path("reports/experiments/legacy") / suffix


def test_category_count_mismatch_is_rejected(valid_data: dict) -> None:
    changed = deepcopy(valid_data)
    changed["estimator"]["categories_k"] = 3

    with pytest.raises(ValidationError, match="must equal"):
        AppConfig.model_validate(changed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("volume_quantile", 0), ("volume_quantile", 1.01),
        ("max_concentration_ratio", 0.99),
        ("max_category_shift", -0.01), ("max_category_shift", 1.01),
        ("max_threshold_shift_iqr", -0.01),
    ],
)
def test_invalid_adaptation_limits_are_rejected(valid_data, field, value):
    valid_data["adaptation"][field] = value
    with pytest.raises(ValidationError, match=field):
        AppConfig.model_validate(valid_data)


@pytest.mark.parametrize(
    "field",
    ["safe_margin", "volume_quantile", "max_concentration_ratio",
     "max_category_shift", "max_threshold_shift_iqr"],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_adaptation_limits_must_be_finite(valid_data, field, value):
    valid_data["adaptation"][field] = value
    with pytest.raises(ValidationError, match=field):
        AppConfig.model_validate(valid_data)


def test_adaptation_limits_can_be_configured(valid_data):
    valid_data["adaptation"].update({
        "volume_quantile": 1.0,
        "max_concentration_ratio": 2.0,
        "max_category_shift": 0.1,
        "max_threshold_shift_iqr": 0.5,
    })
    settings = AppConfig.model_validate(valid_data).adaptation
    assert settings.volume_quantile == 1.0
    assert settings.max_concentration_ratio == 2.0
    assert settings.max_category_shift == 0.1
    assert settings.max_threshold_shift_iqr == 0.5

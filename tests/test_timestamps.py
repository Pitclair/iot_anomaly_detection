from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from lm_idnet.processing.timestamps import (
    CAPTURE_TIME_ZONE_ASSUMPTION,
    normalize_utc_timestamp,
)
from lm_idnet.config import load_config
from lm_idnet.processing.packet_transformer import PacketTransformer

pytestmark = pytest.mark.unit
CATEGORY_ORDER = load_config(
    Path(__file__).resolve().parents[1] / "configs" / "config.json"
).ingest.categories


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Decimal("0"), "1970-01-01T00:00:00Z"),
        (Decimal("-0.000000001"), "1969-12-31T23:59:59.999999999Z"),
        (Decimal("0.123456789"), "1970-01-01T00:00:00.123456789Z"),
        (datetime(2024, 1, 2, 3, 4, 5), "2024-01-02T03:04:05Z"),
        (
            datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            "2024-01-02T03:04:05Z",
        ),
    ],
)
def test_normalizes_epoch_and_datetime_values_to_utc(value, expected):
    actual = normalize_utc_timestamp(value)

    assert actual == pd.Timestamp(expected)
    assert str(actual.tz) == "UTC"


def test_daylight_saving_folds_normalize_to_distinct_utc_instants():
    new_york = ZoneInfo("America/New_York")
    first_occurrence = datetime(2021, 11, 7, 1, 30, tzinfo=new_york, fold=0)
    second_occurrence = datetime(2021, 11, 7, 1, 30, tzinfo=new_york, fold=1)

    assert normalize_utc_timestamp(first_occurrence) == pd.Timestamp(
        "2021-11-07T05:30:00Z"
    )
    assert normalize_utc_timestamp(second_occurrence) == pd.Timestamp(
        "2021-11-07T06:30:00Z"
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_timestamps(value):
    with pytest.raises(ValueError, match="finite"):
        normalize_utc_timestamp(value)


def test_rejects_precision_beyond_nanoseconds():
    with pytest.raises(ValueError, match="nanosecond precision"):
        normalize_utc_timestamp(Decimal("0.0000000001"))


def test_build_time_series_sorts_out_of_order_packets_in_utc():
    records = [
        (Decimal("2.000000001"), "tcp"),
        (Decimal("1.000000001"), "udp"),
    ]

    transformer = PacketTransformer(CATEGORY_ORDER)
    time_series = transformer.build_time_series(records)

    assert time_series.index.tolist() == [
        pd.Timestamp("1970-01-01T00:00:01.000000001Z"),
        pd.Timestamp("1970-01-01T00:00:02.000000001Z"),
    ]
    assert time_series["protocol"].tolist() == ["udp", "tcp"]


def test_capture_timezone_assumption_is_recorded():
    assert "Unix-epoch UTC" in CAPTURE_TIME_ZONE_ASSUMPTION
    assert "naive datetime values are assumed UTC" in CAPTURE_TIME_ZONE_ASSUMPTION

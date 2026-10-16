"""Protect count aggregation and attack timing in the window-duration study."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import runpy

import pytest

from lm_idnet.processing.schemas import ProcessedDataset, WindowRecord

functions = runpy.run_path(str(Path(__file__).parents[1] / "scripts/evaluate_window_sizes.py"))
coarsen = functions["coarsen"]
interval_metrics = functions["interval_metrics"]
calibration_split = functions["calibration_split"]
AttackInterval = functions["load_annotations"].__globals__["AttackInterval"]
pytestmark = pytest.mark.unit


def window(minute, counts=(1, 2, 3, 4), state="observed"):
    start = datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=minute)
    return WindowRecord(start_utc=start, end_utc=start + timedelta(minutes=10),
                        categories=("tcp", "udp", "ssdp", "arp"), counts=counts, state=state)


def test_aggregation_uses_epoch_boundaries_and_conserves_counts():
    rows = tuple(window(minute) for minute in [10, 20, 30, 40])
    result = coarsen(rows, 20)
    assert len(result) == 1
    assert result[0].start_utc.minute == 20 and result[0].end_utc.minute == 40
    assert result[0].counts == (2, 4, 6, 8)
    assert coarsen(rows, 10) == list(rows)


def test_aggregation_preserves_silence_and_missingness_without_bridging_gaps():
    silence = tuple(window(minute, (0, 0, 0, 0), "observed-silent") for minute in [0, 10])
    assert coarsen(silence, 20)[0].state == "observed-silent"
    missing = (window(0), window(10, None, "missing"))
    assert coarsen(missing, 20)[0].counts is None
    assert coarsen((window(0), window(30)), 20) == []


@pytest.mark.parametrize("minutes", [30, 60])
def test_larger_windows_sum_every_constituent_once(minutes):
    rows = tuple(window(minute) for minute in range(0, minutes, 10))
    result = coarsen(rows, minutes)
    assert len(result) == 1
    assert result[0].end_utc - result[0].start_utc == timedelta(minutes=minutes)
    assert result[0].counts == tuple(value * (minutes // 10) for value in (1, 2, 3, 4))


@pytest.mark.parametrize("minimum", [2, 3, 9])
def test_normal_holdout_preserves_chronology_and_sample_guard(config_factory, minimum):
    partitions = {"fit": ["2020-01-01", "2020-01-02"], "calibration": ["2020-01-03"],
                  "development_test": ["2020-01-04"], "final_test": ["2020-01-05"]}
    config = config_factory(ingest={"partitions": partitions}, calibration={"minimum_samples": minimum})
    captures = {cid: ProcessedDataset(
        metadata={"device_id": config.ingest.device_id, "capture_id": cid, "partition": partition,
                  "date": cid, "file_source": cid + ".pcap"},
        windows=tuple(window(minute) for minute in range(0, 120, 10)),
    ) for partition, ids in partitions.items() for cid in ids}
    if minimum == 9:
        with pytest.raises(ValueError, match="retaining fit data"):
            calibration_split(config, captures)
    else:
        adjusted, datasets, record = calibration_split(config, captures)
        assert record["calibration_windows_at_60m"] >= minimum
        assert adjusted.calibration.minimum_samples == minimum
        assert adjusted.ingest.partitions.development_test == config.ingest.partitions.development_test
        assert adjusted.ingest.partitions.final_test == config.ingest.partitions.final_test
        if minimum == 3:
            assert adjusted.ingest.partitions.fit == ("2020-01-01",)
            assert adjusted.ingest.partitions.calibration == ("2020-01-02", "2020-01-03")
            assert datasets["2020-01-02"].metadata.partition == "calibration"
        else:
            assert record["moved_normal_captures"] == []
    assert config.ingest.partitions.fit == ("2020-01-01", "2020-01-02")
    assert captures["2020-01-02"].metadata.partition == "fit"


@pytest.mark.parametrize("minutes", [0, 15, True, 20.5])
def test_invalid_duration_is_rejected(minutes):
    with pytest.raises(ValueError, match="positive multiple"):
        coarsen((window(0),), minutes)


def test_overlapping_or_unaligned_source_rows_are_rejected():
    with pytest.raises(ValueError, match="non-overlapping"):
        coarsen((window(0), window(0)), 20)
    with pytest.raises(ValueError, match="epoch aligned"):
        coarsen((window(1),), 20)
    fractional = window(0).model_copy(update={
        "start_utc": window(0).start_utc + timedelta(microseconds=1),
        "end_utc": window(0).end_utc + timedelta(microseconds=1),
    })
    with pytest.raises(ValueError, match="epoch aligned"):
        coarsen((fractional,), 20)


def test_interval_detection_uses_alert_close_and_excludes_partial_coverage():
    start = window(0).start_utc
    intervals = tuple(AttackInterval(start + timedelta(minutes=a), start + timedelta(minutes=b), (), "test")
                      for a, b in [(2, 4), (10, 12), (19, 22)])
    result = interval_metrics([
        {"start": start, "end": start + timedelta(minutes=10), "prediction": True},
        {"start": start + timedelta(minutes=10), "end": start + timedelta(minutes=20), "prediction": False},
    ], intervals)
    assert result["attack_intervals"] == 2
    assert result["detected_attack_intervals"] == 1
    assert result["interval_recall"] == .5
    assert result["median_delay_seconds"] == 8 * 60
    assert result["alerts_before_attack_end"] == 0
    assert result["partially_observed_intervals_excluded"] == 1

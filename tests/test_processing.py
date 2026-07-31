import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import ARP, Ether
from scapy.utils import wrpcap

from lm_idnet.config import load_config
from lm_idnet.processing.aggregation import aggregate_packet_traces
from lm_idnet.processing.categories import (
    SSDP_IS_EXCLUSIVE_OF_UDP,
    classify_packet,
)
from lm_idnet.processing.packet_transformer import PacketTransformer
from lm_idnet.processing.window_policy import (
    SUPPORTED_WINDOW_MINUTES,
    WINDOW_CLOSED,
    WINDOW_LABEL,
    WINDOW_ORIGIN,
)
from lm_idnet.processing.pcap_processor import PcapProcessor
from lm_idnet.processing.schemas import WindowRecord
from lm_idnet.processing.statistics import Statistics

pytestmark = pytest.mark.unit
CATEGORY_ORDER = load_config(
    Path(__file__).resolve().parents[1] / "configs" / "config.json"
).ingest.categories


def test_aggregate_simple():
    data = {
        "timestamp": [
            "2020-01-01 00:00:00",
            "2020-01-01 00:01:00",
            "2020-01-01 00:09:59",
            "2020-01-01 00:10:00",
        ],
        "protocol": ["tcp", "udp", "tcp", "arp"],
    }
    df = pd.DataFrame(data)
    out = aggregate_packet_traces(
        df,
        categories=CATEGORY_ORDER,
        window_minutes=10,
    )
    # Expect two windows: first contains 3 events (TCP,UDP,TCP), second contains 1 (ARP)
    assert out.shape[0] == 2
    assert tuple(out.columns) == CATEGORY_ORDER
    assert out["tcp"].sum() == 2
    assert out["arp"].sum() == 1


def test_canonical_packet_classification_and_counts(tmp_path):
    packets = [
        Ether() / IP() / TCP(sport=1234, dport=443),
        Ether() / IP() / UDP(sport=1234, dport=53),
        Ether() / IP() / UDP(sport=1900, dport=50000),
        Ether() / ARP(),
        Ether(type=0x88B5) / b"unsupported",
    ]
    expected = ["tcp", "udp", "ssdp", "arp", "unsupported"]

    assert [classify_packet(packet) for packet in packets] == expected
    assert SSDP_IS_EXCLUSIVE_OF_UDP is True

    pcap_path = tmp_path / "categories.pcap"
    wrpcap(str(pcap_path), packets)
    processor = PcapProcessor(categories=list(CATEGORY_ORDER))
    records = list(processor.process_pcap(str(pcap_path)))

    assert [category for _, category in records] == expected
    assert [processor.protocol_count[name] for name in CATEGORY_ORDER] == [1, 1, 1, 1]
    assert processor.packet_count == 4
    assert processor.unsupported_count == 1

    frame = pd.DataFrame(records, columns=["timestamp", "protocol"])
    counts = aggregate_packet_traces(frame, categories=CATEGORY_ORDER)
    assert tuple(counts.columns) == CATEGORY_ORDER
    assert counts.sum().tolist() == [1, 1, 1, 1]


def test_configured_category_order_flows_through_pipeline_and_report(
    tmp_path,
    capsys,
    config_factory,
):
    config = config_factory(
        ingest={"categories": ["arp", "ssdp", "udp", "tcp"]}
    )
    configured_order = config.ingest.categories
    records = [
        ("2020-01-01T00:00:00Z", "tcp"),
        ("2020-01-01T00:00:01Z", "udp"),
        ("2020-01-01T00:00:02Z", "arp"),
    ]

    processor = PcapProcessor(categories=list(configured_order))
    assert tuple(processor.protocol_count) == configured_order

    transformer = PacketTransformer(configured_order, device_id="camera-01")
    windows = transformer.to_windows(transformer.build_time_series(records))
    model_input = transformer.to_numpy_matrix(windows)

    assert windows[0].categories == configured_order
    assert windows[0].counts == (1, 0, 1, 1)
    assert model_input.tolist() == [[1, 0, 1, 1]]

    dataset_path = tmp_path / "capture-001.json"
    dataset_path.write_text(
        json.dumps(
            {
                "metadata": {"date": "capture-001", "file_source": "fixture.pcap"},
                "windows": [window.model_dump(mode="json") for window in windows],
            }
        ),
        encoding="utf-8",
    )
    Statistics(
        json_dir=tmp_path,
        categories=configured_order,
        dates=["capture-001"],
    ).process_all()
    report = capsys.readouterr().out

    report_positions = [report.index(f"| {category}") for category in configured_order]
    assert report_positions == sorted(report_positions)


def test_packet_transformer_builds_windows_and_matrix():
    transformer = PacketTransformer(CATEGORY_ORDER, device_id="camera-01", window_minutes=10)
    records = [
        ("2020-01-01T00:00:00Z", "tcp"),
        ("2020-01-01T00:20:00Z", "arp"),
    ]

    time_series = transformer.build_time_series(records)
    windows = transformer.to_windows(time_series)
    matrix = transformer.to_numpy_matrix(windows)

    assert windows == [
        WindowRecord(
            device_id="camera-01",
            start_utc="2020-01-01T00:00:00Z",
            end_utc="2020-01-01T00:10:00Z",
            categories=CATEGORY_ORDER,
            counts=(1, 0, 0, 0),
            total_count=1,
            state="observed",
        ),
        WindowRecord(
            device_id="camera-01",
            start_utc="2020-01-01T00:10:00Z",
            end_utc="2020-01-01T00:20:00Z",
            categories=CATEGORY_ORDER,
            counts=(0, 0, 0, 0),
            total_count=0,
            state="observed-silent",
        ),
        WindowRecord(
            device_id="camera-01",
            start_utc="2020-01-01T00:20:00Z",
            end_utc="2020-01-01T00:30:00Z",
            categories=CATEGORY_ORDER,
            counts=(0, 0, 0, 1),
            total_count=1,
            state="observed",
        ),
    ]
    assert matrix.tolist() == [
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
    ]


def test_packet_transformer_rejects_invalid_window_size():
    with pytest.raises(ValueError, match="must be one of"):
        PacketTransformer(CATEGORY_ORDER, device_id="camera-01", window_minutes=0)


def test_declared_capture_discontinuity_is_missing_and_excluded_from_matrix():
    transformer = PacketTransformer(CATEGORY_ORDER, "camera-01", window_minutes=10)
    records = [
        ("2020-01-01T00:00:00Z", "tcp"),
        ("2020-01-01T00:20:00Z", "arp"),
    ]

    windows = transformer.to_windows(
        transformer.build_time_series(records),
        capture_discontinuities=[
            ("2020-01-01T00:10:00Z", "2020-01-01T00:20:00Z")
        ],
    )

    assert windows[1].state == "missing"
    assert windows[1].counts is None
    assert windows[1].total_count is None
    assert transformer.to_numpy_matrix(windows).tolist() == [
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ]


def test_missing_window_rejects_zero_counts_disguised_as_silence():
    with pytest.raises(ValidationError, match="must not contain observed counts"):
        WindowRecord(
            device_id="camera-01",
            start_utc="2020-01-01T00:10:00Z",
            end_utc="2020-01-01T00:20:00Z",
            categories=CATEGORY_ORDER,
            counts=(0, 0, 0, 0),
            total_count=0,
            state="missing",
        )


def test_capture_discontinuity_cannot_hide_observed_packets():
    transformer = PacketTransformer(CATEGORY_ORDER, "camera-01", window_minutes=10)
    time_series = transformer.build_time_series(
        [("2020-01-01T00:00:00Z", "tcp")]
    )

    with pytest.raises(ValueError, match="overlaps.*containing packets"):
        transformer.to_windows(
            time_series,
            capture_discontinuities=[
                ("2020-01-01T00:00:00Z", "2020-01-01T00:10:00Z")
            ],
        )


@pytest.mark.parametrize("window_minutes", SUPPORTED_WINDOW_MINUTES)
def test_half_open_window_assignment_at_boundaries(window_minutes):
    transformer = PacketTransformer(
        CATEGORY_ORDER,
        device_id="camera-01",
        window_minutes=window_minutes,
    )
    duration = pd.Timedelta(minutes=window_minutes)
    boundary = pd.Timestamp("1970-01-01T00:00:00Z") + (2 * duration)
    records = [
        (boundary - pd.Timedelta(nanoseconds=1), "tcp"),
        (boundary, "udp"),
        (boundary + pd.Timedelta(nanoseconds=1), "arp"),
    ]

    windows = transformer.to_windows(transformer.build_time_series(records))

    assert windows == [
        WindowRecord(
            device_id="camera-01",
            start_utc=boundary - duration,
            end_utc=boundary,
            categories=CATEGORY_ORDER,
            counts=(1, 0, 0, 0),
            total_count=1,
            state="observed",
        ),
        WindowRecord(
            device_id="camera-01",
            start_utc=boundary,
            end_utc=boundary + duration,
            categories=CATEGORY_ORDER,
            counts=(0, 1, 0, 1),
            total_count=2,
            state="observed",
        ),
    ]
    assert sum(window.total_count for window in windows) == 3


def test_window_policy_is_explicit_and_epoch_aligned():
    assert WINDOW_ORIGIN == "epoch"
    assert WINDOW_CLOSED == "left"
    assert WINDOW_LABEL == "left"


def test_window_record_round_trip_preserves_timestamps_and_category_order():
    original = WindowRecord(
        device_id="camera-01",
        start_utc="2020-01-01T00:00:00Z",
        end_utc="2020-01-01T00:10:00Z",
        categories=CATEGORY_ORDER,
        counts=(2, 1, 0, 1),
        total_count=4,
        state="observed",
        metadata={"capture_id": "capture-001"},
    )

    restored = WindowRecord.model_validate_json(original.model_dump_json())

    assert restored == original
    assert restored.categories == CATEGORY_ORDER
    assert restored.start_utc.isoformat() == "2020-01-01T00:00:00+00:00"
    assert restored.end_utc.isoformat() == "2020-01-01T00:10:00+00:00"
    assert restored.total_count == sum(restored.counts)


def test_window_record_rejects_incorrect_total_count():
    with pytest.raises(ValidationError, match="sum of counts"):
        WindowRecord(
            device_id="camera-01",
            start_utc="2020-01-01T00:00:00Z",
            end_utc="2020-01-01T00:10:00Z",
            categories=CATEGORY_ORDER,
            counts=(2, 1, 0, 1),
            total_count=3,
            state="observed",
        )

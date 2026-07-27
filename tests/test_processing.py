import pandas as pd
import pytest
from pydantic import ValidationError
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import ARP, Ether
from scapy.utils import wrpcap

from lm_idnet.processing.aggregation import aggregate_packet_traces
from lm_idnet.processing.categories import (
    CATEGORIES,
    CLASSIFICATION_PRIORITY,
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

pytestmark = pytest.mark.unit


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
    out = aggregate_packet_traces(df, window_minutes=10)
    # Expect two windows: first contains 3 events (TCP,UDP,TCP), second contains 1 (ARP)
    assert out.shape[0] == 2
    assert list(out.columns) == list(CATEGORIES)
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
    assert CATEGORIES == ("tcp", "udp", "ssdp", "arp")
    assert CLASSIFICATION_PRIORITY == (
        "arp", "tcp", "ssdp", "udp", "unsupported"
    )
    assert SSDP_IS_EXCLUSIVE_OF_UDP is True

    pcap_path = tmp_path / "categories.pcap"
    wrpcap(str(pcap_path), packets)
    processor = PcapProcessor(categories=list(CATEGORIES))
    records = list(processor.process_pcap(str(pcap_path)))

    assert [category for _, category in records] == expected
    assert [processor.protocol_count[name] for name in CATEGORIES] == [1, 1, 1, 1]
    assert processor.packet_count == 4
    assert processor.unsupported_count == 1

    frame = pd.DataFrame(records, columns=["timestamp", "protocol"])
    counts = aggregate_packet_traces(frame)
    assert list(counts.columns) == list(CATEGORIES)
    assert counts.sum().tolist() == [1, 1, 1, 1]


def test_processor_rejects_noncanonical_category_order():
    with pytest.raises(ValueError, match="canonical order"):
        PcapProcessor(categories=["udp", "tcp", "ssdp", "arp"])


def test_packet_transformer_builds_windows_and_matrix():
    transformer = PacketTransformer(CATEGORIES, device_id="camera-01", window_minutes=10)
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
            categories=CATEGORIES,
            counts=(1, 0, 0, 0),
            total_count=1,
            state="observed",
        ),
        WindowRecord(
            device_id="camera-01",
            start_utc="2020-01-01T00:10:00Z",
            end_utc="2020-01-01T00:20:00Z",
            categories=CATEGORIES,
            counts=(0, 0, 0, 0),
            total_count=0,
            state="observed-silent",
        ),
        WindowRecord(
            device_id="camera-01",
            start_utc="2020-01-01T00:20:00Z",
            end_utc="2020-01-01T00:30:00Z",
            categories=CATEGORIES,
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
        PacketTransformer(CATEGORIES, device_id="camera-01", window_minutes=0)


@pytest.mark.parametrize("window_minutes", SUPPORTED_WINDOW_MINUTES)
def test_half_open_window_assignment_at_boundaries(window_minutes):
    transformer = PacketTransformer(
        CATEGORIES,
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
            categories=CATEGORIES,
            counts=(1, 0, 0, 0),
            total_count=1,
            state="observed",
        ),
        WindowRecord(
            device_id="camera-01",
            start_utc=boundary,
            end_utc=boundary + duration,
            categories=CATEGORIES,
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
        categories=CATEGORIES,
        counts=(2, 1, 0, 1),
        total_count=4,
        state="observed",
        metadata={"capture_id": "capture-001"},
    )

    restored = WindowRecord.model_validate_json(original.model_dump_json())

    assert restored == original
    assert restored.categories == CATEGORIES
    assert restored.start_utc.isoformat() == "2020-01-01T00:00:00+00:00"
    assert restored.end_utc.isoformat() == "2020-01-01T00:10:00+00:00"
    assert restored.total_count == sum(restored.counts)


def test_window_record_rejects_incorrect_total_count():
    with pytest.raises(ValidationError, match="sum of counts"):
        WindowRecord(
            device_id="camera-01",
            start_utc="2020-01-01T00:00:00Z",
            end_utc="2020-01-01T00:10:00Z",
            categories=CATEGORIES,
            counts=(2, 1, 0, 1),
            total_count=3,
            state="observed",
        )

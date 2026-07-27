import pandas as pd
import pytest
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
from lm_idnet.processing.pcap_processor import PcapProcessor
from lm_idnet.processing.schemas import WindowCount

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
    transformer = PacketTransformer(CATEGORIES, window_minutes=10)
    records = [
        ("2020-01-01T00:00:00Z", "tcp"),
        ("2020-01-01T00:20:00Z", "arp"),
    ]

    time_series = transformer.build_time_series(records)
    windows = transformer.to_windows(time_series)
    matrix = transformer.to_numpy_matrix(windows)

    assert windows == [
        WindowCount(tcp=1, udp=0, ssdp=0, arp=0),
        WindowCount(tcp=0, udp=0, ssdp=0, arp=0),
        WindowCount(tcp=0, udp=0, ssdp=0, arp=1),
    ]
    assert matrix.tolist() == [
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
    ]


def test_packet_transformer_rejects_invalid_window_size():
    with pytest.raises(ValueError, match="greater than zero"):
        PacketTransformer(CATEGORIES, window_minutes=0)

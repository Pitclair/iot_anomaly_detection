import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import ARP, Ether
from scapy.utils import wrpcap

from lm_idnet.config import load_config
from lm_idnet.processing.categories import (
    SSDP_IS_EXCLUSIVE_OF_UDP,
    classify_packet,
)
from lm_idnet.processing.packet_transformer import PacketTransformer
from lm_idnet.processing.window_policy import (
    WINDOW_CLOSED,
    WINDOW_LABEL,
    WINDOW_ORIGIN,
)
from lm_idnet.processing.pcap_processor import PcapProcessor
from lm_idnet.processing.schemas import Metadata, ProcessedDataset, WindowRecord
from lm_idnet.processing.statistics import Statistics
from lm_idnet.processing.storage import (
    load_processed_dataset,
    save_processed_dataset,
)

pytestmark = pytest.mark.unit
CATEGORY_ORDER = load_config(
    Path(__file__).resolve().parents[1] / "configs" / "config.json"
).ingest.categories


def test_packet_transformer_counts_simple_trace():
    records = [
        ("2020-01-01 00:00:00", "tcp"),
        ("2020-01-01 00:01:00", "udp"),
        ("2020-01-01 00:09:59", "tcp"),
        ("2020-01-01 00:10:00", "arp"),
    ]
    transformer = PacketTransformer(CATEGORY_ORDER, device_id="camera-01")

    windows = transformer.to_windows(transformer.build_time_series(records))

    assert [window.counts for window in windows] == [
        (2, 1, 0, 0),
        (0, 0, 0, 1),
    ]


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

    transformer = PacketTransformer(CATEGORY_ORDER, device_id="camera-01")
    windows = transformer.to_windows(transformer.build_time_series(records))
    assert transformer.to_numpy_matrix(windows).sum(axis=0).tolist() == [1, 1, 1, 1]


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
    windows = transformer.to_windows(
        transformer.build_time_series(records),
        capture_id="capture-001",
    )
    model_input = transformer.to_numpy_matrix(windows)

    assert windows[0].categories == configured_order
    assert windows[0].counts == (1, 0, 1, 1)
    assert model_input.tolist() == [[1, 0, 1, 1]]

    dataset_path = tmp_path / "capture-001.json"
    dataset_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "capture_id": "capture-001",
                    "partition": "fit",
                    "date": "capture-001",
                    "file_source": "fixture.pcap",
                },
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
            state="observed",
        ),
        WindowRecord(
            device_id="camera-01",
            start_utc="2020-01-01T00:10:00Z",
            end_utc="2020-01-01T00:20:00Z",
            categories=CATEGORY_ORDER,
            counts=(0, 0, 0, 0),
            state="observed-silent",
        ),
        WindowRecord(
            device_id="camera-01",
            start_utc="2020-01-01T00:20:00Z",
            end_utc="2020-01-01T00:30:00Z",
            categories=CATEGORY_ORDER,
            counts=(0, 0, 0, 1),
            state="observed",
        ),
    ]
    assert matrix.tolist() == [
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
    ]


@pytest.mark.parametrize("window_minutes", [0, -1])
def test_packet_transformer_rejects_non_positive_window_size(window_minutes):
    with pytest.raises(ValueError, match="must be positive"):
        PacketTransformer(
            CATEGORY_ORDER,
            device_id="camera-01",
            window_minutes=window_minutes,
        )


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
    with pytest.raises(ValidationError, match="cannot have counts"):
        WindowRecord(
            device_id="camera-01",
            start_utc="2020-01-01T00:10:00Z",
            end_utc="2020-01-01T00:20:00Z",
            categories=CATEGORY_ORDER,
            counts=(0, 0, 0, 0),
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


@pytest.mark.parametrize("window_minutes", [10, 7])
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
            state="observed",
        ),
        WindowRecord(
            device_id="camera-01",
            start_utc=boundary,
            end_utc=boundary + duration,
            categories=CATEGORY_ORDER,
            counts=(0, 1, 0, 1),
            state="observed",
        ),
    ]
    assert sum(window.total_count for window in windows) == 3


def test_window_policy_is_explicit_and_epoch_aligned():
    assert WINDOW_ORIGIN == "epoch"
    assert WINDOW_CLOSED == "left"
    assert WINDOW_LABEL == "left"


@pytest.mark.parametrize(
    ("state", "counts"),
    [
        ("observed", (2, 1, 0, 1)),
        ("observed-silent", (0, 0, 0, 0)),
        ("missing", None),
    ],
)
def test_window_record_round_trip_preserves_all_states(state, counts):
    original = WindowRecord(
        device_id="camera-01",
        capture_id="capture-001",
        start_utc="2020-01-01T00:00:00Z",
        end_utc="2020-01-01T00:10:00Z",
        categories=CATEGORY_ORDER,
        counts=counts,
        state=state,
    )

    restored = WindowRecord.model_validate_json(original.model_dump_json())

    assert restored == original
    assert restored.categories == CATEGORY_ORDER
    assert restored.start_utc.isoformat() == "2020-01-01T00:00:00+00:00"
    assert restored.end_utc.isoformat() == "2020-01-01T00:10:00+00:00"
    expected_total = None if counts is None else sum(counts)
    assert restored.total_count == expected_total


def test_processed_dataset_save_load_round_trip_preserves_model_inputs(tmp_path):
    windows = tuple(
        WindowRecord(
            device_id="camera-01",
            capture_id="capture-001",
            start_utc=f"2020-01-01T00:{index * 10:02d}:00Z",
            end_utc=f"2020-01-01T00:{(index + 1) * 10:02d}:00Z",
            categories=CATEGORY_ORDER,
            counts=counts,
            state=state,
        )
        for index, (state, counts) in enumerate(
            (
                ("observed", (2, 1, 0, 1)),
                ("observed-silent", (0, 0, 0, 0)),
                ("missing", None),
            )
        )
    )
    original = ProcessedDataset(
        metadata=Metadata(
            capture_id="capture-001",
            partition="calibration",
            date="2020-01-01",
            file_source="capture-001.pcap",
        ),
        windows=windows,
    )
    output_path = tmp_path / "capture-001.json"

    save_processed_dataset(original, output_path)
    restored = load_processed_dataset(output_path)

    saved_document = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(saved_document) == {"metadata", "windows"}
    assert restored == original
    assert [window.counts for window in restored.windows] == [
        (2, 1, 0, 1),
        (0, 0, 0, 0),
        None,
    ]
    assert [window.state for window in restored.windows] == [
        "observed",
        "observed-silent",
        "missing",
    ]
    assert restored.windows[0].categories == CATEGORY_ORDER
    assert restored.windows[0].start_utc.isoformat() == "2020-01-01T00:00:00+00:00"
    assert restored.metadata.capture_id == "capture-001"
    assert restored.metadata.partition == "calibration"


def test_window_record_is_immutable():
    window = WindowRecord(
        device_id="camera-01",
        start_utc="2020-01-01T00:00:00Z",
        end_utc="2020-01-01T00:10:00Z",
        categories=CATEGORY_ORDER,
        counts=(1, 0, 0, 0),
        state="observed",
    )

    with pytest.raises(ValidationError, match="frozen"):
        window.state = "missing"


def test_silent_window_rejects_nonzero_counts():
    with pytest.raises(ValidationError, match="require zero counts"):
        WindowRecord(
            device_id="camera-01",
            start_utc="2020-01-01T00:00:00Z",
            end_utc="2020-01-01T00:10:00Z",
            categories=CATEGORY_ORDER,
            counts=(1, 0, 0, 0),
            state="observed-silent",
        )


def test_observed_window_rejects_zero_counts():
    with pytest.raises(ValidationError, match="require packet counts"):
        WindowRecord(
            device_id="camera-01",
            start_utc="2020-01-01T00:00:00Z",
            end_utc="2020-01-01T00:10:00Z",
            categories=CATEGORY_ORDER,
            counts=(0, 0, 0, 0),
            state="observed",
        )

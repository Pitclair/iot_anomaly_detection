import json
import logging
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import ARP, Ether
from scapy.utils import wrpcap

from lm_idnet.algorithms.dirichlet import DirichletFit
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
    Path(__file__).resolve().parents[1] / "configs" / "d_link_day_cam5.json"
).ingest.categories


def test_packet_transformer_counts_simple_trace():
    records = [
        ("2020-01-01 00:00:00", "tcp"),
        ("2020-01-01 00:01:00", "udp"),
        ("2020-01-01 00:09:59", "tcp"),
        ("2020-01-01 00:10:00", "arp"),
    ]
    transformer = PacketTransformer(CATEGORY_ORDER)

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

    transformer = PacketTransformer(CATEGORY_ORDER)
    windows = transformer.to_windows(transformer.build_time_series(records))
    assert transformer.to_numpy_matrix(windows).sum(axis=0).tolist() == [1, 1, 1, 1]


def test_device_filter_matches_ethernet_and_arp_mac(tmp_path):
    device_mac = "f4:f5:d8:8f:0a:3c"
    packets = [
        Ether(src=device_mac) / IP() / TCP(dport=443),
        Ether(dst=device_mac) / IP() / UDP(dport=53),
        Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(hwdst=device_mac),
        Ether() / IP(dst="192.0.2.1") / UDP(dport=53),
    ]
    pcap_path = tmp_path / "mixed-devices.pcap"
    wrpcap(str(pcap_path), packets)
    processor = PcapProcessor(
        categories=list(CATEGORY_ORDER),
        device_mac=device_mac,
    )

    records = list(processor.process_pcap(pcap_path))

    assert [category for _, category in records] == ["tcp", "udp", "arp"]
    assert processor.filtered_count == 1


def test_first_capture_in_fresh_process_loads_ethernet_layers(tmp_path):
    pcap_path = tmp_path / "first-capture.pcap"
    wrpcap(
        str(pcap_path),
        [
            Ether() / IP() / TCP(sport=1234, dport=443),
            Ether() / IP() / UDP(sport=1234, dport=53),
            Ether() / ARP(),
        ],
    )
    script = """
import json
import sys
from lm_idnet.processing.pcap_processor import PcapProcessor

processor = PcapProcessor(categories=["tcp", "udp", "ssdp", "arp"])
records = list(processor.process_pcap(sys.argv[1]))
print(json.dumps([category for _, category in records]))
"""

    result = subprocess.run(
        [sys.executable, "-c", script, str(pcap_path)],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["tcp", "udp", "arp"]


def test_configured_category_order_flows_through_pipeline_and_report(
    tmp_path,
    config_factory,
    caplog,
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

    transformer = PacketTransformer(configured_order)
    windows = transformer.to_windows(transformer.build_time_series(records))
    model_input = transformer.to_numpy_matrix(windows)

    assert windows[0].categories == configured_order
    assert windows[0].counts == (1, 0, 1, 1)
    assert model_input.tolist() == [[1, 0, 1, 1]]

    dataset_path = tmp_path / "capture-001.json"
    dataset_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "device_id": "camera-01",
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
    estimator = Mock()
    estimator.fit.return_value = DirichletFit(
        initial_alpha=np.ones(4),
        alpha=np.full(4, 2.0),
        concentration=8.0,
        psi=0.125,
        iterations=12,
        converged=True,
        initial_log_likelihood=-20.0,
        final_log_likelihood=-10.0,
    )
    with caplog.at_level(logging.INFO):
        report = Statistics(
            processed_dir=tmp_path,
            categories=configured_order,
            estimator=estimator,
            dates=["capture-001"],
        ).build_report()

    capture_report = report["captures"][0]
    assert capture_report["capture_id"] == "capture-001"
    assert capture_report["partition"] == "fit"
    assert [item["category"] for item in capture_report["categories"]] == list(
        configured_order
    )
    assert [item["mean"] for item in capture_report["categories"]] == [1, 0, 1, 1]
    assert capture_report["dirichlet_fit"] == {
        "psi": 0.125,
        "iterations": 12,
        "converged": True,
    }
    estimator.fit.assert_called_once()
    assert estimator.fit.call_args.args[0].tolist() == [[1, 0, 1, 1]]
    assert "Daily psi fitted for capture-001" in caplog.text


def test_packet_transformer_builds_windows_and_matrix():
    transformer = PacketTransformer(CATEGORY_ORDER, window_minutes=10)
    records = [
        ("2020-01-01T00:00:00Z", "tcp"),
        ("2020-01-01T00:20:00Z", "arp"),
    ]

    time_series = transformer.build_time_series(records)
    windows = transformer.to_windows(time_series)
    matrix = transformer.to_numpy_matrix(windows)

    assert windows == [
        WindowRecord(
            start_utc="2020-01-01T00:00:00Z",
            end_utc="2020-01-01T00:10:00Z",
            categories=CATEGORY_ORDER,
            counts=(1, 0, 0, 0),
            state="observed",
        ),
        WindowRecord(
            start_utc="2020-01-01T00:10:00Z",
            end_utc="2020-01-01T00:20:00Z",
            categories=CATEGORY_ORDER,
            counts=(0, 0, 0, 0),
            state="observed-silent",
        ),
        WindowRecord(
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
            window_minutes=window_minutes,
        )


def test_declared_capture_discontinuity_is_missing_and_excluded_from_matrix():
    transformer = PacketTransformer(CATEGORY_ORDER, window_minutes=10)
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
            start_utc="2020-01-01T00:10:00Z",
            end_utc="2020-01-01T00:20:00Z",
            categories=CATEGORY_ORDER,
            counts=(0, 0, 0, 0),
            state="missing",
        )


def test_capture_discontinuity_cannot_hide_observed_packets():
    transformer = PacketTransformer(CATEGORY_ORDER, window_minutes=10)
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
            start_utc=boundary - duration,
            end_utc=boundary,
            categories=CATEGORY_ORDER,
            counts=(1, 0, 0, 0),
            state="observed",
        ),
        WindowRecord(
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
            device_id="camera-01",
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
    assert saved_document["metadata"]["device_id"] == "camera-01"
    assert all("device_id" not in window for window in saved_document["windows"])
    assert all("capture_id" not in window for window in saved_document["windows"])
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
    assert restored.metadata.device_id == "camera-01"
    assert restored.metadata.partition == "calibration"


def test_window_record_is_immutable():
    window = WindowRecord(
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
            start_utc="2020-01-01T00:00:00Z",
            end_utc="2020-01-01T00:10:00Z",
            categories=CATEGORY_ORDER,
            counts=(1, 0, 0, 0),
            state="observed-silent",
        )


def test_observed_window_rejects_zero_counts():
    with pytest.raises(ValidationError, match="require packet counts"):
        WindowRecord(
            start_utc="2020-01-01T00:00:00Z",
            end_utc="2020-01-01T00:10:00Z",
            categories=CATEGORY_ORDER,
            counts=(0, 0, 0, 0),
            state="observed",
        )

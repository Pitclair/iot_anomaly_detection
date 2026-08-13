from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unsw_iot_attack_pcaps.prepare_labels import (
    AttackInterval,
    _matches_device,
    label_window,
)

pytestmark = pytest.mark.unit


def test_raw_device_filter_matches_ethernet_and_arp_mac() -> None:
    mac = bytes.fromhex("f4f5d88f0a3c")
    ethernet_match = mac + bytes(8)
    arp_match = bytes(12) + bytes.fromhex("0806") + bytes(8) + mac + bytes(14)

    assert _matches_device(ethernet_match, mac)
    assert _matches_device(arp_match, mac)
    assert not _matches_device(bytes(42), mac)


def test_label_window_uses_half_open_overlap_and_unioned_duration() -> None:
    utc = timezone.utc
    start = datetime(2018, 10, 22, 14, 40, tzinfo=utc)
    end = datetime(2018, 10, 22, 14, 50, tzinfo=utc)
    annotations = (
        AttackInterval(
            start=datetime(2018, 10, 22, 14, 49, tzinfo=utc),
            end=datetime(2018, 10, 22, 14, 51, tzinfo=utc),
            affected_features=("Localfeatures", "Arpfeatures"),
            attack_type="ArpSpoof1L2D",
        ),
        AttackInterval(
            start=end,
            end=datetime(2018, 10, 22, 14, 52, tzinfo=utc),
            affected_features=("Tcpfeatures",),
            attack_type="boundary-does-not-overlap",
        ),
    )

    label, overlapping = label_window(start, end, annotations)

    assert len(overlapping) == 1
    assert label == {
        "window_start_utc": "2018-10-22T14:40:00Z",
        "window_end_utc": "2018-10-22T14:50:00Z",
        "is_attack": True,
        "attack_types": ["ArpSpoof1L2D"],
        "affected_features": ["Arpfeatures", "Localfeatures"],
        "attack_overlap_seconds": 60,
    }

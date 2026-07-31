"""Canonical packet-category semantics used by the processing pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from scapy.packet import Packet

SUPPORTED_CATEGORY_NAMES: Final[frozenset[str]] = frozenset(
    {"tcp", "udp", "ssdp", "arp"}
)
"""Fixed paper taxonomy. Configuration owns the order of these names."""

UNSUPPORTED: Final[str] = "unsupported"
SSDP_PORT: Final[int] = 1900
SSDP_IS_EXCLUSIVE_OF_UDP: Final[bool] = True

def classify_packet(packet: "Packet") -> str:
    """Return the one canonical category assigned to ``packet``.

    SSDP is exclusive: a UDP packet whose source or destination port is 1900
    is counted as ``ssdp`` and is not also counted as ``udp``.
    """
    # Keep Scapy out of configuration-only imports; it may inspect host network
    # interfaces while loading on some platforms.
    from scapy.layers.inet import TCP, UDP
    from scapy.layers.l2 import ARP

    if packet.haslayer(ARP):
        return "arp"
    if packet.haslayer(TCP):
        return "tcp"
    if packet.haslayer(UDP):
        udp_layer = packet.getlayer(UDP)
        if udp_layer.sport == SSDP_PORT or udp_layer.dport == SSDP_PORT:
            return "ssdp"
        return "udp"
    return UNSUPPORTED


def validate_categories(categories: Sequence[str]) -> tuple[str, ...]:
    """Normalize configured categories while preserving their declared order."""
    normalized = tuple(str(category).strip().lower() for category in categories)
    if any(not category for category in normalized):
        raise ValueError("categories must not contain empty names")
    if len(normalized) != len(set(normalized)):
        raise ValueError("categories must be unique after normalization")

    configured_names = set(normalized)
    if configured_names != SUPPORTED_CATEGORY_NAMES:
        missing = sorted(SUPPORTED_CATEGORY_NAMES - configured_names)
        unsupported = sorted(configured_names - SUPPORTED_CATEGORY_NAMES)
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unsupported:
            details.append(f"unsupported: {', '.join(unsupported)}")
        raise ValueError(
            "categories must contain the fixed paper taxonomy ("
            + "; ".join(details)
            + ")"
        )
    return normalized

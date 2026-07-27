"""Canonical packet-category semantics used by the processing pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from scapy.packet import Packet

CATEGORIES: Final[tuple[str, ...]] = ("tcp", "udp", "ssdp", "arp")
"""Feature-column order. This order is part of the dataset contract."""

UNSUPPORTED: Final[str] = "unsupported"
SSDP_PORT: Final[int] = 1900
SSDP_IS_EXCLUSIVE_OF_UDP: Final[bool] = True

# The order is explicit because packets may contain more than one protocol layer.
CLASSIFICATION_PRIORITY: Final[tuple[str, ...]] = (
    "arp",
    "tcp",
    "ssdp",
    "udp",
    UNSUPPORTED,
)


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


def validate_category_order(categories: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Normalize and validate categories against the fixed feature contract."""
    normalized = tuple(str(category).strip().lower() for category in categories)
    if normalized != CATEGORIES:
        raise ValueError(
            f"categories must have the canonical order {CATEGORIES}; got {normalized}"
        )
    return normalized

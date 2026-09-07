"""Stream and classify packet records from PCAP files."""

import logging
from decimal import Decimal
from pathlib import Path
from typing import Iterator

import pandas as pd
# Register network-layer decoders before reconstructing raw Ethernet frames.
from scapy.layers import inet as _inet_layers  # noqa: F401
from scapy.layers.l2 import Ether
from scapy.utils import RawPcapReader

from .categories import (
    UNSUPPORTED,
    classify_packet,
    validate_categories,
)
from .timestamps import normalize_utc_timestamp

logger = logging.getLogger(__name__)
ETHERNET_LINKTYPE = 1
ARP_ETHERTYPE = 0x0806
VLAN_ETHERTYPES = {0x8100, 0x88A8, 0x9100}


class PcapProcessor:
    def __init__(
        self,
        categories: list[str],
        device_mac: str | None = None,
    ):
        """Initialize the processor with protocol categories."""
        self.categories = validate_categories(categories)
        self.device_mac = device_mac
        self.device_mac_bytes = (
            bytes.fromhex(device_mac.replace(":", "")) if device_mac else None
        )
        self.packet_count = 0
        self.filtered_count = 0
        self.unsupported_count = 0
        self.protocol_count = {category: 0 for category in self.categories}

    def _matches_device(self, packet: bytes) -> bool:
        if self.device_mac_bytes is None:
            return True
        if len(packet) < 14:
            return False
        if self.device_mac_bytes in (packet[:6], packet[6:12]):
            return True

        ethertype = int.from_bytes(packet[12:14])
        payload_offset = 14
        while ethertype in VLAN_ETHERTYPES and len(packet) >= payload_offset + 4:
            ethertype = int.from_bytes(packet[payload_offset + 2 : payload_offset + 4])
            payload_offset += 4

        if ethertype != ARP_ETHERTYPE or len(packet) < payload_offset + 8:
            return False
        hardware_length = packet[payload_offset + 4]
        protocol_length = packet[payload_offset + 5]
        sender_offset = payload_offset + 8
        target_offset = sender_offset + hardware_length + protocol_length
        return hardware_length == 6 and len(packet) >= target_offset + 6 and (
            packet[sender_offset : sender_offset + 6] == self.device_mac_bytes
            or packet[target_offset : target_offset + 6] == self.device_mac_bytes
        )

    @staticmethod
    def _timestamp(metadata: object, reader: object) -> Decimal:
        if hasattr(metadata, "tshigh"):
            ticks = (metadata.tshigh << 32) + metadata.tslow
            return Decimal(ticks) / Decimal(metadata.tsresol)
        resolution = 1_000_000_000 if reader.nano else 1_000_000
        return Decimal(metadata.sec) + Decimal(metadata.usec) / resolution

    def process_pcap(self, pcap_path: str | Path) -> Iterator[tuple[pd.Timestamp, str]]:
        """Yield normalized UTC timestamps and categories from a PCAP file."""
        capture_path = Path(pcap_path)
        logger.info("Starting to process PCAP file: %s", capture_path)

        if not capture_path.is_file():
            raise FileNotFoundError(capture_path)

        with RawPcapReader(str(capture_path)) as reader:
            for packet_number, (raw_packet, metadata) in enumerate(reader, start=1):
                linktype = getattr(metadata, "linktype", None)
                if linktype is None:
                    linktype = reader.linktype
                if linktype != ETHERNET_LINKTYPE:
                    raise ValueError(f"capture is not Ethernet: {capture_path}")
                if not self._matches_device(raw_packet):
                    self.filtered_count += 1
                    continue
                try:
                    timestamp = normalize_utc_timestamp(
                        self._timestamp(metadata, reader)
                    )
                except ValueError as error:
                    raise ValueError(
                        f"invalid timestamp in packet {packet_number} of {capture_path}"
                    ) from error

                packet = Ether(raw_packet)
                protocol = classify_packet(packet)
                if protocol == UNSUPPORTED:
                    self.unsupported_count += 1
                else:
                    self.packet_count += 1
                    self.protocol_count[protocol] += 1
                yield timestamp, protocol

        logger.info("Finished processing PCAP file: %s", capture_path)
        logger.info("Total packets processed: %s", self.packet_count)
        logger.info("Packets excluded by device filter: %s", self.filtered_count)
        logger.info("Unsupported packets: %s", self.unsupported_count)
        logger.info("Protocol counts: %s", self.protocol_count)

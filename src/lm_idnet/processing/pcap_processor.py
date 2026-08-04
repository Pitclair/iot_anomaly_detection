"""Stream and classify packet records from PCAP files."""

import logging
from pathlib import Path
from typing import Iterator

import pandas as pd
# Register link-layer decoders before PcapReader inspects the capture header.
# Otherwise, the first capture opened in a fresh process can become Raw packets.
from scapy.layers import inet as _inet_layers  # noqa: F401
from scapy.layers import l2 as _l2_layers  # noqa: F401
from scapy.utils import PcapReader

from .categories import (
    UNSUPPORTED,
    classify_packet,
    validate_categories,
)
from .timestamps import normalize_utc_timestamp

logger = logging.getLogger(__name__)


class PcapProcessor:
    def __init__(self, categories: list[str]):
        """Initialize the processor with protocol categories."""
        self.categories = validate_categories(categories)
        self.packet_count = 0
        self.unsupported_count = 0
        self.protocol_count = {category: 0 for category in self.categories}

    def process_pcap(self, pcap_path: str | Path) -> Iterator[tuple[pd.Timestamp, str]]:
        """Yield normalized UTC timestamps and categories from a PCAP file."""
        capture_path = Path(pcap_path)
        logger.info("Starting to process PCAP file: %s", capture_path)

        if not capture_path.is_file():
            raise FileNotFoundError(capture_path)

        with PcapReader(str(capture_path)) as reader:
            for packet_number, packet in enumerate(reader, start=1):
                try:
                    timestamp = normalize_utc_timestamp(packet.time)
                except ValueError as error:
                    raise ValueError(
                        f"invalid timestamp in packet {packet_number} of {capture_path}"
                    ) from error

                protocol = classify_packet(packet)
                if protocol == UNSUPPORTED:
                    self.unsupported_count += 1
                else:
                    self.packet_count += 1
                    self.protocol_count[protocol] += 1
                yield timestamp, protocol

        logger.info("Finished processing PCAP file: %s", capture_path)
        logger.info("Total packets processed: %s", self.packet_count)
        logger.info("Unsupported packets: %s", self.unsupported_count)
        logger.info("Protocol counts: %s", self.protocol_count)

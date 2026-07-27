"""Stream packet records with :class:`scapy.utils.PcapReader`."""
import logging
import os
from typing import Iterator, Tuple

import pandas as pd
from scapy.utils import PcapReader

from .categories import (
    CATEGORIES,
    UNSUPPORTED,
    classify_packet,
    validate_category_order,
)
from .timestamps import normalize_utc_timestamp

# Configure logging for the module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PcapProcessor:
    def __init__(self, categories: list[str]):
        """Initialize the processor with protocol categories."""
        self.categories = validate_category_order(categories)
        self.packet_count = 0
        self.unsupported_count = 0
        self.protocol_count = {category: 0 for category in CATEGORIES}

    def process_pcap(self, pcap_path: str) -> Iterator[Tuple[pd.Timestamp, str]]:
        """Yield (timestamp, protocol) for each packet in a pcap file."""
        logger.info(f"Starting to process PCAP file: {pcap_path}")

        if not os.path.exists(pcap_path):
            logger.error(f"File not found: {pcap_path}")
            raise FileNotFoundError(pcap_path)

        with PcapReader(pcap_path) as reader:
            for packet_number, pkt in enumerate(reader, start=1):
                try:
                    timestamp = normalize_utc_timestamp(pkt.time)
                except ValueError as error:
                    raise ValueError(
                        f"invalid timestamp in packet {packet_number} of {pcap_path}"
                    ) from error

                proto = classify_packet(pkt)
                if proto == UNSUPPORTED:
                    self.unsupported_count += 1
                else:
                    self.packet_count += 1
                    self.protocol_count[proto] += 1
                yield timestamp, proto

        logger.info(f"Finished processing PCAP file: {pcap_path}")
        logger.info(f"Total packets processed: {self.packet_count}")
        logger.info(f"Unsupported packets: {self.unsupported_count}")
        logger.info(f"Protocol counts: {self.protocol_count}")

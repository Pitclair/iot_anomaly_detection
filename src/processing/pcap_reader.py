"""
PCAP streaming reader using scapy.utils.PcapReader to yield packet records.
"""
from typing import Iterator, Tuple
import os
from scapy.utils import PcapReader
from scapy.layers.inet import TCP, UDP
from scapy.layers.l2 import ARP
import logging

# Configure logging for the module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#TODO categories should come from config file, but hardcoding for now

class PcapProcessor:
    def __init__(self, categories: list[str]):
        """Initialize the processor with protocol categories."""
        self.categories = categories
        self.packet_count = 0
        self.protocol_count = {category: 0 for category in categories}

    def process_pcap(self, pcap_path: str) -> Iterator[Tuple[float, str]]:
        """Yield (timestamp, protocol) for each packet in a pcap file."""
        logger.info(f"Starting to process PCAP file: {pcap_path}")

        if not os.path.exists(pcap_path):
            logger.error(f"File not found: {pcap_path}")
            raise FileNotFoundError(pcap_path)

        with PcapReader(pcap_path) as reader:
            for pkt in reader:
                try:
                    ts = float(pkt.time)
                except Exception:
                    continue

                proto = None
                if 'ARP' in self.categories and ARP in pkt:
                    proto = 'ARP'
                elif 'TCP' in self.categories and TCP in pkt:
                    proto = 'TCP'
                elif 'UDP' in self.categories and UDP in pkt:
                    udp = pkt[UDP]
                    sport = getattr(udp, 'sport', None)
                    dport = getattr(udp, 'dport', None)
                    if 'SSDP' in self.categories and (sport == 1900 or dport == 1900):
                        proto = 'SSDP'
                    elif 'UDP' in self.categories:
                        proto = 'UDP'

                if proto:
                    self.packet_count += 1
                    self.protocol_count[proto] += 1
                    yield ts, proto

        logger.info(f"Finished processing PCAP file: {pcap_path}")
        logger.info(f"Total packets processed: {self.packet_count}")
        logger.info(f"Protocol counts: {self.protocol_count}")

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

def iter_pcap_packets(pcap_path: str) -> Iterator[Tuple[float, str]]:
    """Yield (timestamp, protocol) for each packet in a pcap file.

    Protocol categories: 'TCP', 'UDP', 'SSDP', 'ARP'
    SSDP is UDP traffic where either sport or dport == 1900.
    """
    logger.info(f"Starting to process PCAP file: {pcap_path}")

    if not os.path.exists(pcap_path):
        logger.error(f"File not found: {pcap_path}")
        raise FileNotFoundError(pcap_path)

    with PcapReader(pcap_path) as reader:
        logger.info(f"Opened PCAP file: {pcap_path}")
        for pkt in reader:
            try:
                ts = float(pkt.time)
                logger.debug(f"Packet timestamp: {ts}")
            except Exception as e:
                logger.warning(f"Skipping malformed packet without timestamp: {e}")
                continue

            proto = None
            if ARP in pkt:
                proto = 'ARP'
                logger.debug("Packet identified as ARP")
            elif TCP in pkt:
                proto = 'TCP'
                logger.debug("Packet identified as TCP")
            elif UDP in pkt:
                udp = pkt[UDP]
                sport = getattr(udp, 'sport', None)
                dport = getattr(udp, 'dport', None)
                if sport == 1900 or dport == 1900:
                    proto = 'SSDP'
                    logger.debug("Packet identified as SSDP")
                else:
                    proto = 'UDP'
                    logger.debug("Packet identified as UDP")
            else:
                logger.debug("Packet ignored (not TCP, UDP, ARP, or SSDP)")
                continue

            logger.info(f"Yielding packet: timestamp={ts}, protocol={proto}")
            yield ts, proto

    logger.info(f"Finished processing PCAP file: {pcap_path}")

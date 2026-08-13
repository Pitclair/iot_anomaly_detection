"""Orchestrate PCAP reading, transformation, and storage."""

import logging
from collections.abc import Mapping
from pathlib import Path

from .packet_transformer import PacketTransformer
from .pcap_processor import PcapProcessor
from .schemas import Metadata, ProcessedDataset
from .storage import save_processed_dataset

logger = logging.getLogger(__name__)


class ProcessingManager:
    def __init__(
        self,
        raw_root: str,
        processed_root: str,
        device_id: str,
        categories: list[str],
        capture_partitions: Mapping[str, str],
        device_mac: str | None = None,
        device_ips: tuple[str, ...] = (),
        window_minutes: int = 10,
    ) -> None:
        self.raw_root = Path(raw_root).resolve(strict=False)
        self.processed_root = Path(processed_root).resolve(strict=False)
        self.device_id = device_id.strip()
        if not self.device_id:
            raise ValueError("device_id must not be empty")
        self.capture_partitions = dict(capture_partitions)
        self.processor = PcapProcessor(
            categories=categories,
            device_mac=device_mac,
            device_ips=device_ips,
        )
        self.transformer = PacketTransformer(
            categories=categories,
            window_minutes=window_minutes,
        )

    def process_file(self, pcap_path: Path, partition: str) -> ProcessedDataset:
        records = self.processor.process_pcap(pcap_path)
        time_series = self.transformer.build_time_series(records)
        windows = self.transformer.to_windows(time_series)
        metadata = Metadata(
            device_id=self.device_id,
            capture_id=pcap_path.stem,
            partition=partition,
            date=pcap_path.stem,
            file_source=str(pcap_path),
        )
        return ProcessedDataset(metadata=metadata, windows=windows)

    def run(self) -> None:
        if not self.raw_root.is_dir():
            raise FileNotFoundError(f"raw data folder not found: {self.raw_root}")
        self.processed_root.mkdir(parents=True, exist_ok=True)

        for capture_id, partition in self.capture_partitions.items():
            pcap_path = self.raw_root / f"{capture_id}.pcap"
            if not pcap_path.is_file():
                logger.warning("PCAP file not found: %s", pcap_path)
                continue

            dataset = self.process_file(pcap_path, partition)
            output_path = self.processed_root / f"{pcap_path.stem}.json"
            save_processed_dataset(dataset, output_path)
            logger.info("Saved processed dataset: %s", output_path)

"""Orchestrate PCAP reading, transformation, and export."""

import json
import logging
from pathlib import Path
from typing import Sequence

from lm_idnet.artifact_schemas import CURRENT_SCHEMA_VERSION
from lm_idnet.artifacts import artifact_checksum

from .packet_transformer import PacketTransformer
from .pcap_processor import PcapProcessor
from .schemas import Metadata, ProcessedDataset

logger = logging.getLogger(__name__)


class ProcessingManager:
    def __init__(
        self,
        raw_root: str,
        processed_root: str,
        device_id: str,
        categories: list[str],
        dates: Sequence[str],
        window_minutes: int = 10,
    ) -> None:
        self.raw_root = Path(raw_root).resolve(strict=False)
        self.processed_root = Path(processed_root).resolve(strict=False)
        self.dates = tuple(dates)
        self.processor = PcapProcessor(categories=categories)
        self.transformer = PacketTransformer(
            categories=categories,
            device_id=device_id,
            window_minutes=window_minutes,
        )

    def process_file(self, pcap_path: Path) -> ProcessedDataset:
        records = self.processor.process_pcap(pcap_path)
        time_series = self.transformer.build_time_series(records)
        windows = self.transformer.to_windows(
            time_series,
            capture_id=pcap_path.stem,
        )
        metadata = Metadata(date=pcap_path.stem, file_source=str(pcap_path))
        return ProcessedDataset(metadata=metadata, windows=windows)

    def export_json(self, dataset: ProcessedDataset, output_path: Path) -> None:
        document = {
            "artifact_type": "processed_dataset",
            "schema_version": CURRENT_SCHEMA_VERSION,
            "metadata": dataset.metadata.model_dump(mode="json"),
            "windows": [
                window.model_dump(mode="json") for window in dataset.windows
            ],
        }
        document["checksum"] = artifact_checksum(document)
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(document, output_file, indent=2)

    def run(self) -> None:
        if not self.raw_root.is_dir():
            raise FileNotFoundError(f"raw data folder not found: {self.raw_root}")
        self.processed_root.mkdir(parents=True, exist_ok=True)

        for capture_id in self.dates:
            pcap_path = self.raw_root / f"{capture_id}.pcap"
            if not pcap_path.is_file():
                logger.warning("PCAP file not found: %s", pcap_path)
                continue

            dataset = self.process_file(pcap_path)
            output_path = self.processed_root / f"{pcap_path.stem}.json"
            self.export_json(dataset, output_path)
            logger.info("Exported JSON: %s", output_path)

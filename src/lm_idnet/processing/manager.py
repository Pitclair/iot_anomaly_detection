"""Orchestrate reading, transforming, validating, and exporting PCAP data."""
from pathlib import Path
from typing import List
import json
from lm_idnet.artifacts import artifact_checksum
from lm_idnet.artifact_schemas import CURRENT_SCHEMA_VERSION
from .pcap_reader import PcapProcessor
from .transformers import build_time_series, to_10min_windows
from .schemas import ProcessedDataset, Metadata
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessingManager:
    def __init__(self, raw_root: str, processed_root: str, categories: list[str], dates: List[str]):
        # Risolvi i percorsi relativi rispetto alla directory di lavoro corrente
        self.raw_root = Path(raw_root).resolve(strict=False)
        self.processed_root = Path(processed_root).resolve(strict=False)
        self.categories = categories
        self.dates = dates
        self.processor = PcapProcessor(categories=self.categories)
        logger.info(f"Initialized ProcessingManager with raw_root={self.raw_root}, processed_root={self.processed_root}, categories={self.categories}, and dates={self.dates}")

    def process_file(self, pcap_path: Path) -> ProcessedDataset:
        records = self.processor.process_pcap(str(pcap_path))
        df = build_time_series(records)
        windows = to_10min_windows(df, categories=self.categories)

        metadata = Metadata(date=pcap_path.stem, file_source=str(pcap_path))
        dataset = ProcessedDataset(metadata=metadata, windows=windows)
        return dataset

    def export_json(self, dataset: ProcessedDataset, out_path: Path):
        out = {
            'artifact_type': 'processed_dataset',
            'schema_version': CURRENT_SCHEMA_VERSION,
            'metadata': dataset.metadata.dict(),
            'windows': [w.dict() for w in dataset.windows]
        }
        out['checksum'] = artifact_checksum(out)
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)

    # def export_csv(self, dataset: ProcessedDataset, out_path: Path):
    #     fields = ['tcp', 'udp', 'ssdp', 'arp']
    #     with out_path.open('w', newline='') as f:
    #         writer = csv.DictWriter(f, fieldnames=fields)
    #         writer.writeheader()
    #         for w in dataset.windows:
    #             writer.writerow(w.dict())

    def run(self):
        if not self.raw_root.exists():
            raise FileNotFoundError(f"Raw data folder not found: {self.raw_root}")
        logger.info(f"Starting processing for folder: {self.raw_root}")

        for date in self.dates:
            pcap = self.raw_root / f"{date}.pcap"
            if not pcap.exists():
                logger.warning(f"PCAP file not found for date: {date}")
                continue

            logger.info(f"Processing PCAP file: {pcap}")
            dataset = self.process_file(pcap)

            out_json = self.processed_root / f"{pcap.stem}.json"
            self.export_json(dataset, out_json)
            logger.info(f"Exported JSON: {out_json}")

        logger.info("Processing complete.")

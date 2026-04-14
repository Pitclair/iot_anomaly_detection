"""
Processing manager to orchestrate reading PCAPs, transforming into windows, validating and exporting.
"""
from pathlib import Path
from typing import List
import json
import csv
from .pcap_reader import iter_pcap_packets
from .transformers import build_time_series, to_10min_windows
from .schemas import ProcessedDataset, Metadata
import logging

# Configure logging for the module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessingManager:
    def __init__(self, raw_root: str, processed_root: str):
        # Risolvi i percorsi relativi rispetto alla directory di lavoro corrente
        self.raw_root = Path(raw_root).resolve(strict=False)
        self.processed_root = Path(processed_root).resolve(strict=False)
        self.processed_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized ProcessingManager with raw_root={self.raw_root} and processed_root={self.processed_root}")

    def process_file(self, pcap_path: Path) -> ProcessedDataset:
        records = iter_pcap_packets(str(pcap_path))
        df = build_time_series(records)
        windows = to_10min_windows(df)

        metadata = Metadata(date=pcap_path.stem, file_source=str(pcap_path))
        dataset = ProcessedDataset(metadata=metadata, windows=windows)
        return dataset

    def export_json(self, dataset: ProcessedDataset, out_path: Path):
        out = {
            'metadata': dataset.metadata.dict(),
            'windows': [w.dict() for w in dataset.windows]
        }
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)

    def export_csv(self, dataset: ProcessedDataset, out_path: Path):
        fields = ['tcp', 'udp', 'ssdp', 'arp']
        with out_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for w in dataset.windows:
                writer.writerow(w.dict())

    def run(self):
        if not self.raw_root.exists():
            raise FileNotFoundError(f"Raw data folder not found: {self.raw_root}")
        logger.info(f"Starting processing for folder: {self.raw_root}")

        for pcap in self.raw_root.iterdir():
            if pcap.suffix.lower() != '.pcap':
                logger.debug(f"Skipping non-PCAP file: {pcap}")
                continue

            logger.info(f"Processing PCAP file: {pcap}")
            dataset = self.process_file(pcap)

            out_json = self.processed_root / f"{pcap.stem}.json"
            out_csv = self.processed_root / f"{pcap.stem}.csv"

            self.export_json(dataset, out_json)
            logger.info(f"Exported JSON: {out_json}")

            self.export_csv(dataset, out_csv)
            logger.info(f"Exported CSV: {out_csv}")

        logger.info("Processing complete.")

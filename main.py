# python
"""
Entry point CLI for Modeling and Forecasting stages.
This script adjusts sys.path so `src/` is importable and exposes a simple CLI.
"""
import argparse
import json
import sys
from pathlib import Path

# ensure src package is importable
ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from processing.aggregation import aggregate_packet_traces
from models.modeling_stage import run_modeling
from models.forecasting_stage import run_forecasting
from processing.manager import ProcessingManager
from processing.statistics import Statistics
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return json.loads(p.read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description='IoT Anomaly Detection Baseline')
    parser.add_argument('--config', '-c', default=str(ROOT / 'configs' / 'config.json'))
    parser.add_argument('--stage', '-s', choices=['model', 'forecast'], required=True,
                        help='Stage to run: "model" or "forecast"')


    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    ingest_cfg = cfg.get('ingest', {})
    categories = ingest_cfg.get('categories')
    dataset_folder = ingest_cfg.get('dataset_folder')
    training_dates = ingest_cfg.get('training_dates')
    testing_dates = ingest_cfg.get('testing_dates')
    raw_path = Path(ingest_cfg.get('raw_root'))  / dataset_folder
    processed_path = Path(ingest_cfg.get('processed_root')) / dataset_folder
    categories_k = cfg.get('categories_k', 4)
    tolerance_delta = cfg.get('tolerance_delta', 1e-9)
    model_out_path = cfg.get('model', {}).get('persist_path', 'data/processed/model_alpha.json')


    if categories is None or dataset_folder is None or training_dates is None or testing_dates is None:
        raise ValueError("categories, dataset_folder, training_dates e testing_dates devono essere specificati nel file di configurazione o tramite CLI.")

    dates = training_dates + testing_dates


    mgr = ProcessingManager(raw_root=str(raw_path), processed_root=str(processed_path), categories=categories, dates=dates)
    logger.info(f"Auto preprocessing {len(dates)} PCAP packets to {processed_path}")
    mgr.run()
    logger.info('Auto preprocessing complete.')

    stats = Statistics(json_dir=processed_path, categories=categories, dates=dates)
    stats.process_all()



    if args.stage == 'model':
        run_modeling(
            data_path=processed_path,
            categories_k=categories_k,
            tolerance_delta=tolerance_delta,
            model_out_path=model_out_path
        )
    elif args.stage == 'forecast':
        run_forecasting(cfg, dataset=processed_root)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

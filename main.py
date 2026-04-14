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


def load_config(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return json.loads(p.read_text())


def main(argv=None):
    # Start preprocessing immediately when the project starts (Phase 1)
    training_dates = [
        'D-LinkDayCam5_88-2020-10-08',
        'D-LinkDayCam5_88-2020-10-09',
        'D-LinkDayCam5_88-2020-10-10',
        'D-LinkDayCam5_88-2020-10-11',
        'D-LinkDayCam5_88-2020-10-12',
        'D-LinkDayCam5_88-2020-10-13',
        'D-LinkDayCam5_88-2020-10-14',
        'D-LinkDayCam5_88-2020-10-15',
        'D-LinkDayCam5_88-2020-10-16',
    ]
    testing_dates = [
        'D-LinkDayCam5_88-2020-10-19',
        'D-LinkDayCam5_88-2020-10-21',
        'D-LinkDayCam5_88-2020-10-22',
    ]
    dates = training_dates + testing_dates

    dataset_folder = 'D-LinkDayCam5'

    parser = argparse.ArgumentParser(description='IoT Anomaly Detection Baseline')
    parser.add_argument('--config', '-c', default=str(ROOT / 'configs' / 'config.json'))
    parser.add_argument('--stage', '-s', choices=['model', 'forecast'], required=True,
                        help='Stage to run: "model" or "forecast"')
    parser.add_argument('--dataset', '-d', default=dataset_folder,)

    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.stage == 'model':
        run_modeling(cfg, dataset=args.dataset)
    elif args.stage == 'forecast':
        run_forecasting(cfg, dataset=args.dataset)
    else:
        parser.print_help()


    raw_root = Path('data') / 'raw' / dataset_folder
    processed_root = Path('data') / 'processed' / dataset_folder
    mgr = ProcessingManager(raw_root=str(raw_root), processed_root=str(processed_root))
    print(f"Auto preprocessing {len(dates)} PCAP stems to {processed_root}")
    mgr.run()
    print('Auto preprocessing complete.')


if __name__ == '__main__':
    main()

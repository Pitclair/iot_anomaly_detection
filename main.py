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
    parser.add_argument('--dataset', '-d', default='camera_5')

    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    if args.stage == 'model':
        run_modeling(cfg, dataset=args.dataset)
    elif args.stage == 'forecast':
        run_forecasting(cfg, dataset=args.dataset)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

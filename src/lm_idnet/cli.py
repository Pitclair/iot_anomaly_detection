"""Command-line interface for LM-IDNet."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from lm_idnet.config import load_config

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without importing optional pipeline dependencies."""
    parser = argparse.ArgumentParser(
        prog="lm-idnet",
        description="IoT Anomaly Detection Baseline",
    )
    parser.add_argument(
        "stage",
        choices=("model", "forecast"),
        help='Stage to run: "model" or "forecast"',
    )
    parser.add_argument(
        "--config",
        "-c",
        default="configs/config.json",
        help="Path to the JSON configuration (default: configs/config.json)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected pipeline stage."""
    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    ingest = config.ingest

    # Keep heavyweight imports out of module import and --help execution.
    from lm_idnet.models.forecasting_stage import run_forecasting
    from lm_idnet.models.modeling_stage import run_modeling
    from lm_idnet.processing.manager import ProcessingManager
    from lm_idnet.processing.statistics import Statistics

    raw_path = ingest.raw_root / ingest.dataset_folder
    processed_path = ingest.processed_root / ingest.dataset_folder
    dates = list(ingest.training_dates + ingest.testing_dates)

    manager = ProcessingManager(
        raw_root=str(raw_path),
        processed_root=str(processed_path),
        categories=list(ingest.categories),
        dates=dates,
    )
    logger.info("Preprocessing %d configured captures to %s", len(dates), processed_path)
    manager.run()
    Statistics(
        json_dir=processed_path,
        categories=list(ingest.categories),
        dates=dates,
    ).process_all()

    if args.stage == "model":
        run_modeling(
            data_path=processed_path,
            categories_k=config.estimator.categories_k,
            tolerance_delta=config.estimator.tolerance_delta,
            model_out_path=str(config.outputs.model_path),
        )
    else:
        run_forecasting(config.model_dump(mode="json"), dataset=str(processed_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line interface for LM-IDNet."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict:
    """Load a JSON configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


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

    ingest = config.get("ingest", {})
    categories = ingest.get("categories")
    dataset_folder = ingest.get("dataset_folder")
    training_dates = ingest.get("training_dates")
    testing_dates = ingest.get("testing_dates")
    if any(
        value is None
        for value in (categories, dataset_folder, training_dates, testing_dates)
    ):
        raise ValueError(
            "categories, dataset_folder, training_dates, and testing_dates "
            "must be specified in the configuration"
        )

    # Keep heavyweight imports out of module import and --help execution.
    from lm_idnet.models.forecasting_stage import run_forecasting
    from lm_idnet.models.modeling_stage import run_modeling
    from lm_idnet.processing.manager import ProcessingManager
    from lm_idnet.processing.statistics import Statistics

    raw_path = Path(ingest["raw_root"]) / dataset_folder
    processed_path = Path(ingest["processed_root"]) / dataset_folder
    dates = training_dates + testing_dates

    manager = ProcessingManager(
        raw_root=str(raw_path),
        processed_root=str(processed_path),
        categories=categories,
        dates=dates,
    )
    logger.info("Preprocessing %d configured captures to %s", len(dates), processed_path)
    manager.run()
    Statistics(
        json_dir=processed_path,
        categories=categories,
        dates=dates,
    ).process_all()

    if args.stage == "model":
        run_modeling(
            data_path=processed_path,
            categories_k=config.get("categories_k", 4),
            tolerance_delta=config.get("tolerance_delta", 1e-9),
            model_out_path=config.get("model", {}).get(
                "persist_path", "data/processed/model_alpha.json"
            ),
        )
    else:
        run_forecasting(config, dataset=str(processed_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

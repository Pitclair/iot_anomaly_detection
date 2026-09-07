"""Stable command-line interface for LM-IDNet."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from lm_idnet.config import AppConfig, load_config
from lm_idnet.exceptions import (
    CommandUnavailableError,
    ConfigurationError,
    IngestionError,
    LMIDNetError,
)
from lm_idnet.partitioning import all_capture_ids, partition_name_for_capture

logger = logging.getLogger(__name__)
DEFAULT_LOG_PATH = Path("logs/lm_idnet.log")
LOG_FORMAT = "%(asctime)s.%(msecs)03dZ %(levelname)s %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


class UtcLogFormatter(logging.Formatter):
    """Format log timestamps as explicit ISO 8601 UTC values."""

    converter = time.gmtime


COMMANDS = (
    "preprocess",
    "diagnose",
    "train",
    "calibrate",
    "score",
    "evaluate",
    "adapt",
    "forecast",
    "benchmark",
)

COMMAND_HELP = {
    "preprocess": "convert configured packet captures into processed windows",
    "diagnose": "report descriptive statistics and daily psi estimates",
    "train": "load fit counts and initialize the normal-traffic model",
    "calibrate": "calibrate an anomaly threshold for a trained model",
    "score": "score processed windows and emit anomaly decisions",
    "evaluate": "evaluate detector outputs using the frozen protocol",
    "adapt": "train and assess a guarded adaptive-model candidate",
    "forecast": "forecast held-out traffic from a verified model",
    "benchmark": "measure configured backend and pipeline performance",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the parser without importing pipeline dependencies."""
    parser = argparse.ArgumentParser(
        prog="lm-idnet",
        description="LM-IDNet IoT anomaly detector",
    )
    parser.add_argument(
        "--error-format",
        choices=("text", "json"),
        default="text",
        help="format for expected runtime errors (default: text)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show informational logs in the terminal",
    )
    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        required=True,
    )
    command_parsers: dict[str, argparse.ArgumentParser] = {}
    for command in COMMANDS:
        command_parser = subparsers.add_parser(
            command,
            help=COMMAND_HELP[command],
            description=COMMAND_HELP[command].capitalize() + ".",
        )
        command_parsers[command] = command_parser
        command_parser.add_argument(
            "--config",
            "-c",
            required=True,
            help="path to a validated JSON configuration",
        )
        command_parser.add_argument(
            "--dry-run",
            action="store_true",
            help="validate configuration and command availability without execution",
        )
    capture_mode = command_parsers["preprocess"].add_mutually_exclusive_group()
    capture_mode.add_argument(
        "--inventory-only",
        action="store_true",
        help="inventory configured captures without preprocessing them",
    )
    command_parsers["preprocess"].add_argument(
        "--inventory-output",
        type=Path,
        help="inventory JSON path (default: <reports_dir>/capture_inventory.json)",
    )
    capture_mode.add_argument(
        "--validate-captures-only",
        action="store_true",
        help="validate configured captures to EOF without preprocessing them",
    )
    command_parsers["preprocess"].add_argument(
        "--validation-output",
        type=Path,
        help="validation JSON path (default: <reports_dir>/capture_validation.json)",
    )
    capture_mode.add_argument(
        "--fingerprint-only",
        action="store_true",
        help="fingerprint source bytes and data-shaping policy without preprocessing",
    )
    command_parsers["preprocess"].add_argument(
        "--fingerprint-output",
        type=Path,
        help="fingerprint JSON path (default: <reports_dir>/dataset_fingerprint.json)",
    )
    capture_mode.add_argument(
        "--validate-timeline-only",
        action="store_true",
        help="validate the canonical processed timeline without preprocessing",
    )
    command_parsers["preprocess"].add_argument(
        "--timeline-output",
        type=Path,
        help="timeline JSON path (default: <reports_dir>/timeline_validation.json)",
    )
    command_parsers["diagnose"].add_argument(
        "--output",
        type=Path,
        help="statistics JSON path (default: <reports_dir>/capture_statistics.json)",
    )
    return parser


def _configure_logging(verbose: bool) -> Path:
    """Write informational logs to disk and optionally echo them to stderr."""
    log_path = Path(os.environ.get("LM_IDNET_LOG_PATH", DEFAULT_LOG_PATH))
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
    except OSError as error:
        raise ConfigurationError(f"cannot open log file: {log_path}") from error

    formatter = UtcLogFormatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    terminal_handler = logging.StreamHandler()
    terminal_handler.setLevel(logging.INFO if verbose else logging.WARNING)
    terminal_handler.setFormatter(formatter)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, terminal_handler],
        force=True,
    )
    return log_path


def _processed_path(config: AppConfig) -> Path:
    return config.ingest.processed_root / config.ingest.dataset_folder


def _preprocess(config: AppConfig) -> None:
    from lm_idnet.processing.processing_manager import ProcessingManager

    ingest = config.ingest
    manager = ProcessingManager(
        raw_root=str(ingest.raw_root / ingest.dataset_folder),
        processed_root=str(_processed_path(config)),
        device_id=ingest.device_id,
        categories=list(ingest.categories),
        capture_partitions={
            capture_id: partition_name_for_capture(config, capture_id)
            for capture_id in all_capture_ids(config)
        },
        device_mac=ingest.device_mac,
        window_minutes=ingest.window_minutes,
    )
    try:
        manager.run()
    except (OSError, ValueError) as error:
        raise IngestionError(f"preprocessing failed: {error}") from error


def _diagnose(config: AppConfig, output_path: Path) -> None:
    from lm_idnet.algorithms.estimator_factory import create_estimator
    from lm_idnet.processing.statistics import Statistics

    ingest = config.ingest
    Statistics(
        processed_dir=_processed_path(config),
        categories=list(ingest.categories),
        estimator=create_estimator(config.estimator, config.precision_digits),
        dates=list(all_capture_ids(config)),
    ).write_report(output_path)


def _forecast(config: AppConfig) -> None:
    from lm_idnet.models.forecasting_stage import run_forecasting

    run_forecasting(config)


def _not_implemented(command: str) -> Callable[[AppConfig], None]:
    def reject(_config: AppConfig) -> None:
        raise CommandUnavailableError(
            f"{command} is registered but its implementation is not available yet"
        )

    return reject


HANDLERS: dict[str, Callable[[AppConfig], None]] = {
    "preprocess": _preprocess,
    "adapt": _not_implemented("adapt"),
    "forecast": _forecast,
}


def run_command(argv: Sequence[str] | None = None) -> None:
    """Validate and dispatch exactly one CLI command."""
    args = build_parser().parse_args(argv)
    log_path = _configure_logging(args.verbose)
    logger.info("Writing application logs to %s", log_path)
    logger.info("Starting command: %s", args.command)
    config = load_config(args.config)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "command": args.command,
                    "config": str(args.config),
                    "status": "ready",
                },
                sort_keys=True,
            )
        )
        return

    if args.command == "preprocess" and args.inventory_only:
        from lm_idnet.processing.inventory import inventory_configured_captures

        output_path = (
            args.inventory_output
            or config.outputs.reports_dir / "capture_inventory.json"
        )
        inventory_configured_captures(config, output_path)
        print(
            json.dumps(
                {
                    "command": "preprocess",
                    "inventory": str(output_path),
                    "status": "completed",
                },
                sort_keys=True,
            )
        )
        return

    if args.command == "preprocess" and args.validate_captures_only:
        from lm_idnet.processing.pcap_validation import validate_configured_captures

        output_path = (
            args.validation_output
            or config.outputs.reports_dir / "capture_validation.json"
        )
        validate_configured_captures(config, output_path)
        print(
            json.dumps(
                {
                    "command": "preprocess",
                    "status": "completed",
                    "validation": str(output_path),
                },
                sort_keys=True,
            )
        )
        return

    if args.command == "preprocess" and args.fingerprint_only:
        from lm_idnet.dataset_fingerprint import fingerprint_configured_dataset

        output_path = (
            args.fingerprint_output
            or config.outputs.reports_dir / "dataset_fingerprint.json"
        )
        manifest = fingerprint_configured_dataset(config, output_path)
        print(
            json.dumps(
                {
                    "command": "preprocess",
                    "dataset_version": manifest["dataset_version"],
                    "fingerprint": str(output_path),
                    "status": "completed",
                },
                sort_keys=True,
            )
        )
        return

    if args.command == "preprocess" and args.validate_timeline_only:
        from lm_idnet.processing.timeline import validate_unique_timeline

        output_path = (
            args.timeline_output
            or config.outputs.reports_dir / "timeline_validation.json"
        )
        validate_unique_timeline(config, output_path)
        print(
            json.dumps(
                {
                    "command": "preprocess",
                    "status": "completed",
                    "timeline_validation": str(output_path),
                },
                sort_keys=True,
            )
        )
        return

    if args.command == "diagnose":
        output_path = (
            args.output or config.outputs.reports_dir / "capture_statistics.json"
        )
        _diagnose(config, output_path)
        print(
            json.dumps(
                {
                    "command": "diagnose",
                    "report": str(output_path),
                    "status": "completed",
                },
                sort_keys=True,
            )
        )
        return

    if args.command == "train":
        from lm_idnet.models.modeling_stage import train_model

        result = train_model(config)
        print(
            json.dumps(
                {
                    "command": "train",
                    "status": "completed",
                    **result,
                },
                sort_keys=True,
            )
        )
        return

    if args.command == "calibrate":
        from lm_idnet.models.calibration_stage import calibrate_threshold

        result = calibrate_threshold(config)
        print(
            json.dumps(
                {
                    "command": "calibrate",
                    "status": "completed",
                    **result,
                },
                sort_keys=True,
            )
        )
        return

    if args.command == "score":
        from lm_idnet.models.scoring_stage import score_windows

        result = score_windows(config)
        print(
            json.dumps(
                {
                    "command": "score",
                    "status": "completed",
                    **result,
                },
                sort_keys=True,
            )
        )
        return

    if args.command == "evaluate":
        from lm_idnet.evaluation.evaluation_stage import evaluate_scores

        result = evaluate_scores(config)
        print(
            json.dumps(
                {
                    "command": "evaluate",
                    "status": "completed",
                    **result,
                },
                sort_keys=True,
            )
        )
        return

    if args.command == "benchmark":
        from lm_idnet.evaluation.benchmark_stage import benchmark_backends

        result = benchmark_backends(config)
        print(
            json.dumps(
                {
                    "command": "benchmark",
                    "status": "completed",
                    **result,
                },
                sort_keys=True,
            )
        )
        return

    HANDLERS[args.command](config)
    print(json.dumps({"command": args.command, "status": "completed"}))


def _json_errors_requested(argv: Sequence[str] | None) -> bool:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        position = arguments.index("--error-format")
    except ValueError:
        return False
    return (
        position + 1 < len(arguments)
        and arguments[position + 1] == "json"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and map expected failures to stable exit codes."""
    try:
        run_command(argv)
    except LMIDNetError as error:
        if _json_errors_requested(argv):
            message = json.dumps(
                {
                    "error": {
                        "code": error.error_code,
                        "message": str(error),
                    }
                },
                sort_keys=True,
            )
        else:
            message = f"ERROR [{error.error_code}]: {error}"
        print(message, file=sys.stderr)
        return error.exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

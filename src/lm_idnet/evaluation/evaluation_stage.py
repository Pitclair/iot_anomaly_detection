"""Compare scored windows with their ground-truth labels."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from lm_idnet.config import AppConfig
from lm_idnet.exceptions import DataValidationError
from lm_idnet.models.schemas import AnomalyEventArtifact
from lm_idnet.partitioning import development_partition_for_evaluation


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DataValidationError(f"cannot read evaluation input: {path}") from error


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("window timestamp must be a string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("window timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def evaluate_scores(config: AppConfig) -> dict[str, object]:
    """Evaluate development-test anomaly decisions and save a JSON report."""
    selection = development_partition_for_evaluation(config)
    raw_events = _read_json(Path(config.outputs.events_path))
    if not isinstance(raw_events, list):
        raise DataValidationError("scoring results must be a JSON array")

    try:
        events = [AnomalyEventArtifact.model_validate(event) for event in raw_events]
    except ValidationError as error:
        raise DataValidationError(f"invalid scoring result: {error}") from error

    predictions: dict[tuple[str, datetime, datetime], bool] = {}
    for event in events:
        if event.capture_id not in selection.capture_ids:
            raise DataValidationError(
                f"score is outside {selection.name}: {event.capture_id}"
            )
        key = (event.capture_id, event.window_start_utc, event.window_end_utc)
        if key in predictions:
            raise DataValidationError(f"duplicate scored window: {event.capture_id}")
        predictions[key] = event.is_anomaly

    labels: dict[tuple[str, datetime, datetime], bool] = {}
    labels_dir = (
        config.ingest.processed_root.parent
        / "labels"
        / config.ingest.dataset_folder
    )
    try:
        for capture_id in selection.capture_ids:
            raw_labels = _read_json(labels_dir / f"{capture_id}.json")
            if not isinstance(raw_labels, dict) or not isinstance(
                raw_labels.get("windows"), list
            ):
                raise ValueError("label file must contain a windows array")
            if raw_labels.get("capture_id") != capture_id:
                raise ValueError("label capture ID does not match its filename")
            for window in raw_labels["windows"]:
                if not isinstance(window, dict) or not isinstance(
                    window.get("is_attack"), bool
                ):
                    raise ValueError("window label must contain boolean is_attack")
                key = (
                    capture_id,
                    _timestamp(window.get("window_start_utc")),
                    _timestamp(window.get("window_end_utc")),
                )
                if key in labels:
                    raise ValueError(f"duplicate labeled window: {capture_id}")
                labels[key] = window["is_attack"]
    except (TypeError, ValueError) as error:
        raise DataValidationError(f"invalid evaluation labels: {error}") from error

    matched = predictions.keys() & labels.keys()
    if not matched:
        raise DataValidationError("no scored windows match the evaluation labels")

    true_positive = sum(predictions[key] and labels[key] for key in matched)
    true_negative = sum(not predictions[key] and not labels[key] for key in matched)
    false_positive = sum(predictions[key] and not labels[key] for key in matched)
    false_negative = sum(not predictions[key] and labels[key] for key in matched)

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    precision = ratio(true_positive, true_positive + false_positive)
    recall = ratio(true_positive, true_positive + false_negative)
    report = {
        "partition": selection.name,
        "capture_ids": list(selection.capture_ids),
        "matched_window_count": len(matched),
        "unmatched_score_count": len(predictions.keys() - labels.keys()),
        "unmatched_label_count": len(labels.keys() - predictions.keys()),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "accuracy": (true_positive + true_negative) / len(matched),
        "precision": precision,
        "recall": recall,
        "f1_score": (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        ),
    }
    output_path = config.outputs.reports_dir / "evaluation_statistics.json"
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise DataValidationError(
            f"cannot save evaluation report: {output_path}"
        ) from error
    return {**report, "report_path": str(output_path)}

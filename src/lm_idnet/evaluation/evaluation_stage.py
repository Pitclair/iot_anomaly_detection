"""Compare scored windows with their ground-truth labels."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from pydantic import ValidationError

from lm_idnet.config import AppConfig
from lm_idnet.exceptions import DataValidationError
from lm_idnet.models.schemas import AnomalyEventArtifact
from lm_idnet.partitioning import development_partition_for_evaluation

WindowKey = tuple[str, datetime, datetime]
CAPTURE_SUMMARY_FIELDS = (
    "precision",
    "recall",
    "attack_episode_recall",
    "false_positive",
    "false_alert_episodes_per_day",
    "median_detection_delay_minutes",
    "ground_truth_attack_episode_count",
)


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


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _episodes(
    keys: set[WindowKey],
    flags: dict[WindowKey, bool],
) -> list[list[WindowKey]]:
    """Group exactly adjacent flagged windows without crossing captures or gaps."""
    episodes: list[list[WindowKey]] = []
    for key in sorted(keys):
        if not flags[key]:
            continue
        if (
            episodes
            and episodes[-1][-1][0] == key[0]
            and episodes[-1][-1][2] == key[1]
        ):
            episodes[-1].append(key)
        else:
            episodes.append([key])
    return episodes


def _statistics(
    keys: set[WindowKey],
    predictions: dict[WindowKey, bool],
    labels: dict[WindowKey, bool],
) -> dict[str, object]:
    true_positive = sum(predictions[key] and labels[key] for key in keys)
    true_negative = sum(not predictions[key] and not labels[key] for key in keys)
    false_positive = sum(predictions[key] and not labels[key] for key in keys)
    false_negative = sum(not predictions[key] and labels[key] for key in keys)
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)

    alert_episodes = _episodes(keys, predictions)
    attack_episodes = _episodes(keys, labels)
    detection_delays = [
        (
            next(key for key in episode if predictions[key])[2] - episode[0][1]
        ).total_seconds()
        / 60
        for episode in attack_episodes
        if any(predictions[key] for key in episode)
    ]
    false_alert_episode_count = sum(
        not any(labels[key] for key in episode) for episode in alert_episodes
    )
    evaluated_duration_days = sum(
        (end - start).total_seconds() for _, start, end in keys
    ) / (24 * 60 * 60)

    return {
        "matched_window_count": len(keys),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "accuracy": _ratio(true_positive + true_negative, len(keys)),
        "precision": precision,
        "recall": recall,
        "f1_score": (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        ),
        "predicted_alert_episode_count": len(alert_episodes),
        "ground_truth_attack_episode_count": len(attack_episodes),
        "detected_attack_episode_count": len(detection_delays),
        "missed_attack_episode_count": len(attack_episodes) - len(detection_delays),
        "false_alert_episode_count": false_alert_episode_count,
        "attack_episode_recall": _ratio(len(detection_delays), len(attack_episodes)),
        "evaluated_duration_days": evaluated_duration_days,
        "false_alert_episodes_per_day": _ratio(
            false_alert_episode_count,
            evaluated_duration_days,
        ),
        "median_detection_delay_minutes": (
            median(detection_delays) if detection_delays else None
        ),
    }


def _capture_summary(
    capture_reports: list[dict[str, object]],
) -> dict[str, dict[str, float | int] | None]:
    summary = {}
    for field in CAPTURE_SUMMARY_FIELDS:
        values = [
            value
            for capture in capture_reports
            if (value := capture[field]) is not None
        ]
        summary[field] = (
            {
                "minimum": min(values),
                "median": median(values),
                "maximum": max(values),
            }
            if values
            else None
        )
    return summary


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

    predictions: dict[WindowKey, bool] = {}
    for event in events:
        if event.capture_id not in selection.capture_ids:
            raise DataValidationError(
                f"score is outside {selection.name}: {event.capture_id}"
            )
        key = (event.capture_id, event.window_start_utc, event.window_end_utc)
        if key in predictions:
            raise DataValidationError(f"duplicate scored window: {event.capture_id}")
        predictions[key] = event.is_anomaly

    labels: dict[WindowKey, bool] = {}
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

    evaluation_by_capture = [
        {
            "capture_id": capture_id,
            "unmatched_score_count": len(
                {
                    key
                    for key in predictions.keys() - labels.keys()
                    if key[0] == capture_id
                }
            ),
            "unmatched_label_count": len(
                {
                    key
                    for key in labels.keys() - predictions.keys()
                    if key[0] == capture_id
                }
            ),
            **_statistics(
                {key for key in matched if key[0] == capture_id},
                predictions,
                labels,
            ),
        }
        for capture_id in selection.capture_ids
    ]
    report = {
        "dataset_folder": config.ingest.dataset_folder,
        "partition": selection.name,
        "capture_ids": list(selection.capture_ids),
        "score_type": config.calibration.score_type,
        "calibration_quantile": config.calibration.quantile,
        "window_minutes": config.ingest.window_minutes,
        "unmatched_score_count": len(predictions.keys() - labels.keys()),
        "unmatched_label_count": len(labels.keys() - predictions.keys()),
        **_statistics(matched, predictions, labels),
        "evaluation_by_capture": evaluation_by_capture,
        "capture_summary": _capture_summary(evaluation_by_capture),
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

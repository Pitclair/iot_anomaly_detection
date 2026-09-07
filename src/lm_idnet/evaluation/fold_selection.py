"""Summarize predefined temporal folds and select an eligible detector."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from math import isfinite
from pathlib import Path
from statistics import median
from typing import Any

SUMMARY_FIELDS = (
    "precision",
    "recall",
    "attack_episode_recall",
    "false_positive",
    "false_alert_episodes_per_day",
    "median_detection_delay_minutes",
    "ground_truth_attack_episode_count",
)


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _summary(captures: list[dict[str, Any]]) -> dict[str, object]:
    result = {}
    for field in SUMMARY_FIELDS:
        values = [capture[field] for capture in captures if capture[field] is not None]
        result[field] = (
            {
                "minimum": min(values),
                "median": median(values),
                "maximum": max(values),
            }
            if values
            else None
        )
    return result


def summarize_fold_reports(
    reports: list[dict[str, Any]],
    max_false_alert_episodes_per_day: float,
) -> dict[str, Any]:
    """Group folds by detector settings and choose the best eligible candidate."""
    if (
        not isfinite(max_false_alert_episodes_per_day)
        or max_false_alert_episodes_per_day < 0
    ):
        raise ValueError("maximum false-alert episode rate must be finite and non-negative")
    if not reports:
        raise ValueError("at least one fold report is required")

    grouped: dict[tuple[str, str, float, int], list[dict[str, Any]]] = {}
    for report in reports:
        key = (
            report["dataset_folder"],
            report["score_type"],
            report["calibration_quantile"],
            report["window_minutes"],
        )
        grouped.setdefault(key, []).append(report)

    datasets: dict[str, dict[str, Any]] = {}
    for (dataset, score_type, quantile, window_minutes), folds in grouped.items():
        captures = [
            capture
            for fold in folds
            for capture in fold["evaluation_by_capture"]
        ]
        attack_count = sum(
            capture["ground_truth_attack_episode_count"] for capture in captures
        )
        detected_count = sum(
            capture["detected_attack_episode_count"] for capture in captures
        )
        false_alert_count = sum(
            capture["false_alert_episode_count"] for capture in captures
        )
        duration_days = sum(capture["evaluated_duration_days"] for capture in captures)
        candidate = {
            "score_type": score_type,
            "calibration_quantile": quantile,
            "window_minutes": window_minutes,
        }
        candidate_report = {
            "candidate": candidate,
            "fold_count": len(folds),
            "capture_count": len(captures),
            "ground_truth_attack_episode_count": attack_count,
            "detected_attack_episode_count": detected_count,
            "attack_episode_recall": _ratio(detected_count, attack_count),
            "false_alert_episode_count": false_alert_count,
            "evaluated_duration_days": duration_days,
            "false_alert_episodes_per_day": _ratio(
                false_alert_count,
                duration_days,
            ),
            "capture_summary": _summary(captures),
            "report_paths": [fold.get("_report_path") for fold in folds],
        }
        datasets.setdefault(dataset, {"candidates": []})["candidates"].append(
            candidate_report
        )

    for dataset in datasets.values():
        candidates = dataset["candidates"]
        candidates.sort(
            key=lambda item: (
                item["candidate"]["score_type"],
                item["candidate"]["calibration_quantile"],
                item["candidate"]["window_minutes"],
            )
        )
        eligible = [
            candidate
            for candidate in candidates
            if candidate["attack_episode_recall"] is not None
            and candidate["false_alert_episodes_per_day"]
            <= max_false_alert_episodes_per_day
        ]
        selected = max(
            eligible,
            key=lambda item: (
                item["attack_episode_recall"],
                -item["false_alert_episodes_per_day"],
            ),
            default=None,
        )
        dataset["selected_candidate"] = selected["candidate"] if selected else None

    return {
        "selection_rule": {
            "objective": "highest attack-episode recall",
            "max_false_alert_episodes_per_day": max_false_alert_episodes_per_day,
        },
        "datasets": datasets,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument(
        "--max-false-alert-episodes-per-day",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/temporal-fold-summary.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        reports = []
        for path in args.reports:
            report = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(report, dict):
                raise ValueError(f"fold report must be a JSON object: {path}")
            report["_report_path"] = str(path)
            reports.append(report)
        summary = summarize_fold_reports(
            reports,
            args.max_false_alert_episodes_per_day,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps({"output": str(args.output), "status": "completed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

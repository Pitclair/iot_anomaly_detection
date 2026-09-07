"""Tests for temporal-fold summarization and candidate selection."""

import pytest

from lm_idnet.evaluation.fold_selection import summarize_fold_reports

pytestmark = pytest.mark.unit


def _report(quantile: float, recall: float, false_alerts: int) -> dict:
    attacks = 10
    capture = {
        "precision": 0.8,
        "recall": 0.7,
        "attack_episode_recall": recall,
        "false_positive": false_alerts,
        "false_alert_episodes_per_day": float(false_alerts),
        "median_detection_delay_minutes": 10.0,
        "ground_truth_attack_episode_count": attacks,
        "detected_attack_episode_count": round(attacks * recall),
        "false_alert_episode_count": false_alerts,
        "evaluated_duration_days": 1.0,
    }
    return {
        "dataset_folder": "camera",
        "score_type": "raw",
        "calibration_quantile": quantile,
        "window_minutes": 10,
        "evaluation_by_capture": [capture],
    }


def test_selects_highest_episode_recall_within_false_alert_limit() -> None:
    summary = summarize_fold_reports(
        [
            _report(0.01, 0.8, 1),
            _report(0.01, 0.6, 1),
            _report(0.05, 1.0, 4),
        ],
        max_false_alert_episodes_per_day=2.0,
    )

    dataset = summary["datasets"]["camera"]
    assert dataset["selected_candidate"] == {
        "score_type": "raw",
        "calibration_quantile": 0.01,
        "window_minutes": 10,
    }
    accepted = dataset["candidates"][0]
    assert accepted["fold_count"] == 2
    assert accepted["attack_episode_recall"] == 0.7
    assert accepted["false_alert_episodes_per_day"] == 1.0

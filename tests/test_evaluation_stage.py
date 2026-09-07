"""Tests for the minimal label-versus-decision evaluation stage."""

import json

import pytest

from lm_idnet.evaluation.evaluation_stage import evaluate_scores

pytestmark = pytest.mark.unit


def test_evaluate_scores_writes_window_and_episode_statistics(
    tmp_path,
    config_factory,
) -> None:
    processed_root = tmp_path / "data" / "processed"
    config = config_factory(
        ingest={"processed_root": processed_root},
        outputs={
            "events_path": tmp_path / "events.json",
            "reports_dir": tmp_path / "reports",
        },
    )
    capture_ids = config.ingest.partitions.development_test
    labels_dir = processed_root.parent / "labels" / config.ingest.dataset_folder
    labels_dir.mkdir(parents=True)

    events = []
    periods = (
        ("00:00", "00:10"),
        ("00:10", "00:20"),
        ("00:20", "00:30"),
        ("00:30", "00:40"),
        ("00:40", "00:50"),
        ("00:50", "01:00"),
        ("01:10", "01:20"),
    )
    truths = (True, True, True, False, True, False, False)
    predictions = (False, True, True, False, False, True, True)
    for (start, end), truth, prediction in zip(periods, truths, predictions):
        start = f"2020-10-16T{start}:00Z"
        end = f"2020-10-16T{end}:00Z"
        events.append(
            {
                "artifact_type": "anomaly_event",
                "device_id": config.ingest.device_id,
                "capture_id": capture_ids[0],
                "window_start_utc": start,
                "window_end_utc": end,
                "counts": {"tcp": 1, "udp": 0, "ssdp": 0, "arp": 0},
                "score_type": "raw",
                "score": -1.0 if prediction else 1.0,
                "threshold": 0.0,
                "is_anomaly": prediction,
                "severity": 1.0 if prediction else 0.0,
                "expected_profile": {
                    "tcp": 0.25,
                    "udp": 0.25,
                    "ssdp": 0.25,
                    "arp": 0.25,
                },
                "category_residuals": {
                    "tcp": 0.75,
                    "udp": -0.25,
                    "ssdp": -0.25,
                    "arp": -0.25,
                },
                "model_fingerprint": "0" * 64,
            }
        )
        events[-1]["truth"] = truth

    config.outputs.events_path.write_text(
        json.dumps([{k: v for k, v in event.items() if k != "truth"} for event in events]),
        encoding="utf-8",
    )
    for capture_id in capture_ids:
        windows = (
            [
                {
                    "window_start_utc": event["window_start_utc"],
                    "window_end_utc": event["window_end_utc"],
                    "is_attack": event["truth"],
                }
                for event in events
            ]
            if capture_id == capture_ids[0]
            else []
        )
        (labels_dir / f"{capture_id}.json").write_text(
            json.dumps({"capture_id": capture_id, "windows": windows}),
            encoding="utf-8",
        )

    result = evaluate_scores(config)
    report = json.loads(
        (config.outputs.reports_dir / "evaluation_statistics.json").read_text(
            encoding="utf-8"
        )
    )

    assert report == {key: value for key, value in result.items() if key != "report_path"}
    assert report["matched_window_count"] == 7
    assert report["true_positive"] == 2
    assert report["true_negative"] == 1
    assert report["false_positive"] == 2
    assert report["false_negative"] == 2
    assert report["accuracy"] == pytest.approx(3 / 7)
    assert report["precision"] == 0.5
    assert report["recall"] == 0.5
    assert report["f1_score"] == 0.5
    assert report["predicted_alert_episode_count"] == 3
    assert report["ground_truth_attack_episode_count"] == 2
    assert report["detected_attack_episode_count"] == 1
    assert report["missed_attack_episode_count"] == 1
    assert report["false_alert_episode_count"] == 2
    assert report["attack_episode_recall"] == 0.5
    assert report["evaluated_duration_days"] == pytest.approx(7 / 144)
    assert report["false_alert_episodes_per_day"] == pytest.approx(288 / 7)
    assert report["median_detection_delay_minutes"] == 20

    first_capture = report["evaluation_by_capture"][0]
    assert first_capture["capture_id"] == capture_ids[0]
    assert first_capture["matched_window_count"] == 7
    assert first_capture["ground_truth_attack_episode_count"] == 2

    empty_capture = report["evaluation_by_capture"][1]
    assert empty_capture["capture_id"] == capture_ids[1]
    assert empty_capture["matched_window_count"] == 0
    assert empty_capture["attack_episode_recall"] is None

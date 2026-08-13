"""Tests for the minimal label-versus-decision evaluation stage."""

import json

import pytest

from lm_idnet.evaluation.evaluation_stage import evaluate_scores

pytestmark = pytest.mark.unit


def test_evaluate_scores_writes_classification_statistics(
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
    truths = (True, True, False, False)
    predictions = (True, False, True, False)
    for index, (truth, prediction) in enumerate(zip(truths, predictions)):
        start = f"2020-10-16T00:{index * 10:02d}:00Z"
        end = f"2020-10-16T00:{(index + 1) * 10:02d}:00Z"
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
    assert report["matched_window_count"] == 4
    assert report["true_positive"] == 1
    assert report["true_negative"] == 1
    assert report["false_positive"] == 1
    assert report["false_negative"] == 1
    assert report["accuracy"] == 0.5
    assert report["precision"] == 0.5
    assert report["recall"] == 0.5
    assert report["f1_score"] == 0.5

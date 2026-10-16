"""Compare 10-, 20-, 30- and 60-minute detection on matched labelled traffic."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_detection import summarize
from unsw_iot_attack_pcaps.prepare_labels import label_window, load_annotations
from lm_idnet.artifacts import load_model
from lm_idnet.config import AppConfig, load_config
from lm_idnet.models.calibration_stage import calibrate_threshold
from lm_idnet.models.modeling_stage import train_model
from lm_idnet.models.scoring_stage import WindowDetector
from lm_idnet.processing.schemas import WindowRecord, validate_window_contract
from lm_idnet.processing.storage import load_processed_dataset, save_processed_dataset

BASE_MINUTES = 10
DURATIONS = (10, 20, 30, 60)
PARTITIONS = ("fit", "calibration", "development_test", "final_test")
MODES = ("static", "adaptive_threshold", "periodic_refit")


def coarsen(windows: tuple[WindowRecord, ...], minutes: int) -> list[WindowRecord]:
    """Sum complete UTC-epoch groups within one capture, preserving missingness."""
    if isinstance(minutes, bool) or not isinstance(minutes, int) or minutes <= 0 or minutes % BASE_MINUTES:
        raise ValueError("duration must be a positive multiple of ten minutes")
    if not windows:
        return []
    groups = defaultdict(list)
    categories = windows[0].categories
    previous_end = None
    for window in windows:
        validate_window_contract(window, categories, BASE_MINUTES)
        if previous_end is not None and window.start_utc < previous_end:
            raise ValueError("source windows must be ordered and non-overlapping")
        previous_end = window.end_utc
        epoch = int(window.start_utc.timestamp())
        if window.start_utc.microsecond or epoch % (BASE_MINUTES * 60):
            raise ValueError("source windows must be epoch aligned")
        start = datetime.fromtimestamp(epoch // (minutes * 60) * minutes * 60, timezone.utc)
        groups[start].append(window)
    result = []
    for start, members in sorted(groups.items()):
        if len(members) != minutes // BASE_MINUTES or any(
            member.start_utc != start + timedelta(minutes=i * BASE_MINUTES)
            for i, member in enumerate(members)
        ):
            continue
        counts = None if any(member.state == "missing" for member in members) else tuple(
            sum(member.counts[k] for member in members) for k in range(len(categories))
        )
        result.append(WindowRecord(
            start_utc=start, end_utc=start + timedelta(minutes=minutes), categories=categories,
            counts=counts, state=("missing" if counts is None else
                                 "observed" if sum(counts) else "observed-silent"),
        ))
    return result


def interval_metrics(events: list[dict], annotations: tuple) -> dict:
    """Count fully observed annotated attacks and delays to window-close alerts."""
    records = []
    partial = 0
    for interval in annotations:
        matching = [event for event in events
                    if event["start"] < interval.end and event["end"] > interval.start]
        covered = sum((min(event["end"], interval.end) - max(event["start"], interval.start)).total_seconds()
                      for event in matching)
        if not covered:
            continue
        if covered != (interval.end - interval.start).total_seconds():
            partial += 1
            continue
        alerts = [event["end"] for event in matching if event["prediction"]]
        first = min(alerts) if alerts else None
        records.append({
            "start_utc": interval.start.isoformat(), "end_utc": interval.end.isoformat(),
            "attack_type": interval.attack_type, "detected": first is not None,
            "delay_seconds": (first - interval.start).total_seconds() if first else None,
            "alert_before_attack_end": first <= interval.end if first else False,
        })
    delays = [row["delay_seconds"] for row in records if row["detected"]]
    return {
        "attack_intervals": len(records), "detected_attack_intervals": len(delays),
        "interval_recall": len(delays) / len(records) if records else None,
        "alerts_before_attack_end": sum(row["alert_before_attack_end"] for row in records),
        "median_delay_seconds": float(np.median(delays)) if delays else None,
        "partially_observed_intervals_excluded": partial, "interval_results": records,
    }


def prepare(config: AppConfig, annotations: tuple, annotation_hash: str) -> tuple[dict, dict, dict]:
    """Select the same complete observed hourly blocks at every resolution."""
    captures, labels, coverage = {}, {}, {}
    previous_end = None
    for partition in PARTITIONS:
        for capture_id in getattr(config.ingest.partitions, partition):
            path = config.ingest.processed_root / config.ingest.dataset_folder / f"{capture_id}.json"
            dataset = load_processed_dataset(path)
            if (dataset.metadata.capture_id != capture_id or dataset.metadata.partition != partition
                    or dataset.metadata.device_id != config.ingest.device_id):
                raise ValueError(f"capture metadata mismatch: {path}")
            label_path = config.ingest.processed_root.parent / "labels" / config.ingest.dataset_folder / path.name
            saved = json.loads(label_path.read_text())
            if saved["annotation_fingerprint"] != "sha256:" + annotation_hash or saved["capture_id"] != capture_id:
                raise ValueError(f"annotation provenance mismatch: {label_path}")
            saved_labels = {(row["window_start_utc"], row["window_end_utc"]): row for row in saved["windows"]}
            if len(saved_labels) != len(saved["windows"]) or len(saved_labels) != len(dataset.windows):
                raise ValueError(f"label count mismatch: {path}")
            for window in dataset.windows:
                label, _ = label_window(window.start_utc, window.end_utc, annotations)
                if saved_labels.get((label["window_start_utc"], label["window_end_utc"])) != label:
                    raise ValueError(f"saved and original annotations disagree: {path}")
            blocks = [block for block in coarsen(dataset.windows, max(DURATIONS)) if block.counts is not None]
            starts = {block.start_utc + timedelta(minutes=offset)
                      for block in blocks for offset in range(0, max(DURATIONS), BASE_MINUTES)}
            common = tuple(window for window in dataset.windows if window.start_utc in starts)
            if not common or (previous_end is not None and common[0].start_utc < previous_end):
                raise ValueError(f"empty or overlapping common capture coverage: {path}")
            previous_end = common[-1].end_utc
            captures[capture_id] = dataset.model_copy(update={"windows": common})
            labels[capture_id] = {window.start_utc: label_window(window.start_utc, window.end_utc, annotations)[0]
                                  for window in common}
            if partition in ("fit", "calibration") and any(row["is_attack"] for row in labels[capture_id].values()):
                raise ValueError("normal reference contains annotated attacks")
            coverage[capture_id] = {
                "partition": partition, "source_windows": len(dataset.windows),
                "common_10m_windows": len(common), "dropped_10m_windows": len(dataset.windows) - len(common),
                "retained_packets": [sum(window.counts[k] for window in common) for k in range(4)],
            }
    return captures, labels, coverage


def calibration_split(config: AppConfig, captures: dict) -> tuple[AppConfig, dict, dict]:
    """Reserve latest normal fit captures until the longest window can calibrate."""
    partitions = config.ingest.partitions.model_dump()
    fit, calibration = list(partitions["fit"]), list(partitions["calibration"])
    count = sum(len(coarsen(captures[cid].windows, max(DURATIONS))) for cid in calibration)
    moved = []
    while count < config.calibration.minimum_samples and len(fit) > 1:
        capture_id = fit.pop()
        calibration.insert(0, capture_id)
        moved.append(capture_id)
        count += len(coarsen(captures[capture_id].windows, max(DURATIONS)))
    if count < config.calibration.minimum_samples:
        raise ValueError("normal calibration cannot meet the minimum while retaining fit data")
    raw = config.model_dump()
    raw["ingest"]["partitions"].update(fit=fit, calibration=calibration)
    adjusted = AppConfig.model_validate(raw)
    datasets = {cid: captures[cid].model_copy(update={
        "metadata": captures[cid].metadata.model_copy(update={"partition": partition}),
    }) for partition in PARTITIONS for cid in getattr(adjusted.ingest.partitions, partition)}
    return adjusted, datasets, {
        "original_fit": list(partitions["fit"]), "original_calibration": list(partitions["calibration"]),
        "study_fit": fit, "study_calibration": calibration, "moved_normal_captures": moved,
        "calibration_windows_at_60m": count, "minimum_samples": config.calibration.minimum_samples,
    }


def run_device(config_path: Path, artifact_root: Path, writer: csv.DictWriter) -> tuple[list, dict, dict, dict]:
    config = load_config(config_path)
    annotation_path = ROOT / "unsw_iot_attack_pcaps/annotations" / (config.ingest.device_mac.replace(":", "") + ".csv")
    annotations = load_annotations(annotation_path)
    annotation_hash = hashlib.sha256(annotation_path.read_bytes()).hexdigest()
    captures, labels, coverage = prepare(config, annotations, annotation_hash)
    config, captures, split = calibration_split(config, captures)
    for capture_id, dataset in captures.items():
        coverage[capture_id]["partition"] = dataset.metadata.partition
    results = []
    source_paths = [config_path, annotation_path]
    for capture_id in captures:
        source_paths.extend([
            config.ingest.processed_root / config.ingest.dataset_folder / f"{capture_id}.json",
            config.ingest.processed_root.parent / "labels" / config.ingest.dataset_folder / f"{capture_id}.json",
        ])
    for minutes in DURATIONS:
        directory = artifact_root / config.ingest.dataset_folder / f"{minutes}m"
        processed_root = directory / "processed"
        converted = {}
        for capture_id, dataset in captures.items():
            windows = tuple(coarsen(dataset.windows, minutes))
            if [sum(window.counts[k] for window in windows) for k in range(4)] != coverage[capture_id]["retained_packets"]:
                raise ValueError("count conservation failed")
            converted[capture_id] = windows
            save_processed_dataset(dataset.model_copy(update={"windows": windows}),
                                   processed_root / config.ingest.dataset_folder / f"{capture_id}.json")
        raw = config.model_dump(mode="json")
        raw["ingest"].update(window_minutes=minutes, processed_root=str(processed_root))
        raw["outputs"].update(model_path=str(directory / "model.json"), threshold_path=str(directory / "threshold.json"),
                              events_path=str(directory / "events.json"), reports_dir=str(directory))
        for key in ("buffer_size", "update_every", "minimum_refit_windows"):
            base_minutes = raw["adaptation"][key] * config.ingest.window_minutes
            if base_minutes % minutes:
                raise ValueError("adaptation horizons must convert exactly to the chosen duration")
            raw["adaptation"][key] = base_minutes // minutes
        derived = AppConfig.model_validate(raw)
        (directory / "config.json").write_text(json.dumps(raw, indent=2) + "\n")
        training = train_model(derived)
        calibrate_threshold(derived)
        model = load_model(derived)
        for partition in ("development_test", "final_test"):
            for mode in MODES:
                detector = WindowDetector(derived, mode=mode)
                truth, predictions, scores, events, base_truth, base_predictions, base_scores = [], [], [], [], [], [], []
                promoted_thresholds = promoted_refits = rejected = 0
                for capture_id in getattr(config.ingest.partitions, partition):
                    for window in converted[capture_id]:
                        event, update, refit = detector.process(window, device_id=config.ingest.device_id, capture_id=capture_id)
                        if event is None:
                            raise ValueError("common observed coverage unexpectedly skipped")
                        prediction = event["is_anomaly"]
                        label, _ = label_window(window.start_utc, window.end_utc, annotations)
                        truth.append(label["is_attack"])
                        predictions.append(prediction)
                        scores.append(-event["score"])
                        events.append({"start": window.start_utc, "end": window.end_utc, "prediction": prediction})
                        constituent_labels = [labels[capture_id][window.start_utc + timedelta(minutes=offset)]["is_attack"]
                                              for offset in range(0, minutes, BASE_MINUTES)]
                        if any(constituent_labels) != label["is_attack"]:
                            raise ValueError("coarse and base attack labels disagree")
                        base_truth.extend(constituent_labels)
                        base_predictions.extend([prediction] * len(constituent_labels))
                        base_scores.extend([-event["score"]] * len(constituent_labels))
                        promoted_thresholds += int(update is not None and update["promoted"])
                        promoted_refits += int(refit is not None and refit["promoted"])
                        rejected += int(update is not None and not update["promoted"])
                        writer.writerow({"dataset": config.ingest.dataset_folder, "minutes": minutes,
                                         "partition": partition, "mode": mode, "capture_id": capture_id,
                                         "start_utc": window.start_utc.isoformat(), "end_utc": window.end_utc.isoformat(),
                                         "score": event["score"], "is_attack": label["is_attack"], "is_anomaly": prediction})
                metrics = summarize(truth, predictions, scores)
                hours = len(truth) * minutes / 60
                result = {
                    "dataset": config.ingest.dataset_folder, "minutes": minutes, "partition": partition,
                    "mode": mode, "policy": derived.adaptation.policy, "observed_hours": hours,
                    "fit_windows": training["matrix_shape"][0],
                    "calibration_windows": sum(len(converted[cid]) for cid in config.ingest.partitions.calibration),
                    "model_fingerprint": detector.model_fingerprint, "alpha": list(model.alpha),
                    "fit_iterations": training["iterations"],
                    "adaptation_horizons": {key: raw["adaptation"][key] for key in ("buffer_size", "update_every", "minimum_refit_windows")},
                    "effective_refit_minimum_windows": max(derived.calibration.minimum_samples, derived.adaptation.minimum_refit_windows),
                    "threshold_promotions": promoted_thresholds, "model_promotions": promoted_refits,
                    "rejected_updates": rejected, **metrics,
                    "false_alerts_per_24h": metrics["false_positive"] * 24 / hours,
                    "common_10m_metrics": summarize(base_truth, base_predictions, base_scores),
                    **interval_metrics(events, annotations),
                }
                results.append(result)
                print(json.dumps({key: result[key] for key in ("dataset", "minutes", "partition", "mode", "f1_score", "interval_recall", "median_delay_seconds", "false_alerts_per_24h")}), flush=True)
    return results, coverage, {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}, split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/window_size"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/experiments/window_size"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    baseline_paths = [Path(f"artifacts/{device}/{name}.json")
                      for device in ("UNSW-Chromecast", "UNSW-Samsung-Camera")
                      for name in ("model", "threshold", "events")]
    baseline_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in baseline_paths}
    results, coverage, inputs, splits = [], {}, {}, {}
    with (args.output / "decisions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "minutes", "partition", "mode", "capture_id", "start_utc", "end_utc", "score", "is_attack", "is_anomaly"])
        writer.writeheader()
        for config in (Path("configs/unsw_chromecast.json"), Path("configs/unsw_samsung_camera.json")):
            device_results, device_coverage, hashes, split = run_device(config, args.artifacts, writer)
            results.extend(device_results)
            coverage[device_results[0]["dataset"]] = device_coverage
            splits[device_results[0]["dataset"]] = split
            inputs.update(hashes)
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != value for path, value in baseline_hashes.items()):
        raise ValueError("reference detector artifacts changed during the experiment")
    code_paths = [Path(__file__), Path("scripts/evaluate_detection.py"),
                  Path("unsw_iot_attack_pcaps/prepare_labels.py"), Path("src/lm_idnet/models/modeling_stage.py"),
                  Path("src/lm_idnet/models/calibration_stage.py"), Path("src/lm_idnet/models/scoring_stage.py")]
    report = {
        "protocol": "10/20/30/60-minute sensitivity study on identical capture-local, epoch-aligned observed coverage. Latest whole normal fit captures extend calibration where needed to meet the unchanged 30-sample guard at 60 minutes; the same resulting normal split is used at all four durations. Fresh fit/calibration at every duration; unchanged detector and conservative policy. Development comparison and retrospective later audit are separate; no locked-test claim.",
        "metrics": "Any attack overlap labels a window positive. Common-10m metrics project each decision onto its constituent reference bins for comparable localisation. Fully observed official attack intervals define attack coverage; delay runs from attack onset to first overlapping alert's window end, including alerts after attack end. False alerts are per 24 observed hours.",
        "adaptation": "48 observed hours of buffer, 24 observed hours between attempts, and nominal 24 accepted hours before refitting. The unchanged conservative 30-sample guard raises the effective accepted-data minimum to 30 hours at 60-minute duration. Each partition/mode starts independently.",
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": inputs,
        "code_sha256": {str(path.relative_to(ROOT) if path.is_absolute() else path): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths},
        "baseline_sha256": baseline_hashes, "normal_splits": splits, "coverage": coverage, "results": results,
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (args.output / "plot_data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["minutes", "chromecast_f1", "samsung_f1", "chromecast_delay", "samsung_delay"])
        writer.writeheader()
        for minutes in DURATIONS:
            rows = {row["dataset"]: row for row in results
                    if row["minutes"] == minutes and row["partition"] == "development_test" and row["mode"] == "static"}
            writer.writerow({"minutes": minutes,
                             "chromecast_f1": rows["UNSW-Chromecast"]["f1_score"] * 100,
                             "samsung_f1": rows["UNSW-Samsung-Camera"]["f1_score"] * 100,
                             "chromecast_delay": rows["UNSW-Chromecast"]["median_delay_seconds"] / 60,
                             "samsung_delay": rows["UNSW-Samsung-Camera"]["median_delay_seconds"] / 60})


if __name__ == "__main__":
    main()

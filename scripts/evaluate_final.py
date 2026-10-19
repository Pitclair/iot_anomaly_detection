"""Complete the final-partition evaluation of the selected ten-minute detector."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_detection import summarize
from scripts.evaluate_window_sizes import interval_metrics
from unsw_iot_attack_pcaps.prepare_labels import label_window, load_annotations
from lm_idnet.config import load_config
from lm_idnet.models.scoring_stage import WindowDetector
from lm_idnet.partitioning import final_partition_for_locked_evaluation
from lm_idnet.processing.schemas import validate_window_contract
from lm_idnet.processing.storage import load_processed_dataset

MODES = ("static", "adaptive_threshold", "periodic_refit")
OUTPUT = ROOT / "reports/final_evaluation"
ARTIFACTS = ROOT / "artifacts/final_evaluation"


def load_final(config, annotations, annotation_hash):
    """Require exact labelled final coverage, provenance and chronological windows."""
    samples = []
    previous_end = None
    for capture_id in final_partition_for_locked_evaluation(config).capture_ids:
        dataset = load_processed_dataset(
            config.ingest.processed_root / config.ingest.dataset_folder / f"{capture_id}.json")
        if (dataset.metadata.capture_id != capture_id or dataset.metadata.partition != "final_test"
                or dataset.metadata.device_id != config.ingest.device_id):
            raise ValueError("final capture metadata mismatch")
        path = config.ingest.processed_root.parent / "labels" / config.ingest.dataset_folder / f"{capture_id}.json"
        saved = json.loads(path.read_text())
        if (saved.get("capture_id") != capture_id or saved.get("device_id") != config.ingest.device_id
                or saved.get("annotation_fingerprint") != "sha256:" + annotation_hash):
            raise ValueError("final label provenance mismatch")
        labels = {(row["window_start_utc"], row["window_end_utc"]): row for row in saved["windows"]}
        if len(labels) != len(saved["windows"]) or len(labels) != len(dataset.windows):
            raise ValueError("final labels must match windows exactly without duplicates")
        for window in dataset.windows:
            validate_window_contract(window, config.ingest.categories, config.ingest.window_minutes)
            if previous_end is not None and window.start_utc < previous_end:
                raise ValueError("final windows overlap or are out of order")
            previous_end = window.end_utc
            expected, _ = label_window(window.start_utc, window.end_utc, annotations)
            key = (expected["window_start_utc"], expected["window_end_utc"])
            if labels.get(key) != expected or not isinstance(labels[key]["is_attack"], bool):
                raise ValueError("final labels disagree with original annotations")
            samples.append((capture_id, window, expected["is_attack"]))
    if not samples:
        raise ValueError("no final windows selected")
    return samples


def hashes(paths):
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths))}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    configs = [load_config(ROOT / f"configs/{name}.json")
               for name in ("unsw_chromecast", "unsw_samsung_camera")]
    paths = list((ROOT / "src/lm_idnet").rglob("*.py")) + [
        Path(__file__), ROOT / "scripts/evaluate_detection.py", ROOT / "scripts/evaluate_window_sizes.py",
        ROOT / "unsw_iot_attack_pcaps/prepare_labels.py"]
    for config, name in zip(configs, ("unsw_chromecast", "unsw_samsung_camera")):
        paths.extend([ROOT / f"configs/{name}.json", ROOT / config.outputs.model_path,
                      ROOT / config.outputs.threshold_path,
                      ROOT / "unsw_iot_attack_pcaps/annotations" / (config.ingest.device_mac.replace(":", "") + ".csv")])
        for capture_id in config.ingest.partitions.all_capture_ids():
            paths.append(ROOT / config.ingest.processed_root / config.ingest.dataset_folder / f"{capture_id}.json")
            paths.append(ROOT / config.ingest.processed_root.parent / "labels" / config.ingest.dataset_folder / f"{capture_id}.json")
    frozen_hashes = hashes(paths)
    protocol = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "protocol": "Selected 10-minute joint DM/volume/ARP scoring; saved normal-only models and thresholds; no training, recalibration or candidate selection. Independent chronological replay from the same starting pair for each conservative mode; score before learn, labels only for offline evaluation.",
        "evidence_boundary": "Evaluation covers the configured labelled final partitions for Chromecast and Samsung camera. Starting models and thresholds use normal fitting/calibration partitions; final labels are used only for offline metrics, with no parameter selection during the final run.",
        "modes": MODES,
        "input_and_code_sha256": frozen_hashes,
        "artifacts": str(ARTIFACTS.relative_to(ROOT)),
    }
    # Write the chosen protocol before scoring; reruns retain the same selected method.
    write_json(OUTPUT / "protocol.json", protocol)
    results, pooled = [], {mode: ([], [], []) for mode in MODES}
    for config in configs:
        if config.ingest.window_minutes != 10 or config.adaptation.policy != "conservative":
            raise ValueError("final evaluation requires the selected ten-minute conservative settings")
        dataset = config.ingest.dataset_folder
        directory = ARTIFACTS / dataset
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("model", "threshold"):
            shutil.copyfile(getattr(config.outputs, name + "_path"), directory / f"{name}.json")
        config.outputs.model_path = directory.relative_to(ROOT) / "model.json"
        config.outputs.threshold_path = directory.relative_to(ROOT) / "threshold.json"
        write_json(directory / "config.json", config.model_dump(mode="json"))
        annotation_path = ROOT / "unsw_iot_attack_pcaps/annotations" / (config.ingest.device_mac.replace(":", "") + ".csv")
        annotations = load_annotations(annotation_path)
        samples = load_final(config, annotations, hashlib.sha256(annotation_path.read_bytes()).hexdigest())
        for mode in MODES:
            detector = WindowDetector(config, mode=mode)
            events, updates, refits, truth, predictions, scores, timed = [], [], [], [], [], [], []
            for capture_id, window, label in samples:
                event, update, refit = detector.process(window, device_id=config.ingest.device_id, capture_id=capture_id)
                if update is not None:
                    updates.append(update)
                if refit is not None:
                    refits.append(refit)
                if event is None:
                    continue
                events.append(event)
                truth.append(label)
                predictions.append(event["is_anomaly"])
                scores.append(-event["score"])
                timed.append({"start": window.start_utc, "end": window.end_utc, "prediction": event["is_anomaly"]})
            assert len(events) == sum(window.counts is not None for _, window, _ in samples)
            write_json(directory / f"{mode}_events.json", events)
            write_json(directory / f"{mode}_updates.json", {"threshold_attempts": updates, "refit_attempts": refits})
            result = {
                "dataset": dataset, "partition": "final_test", "mode": mode, "policy": "conservative",
                "capture_ids": list(config.ingest.partitions.final_test),
                "missing_windows_excluded": len(samples) - len(events),
                "threshold_updates": sum(row["promoted"] for row in updates),
                "model_refits": sum(row["promoted"] for row in refits),
                **summarize(truth, predictions, scores), **interval_metrics(timed, annotations),
            }
            results.append(result)
            for target, values in zip(pooled[mode], (truth, predictions, scores)):
                target.extend(values)
            print(json.dumps(result, allow_nan=False), flush=True)
    assert hashes(paths) == frozen_hashes, "frozen source inputs changed during evaluation"
    write_json(OUTPUT / "results.json", {
        "protocol": "protocol.json", "results": results,
        "pooled": [{"mode": mode, **summarize(*pooled[mode])} for mode in MODES],
        "rank_metric_boundary": "Per-device ROC AUC is reported; pooled ROC AUC combines device-calibrated margins and is descriptive only.",
        "frozen_inputs_unchanged": True,
    })


if __name__ == "__main__":
    main()

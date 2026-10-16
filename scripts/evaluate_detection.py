"""Compare small scoring changes with fixed normal-data rules and detector settings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.stats import rankdata

from lm_idnet.algorithms.dirichlet_multinomial import anomaly_score
from lm_idnet.artifacts import load_artifact, save_artifact
from lm_idnet.config import load_config
from lm_idnet.models.calibration_stage import calibrate_threshold, load_calibration_windows
from lm_idnet.models.scoring_stage import WindowDetector
from lm_idnet.processing.storage import load_processed_dataset


def summarize(labels: list[bool], predictions: list[bool], scores: list[float]) -> dict:
    """Report decisions and rank AUC; larger scores mean stronger anomaly evidence."""
    y = np.asarray(labels, dtype=bool)
    p = np.asarray(predictions, dtype=bool)
    if not len(y):
        return {"window_count": len(p), "alert_count": int(p.sum())}
    if len(y) != len(p):
        raise ValueError("every scored window must have a label")
    tp, fp = int((p & y).sum()), int((p & ~y).sum())
    fn, tn = int((~p & y).sum()), int((~p & ~y).sum())
    positives, negatives = int(y.sum()), int((~y).sum())
    ranks = rankdata(scores)
    return {
        "window_count": len(p), "alert_count": int(p.sum()),
        "true_positive": tp, "false_positive": fp,
        "false_negative": fn, "true_negative": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / positives if positives else None,
        "false_positive_rate": fp / negatives if negatives else None,
        "f1_score": 2 * tp / (2 * tp + fp + fn) if positives else None,
        "roc_auc": (
            float((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))
            if positives and negatives else None
        ),
    }


def compare(config_path: Path, partition: str, variants: list[str], work: Path) -> list[dict]:
    config = load_config(config_path)
    config = config.model_copy(update={"outputs": config.outputs.model_copy(update={
        "threshold_path": work / "threshold.json",
    })})
    calibrate_threshold(config)
    pooled = load_artifact(config.outputs.threshold_path, expected_type="threshold")
    initial = WindowDetector(config)
    calibration = load_calibration_windows(config, initial.model_categories)
    counts = np.asarray([window.counts for window in calibration], dtype=np.int64)
    capture_ids = getattr(config.ingest.partitions, partition)
    samples = []
    label_directory = config.ingest.processed_root.parent / "labels" / config.ingest.dataset_folder
    labeled = label_directory.is_dir()
    for capture_id in capture_ids:
        dataset = load_processed_dataset(
            config.ingest.processed_root / config.ingest.dataset_folder / f"{capture_id}.json",
        )
        if dataset.metadata.partition != partition or dataset.metadata.capture_id != capture_id:
            raise ValueError("evaluation capture does not match its partition/filename")
        labels = {}
        if labeled:
            raw_labels = json.loads((label_directory / f"{capture_id}.json").read_text())
            if raw_labels["capture_id"] != capture_id:
                raise ValueError("label capture does not match its filename")
            labels = {
                (row["window_start_utc"], row["window_end_utc"]): row["is_attack"]
                for row in raw_labels["windows"]
            }
            if len(labels) != len(raw_labels["windows"]) or any(
                not isinstance(value, bool) for value in labels.values()
            ):
                raise ValueError("evaluation labels must be unique boolean window labels")
        samples.extend((capture_id, dataset.metadata.device_id, window, labels)
                       for window in dataset.windows)

    results = []
    for variant in variants:
        score_type = "normalized" if variant == "normalized" else config.calibration.score_type
        calibration_scores = np.asarray([
            anomaly_score(row, initial.alpha, initial.log_likelihood, score_type) for row in counts
        ])
        threshold = pooled.model_copy(deep=True)
        if variant in ("baseline", "normalized"):
            threshold.count_tails = {}
            q1, q3 = np.quantile(calibration_scores, (.25, .75))
            threshold.score_type = score_type
            threshold.threshold = float(np.quantile(calibration_scores, threshold.quantile))
            threshold.score_iqr = float(q3 - q1)
            threshold.score_minimum = float(calibration_scores.min())
            threshold.score_median = float(np.median(calibration_scores))
            threshold.score_maximum = float(calibration_scores.max())
        save_artifact(config.outputs.threshold_path, threshold)
        for mode, policy in (
            ("static", "conservative"),
            ("adaptive_threshold", "conservative"), ("periodic_refit", "conservative"),
            ("adaptive_threshold", "legacy"), ("periodic_refit", "legacy"),
        ):
            run_config = config.model_copy(update={
                "calibration": config.calibration.model_copy(update={"score_type": score_type}),
                "adaptation": config.adaptation.model_copy(update={"mode": mode, "policy": policy}),
            })
            detector = WindowDetector(run_config)
            truth, predictions, scores, updates, refits = [], [], [], [], []
            for capture_id, device_id, window, label_map in samples:
                iqr = detector.current_score_iqr
                event, update, refit = detector.process(
                    window, device_id=device_id, capture_id=capture_id,
                )
                if update is not None:
                    updates.append(update)
                if refit is not None:
                    refits.append(refit)
                if event is None:
                    continue
                predictions.append(event["is_anomaly"])
                scores.append(-event["score"] if event["feature_margins"] else
                              (event["threshold"] - event["score"]) / iqr)
                if labeled:
                    truth.append(label_map[(event["window_start_utc"], event["window_end_utc"])])
            result = {
                "dataset": config.ingest.dataset_folder, "partition": partition,
                "variant": variant, "mode": mode, "policy": policy,
                "model_fingerprint": initial.model_fingerprint,
                "threshold_updates": sum(update["promoted"] for update in updates),
                "model_refits": sum(refit["promoted"] for refit in refits),
                "rejected_updates": [update["reason"] for update in updates if not update["promoted"]],
                **summarize(truth, predictions, scores),
            }
            results.append(result)
            print(json.dumps(result, allow_nan=False), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", type=Path, default=[
        Path("configs/unsw_chromecast.json"), Path("configs/unsw_samsung_camera.json"),
        Path("configs/d_link_day_cam5.json"),
    ])
    parser.add_argument("--partition", choices=("development_test", "final_test"), default="development_test")
    parser.add_argument("--output", type=Path, default=Path("reports/detection_experiments.json"))
    args = parser.parse_args()
    variants = (["baseline", "hybrid"] if args.partition == "final_test" else
                ["baseline", "normalized", "hybrid"])
    with TemporaryDirectory(prefix="lm-detection-") as directory:
        results = [row for path in args.configs
                   for row in compare(path, args.partition, variants, Path(directory))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "protocol": "Retrospective comparison on existing labeled traces; normal-only rules and fixed settings.",
        "calibration": "Fit-only medians/IQRs for DM, log1p packet volume, log1p ARP; joint normal-calibration quantile.",
        "adaptation": "Frozen volume/ARP tails and reference DM; independent chronological replay per variant and mode.",
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), Path("src/lm_idnet/models/calibration_stage.py"),
                         Path("src/lm_idnet/models/scoring_stage.py"), Path("src/lm_idnet/models/schemas.py")]
        },
        "results": results,
    }, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

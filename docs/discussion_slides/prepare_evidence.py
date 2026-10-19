"""Regenerate slide values from retained results; run from any directory."""

import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def read(relative):
    return json.loads((ROOT / relative).read_text())


def coordinates(points):
    return " ".join(f"({x},{y:.8g})" for x, y in points)


def main():
    final = read("reports/final_evaluation/results.json")
    devices = ["UNSW-Chromecast", "UNSW-Samsung-Camera"]
    static = [next(r for r in final["results"] if r["dataset"] == d
                   and r["mode"] == "static") for d in devices]
    pooled = next(r for r in final["pooled"] if r["mode"] == "static")
    counts = ["true_positive", "false_positive", "false_negative", "true_negative"]
    for key in [*counts, "window_count"]:
        assert sum(r[key] for r in static) == pooled[key]
    for result in final["results"]:
        original = static[devices.index(result["dataset"])]
        assert all(result[k] == original[k] for k in counts)
        assert result["threshold_updates"] == result["model_refits"] == 0
    assert pooled["window_count"] == 1008
    assert sum(pooled[k] for k in counts) == 1008
    assert sum(r["attack_intervals"] for r in static) == 9
    assert sum(r["detected_attack_intervals"] for r in static) == 8
    assert sum(r["alerts_before_attack_end"] for r in static) == 7

    macros = {}
    for key, metric in [("Precision", "precision"), ("Recall", "recall"),
                        ("Fone", "f1_score")]:
        macros[f"Final{key}"] = coordinates(
            (i, 100 * r[metric]) for i, r in enumerate([*static, pooled], 1))
    for key, metric in [("TP", "true_positive"), ("FP", "false_positive"),
                        ("FN", "false_negative")]:
        macros[f"Final{key}"] = coordinates(
            (i, r[metric]) for i, r in enumerate(static, 1))
    for short, device in zip(["Chromecast", "Samsung"], devices):
        alpha = read(f"artifacts/final_evaluation/{device}/model.json")["alpha"]
        macros[f"Profile{short}"] = coordinates(
            (i, 100 * value / sum(alpha)) for i, value in enumerate(alpha, 1))

    study = read("reports/window_size/results.json")["results"]
    detected, timely, fone = [], [], []
    for minutes in [10, 20, 30, 60]:
        rows = [r for r in study if r["minutes"] == minutes
                and r["partition"] == "development_test" and r["mode"] == "static"]
        assert len(rows) == 2 and sum(r["attack_intervals"] for r in rows) == 54
        tp, fp, fn = (sum(r[k] for r in rows) for k in counts[:3])
        detected.append((minutes, sum(r["detected_attack_intervals"] for r in rows)))
        timely.append((minutes, sum(r["alerts_before_attack_end"] for r in rows)))
        fone.append((minutes, 100 * 2 * tp / (2 * tp + fp + fn)))
    macros["WindowDetected"] = coordinates(detected)
    macros["WindowTimely"] = coordinates(timely)
    macros["WindowFone"] = coordinates(fone)
    for short, device in zip(["Chromecast", "Samsung"], devices):
        rows = sorted((r for r in study if r["dataset"] == device
                       and r["partition"] == "development_test"
                       and r["mode"] == "static"), key=lambda r: r["minutes"])
        macros[f"Delay{short}"] = coordinates(
            (r["minutes"], r["median_delay_seconds"] / 60) for r in rows)

    rejected = Counter()
    for device in devices:
        for mode in ["adaptive_threshold", "periodic_refit"]:
            updates = read(f"artifacts/final_evaluation/{device}/{mode}_updates.json")
            # A refit also has a paired threshold record; count each attempt once.
            attempts = updates["threshold_attempts" if mode == "adaptive_threshold"
                               else "refit_attempts"]
            assert not any(a["promoted"] for a in attempts)
            rejected.update(a["reason"] for a in attempts)
    assert sum(rejected.values()) == 14
    macros["Rejections"] = coordinates(enumerate([
        rejected["candidate threshold change exceeds configured limit"],
        rejected["candidate concentration change exceeds configured limit"],
        rejected["not enough accepted windows"],
    ], 1))

    benchmark = read("reports/D-LinkDayCam5/lm_backend_benchmark.json")
    assert benchmark["comparisons"]["decision_mismatch_count"] == 0
    metrics = ["model_fit_median_seconds", "likelihood_matrix_median_seconds",
               "single_window_score_median_seconds"]
    macros["BackendRatios"] = coordinates(
        (i, benchmark["backends"]["lm"][m] / benchmark["backends"]["scipy"][m])
        for i, m in enumerate(metrics, 1))
    edge = read("reports/edge/simulation-20000.json")
    assert edge["input_windows"] == 20000
    assert edge["measurements"]["scored_windows"] == 16000
    assert edge["measurements"]["promoted_updates"] == 0
    macros["EdgeMemory"] = coordinates(
        (s["after_input_windows"] / 1000, s["rss_bytes"] / 2**20)
        for s in edge["samples"])
    macros["EdgePeak"] = f"{max(s['rss_bytes'] for s in edge['samples']) / 2**20:.2f}"
    macros["EdgeWorst"] = f"{edge['measurements']['latency_max_seconds']:.3f}"

    # Align each measured attack to its onset; bars show its annotated duration.
    from datetime import datetime

    intervals = []
    for prefix, row in zip(["C", "S"], static):
        for i, attack in enumerate(sorted(row["interval_results"],
                                         key=lambda a: a["start_utc"]), 1):
            duration = (datetime.fromisoformat(attack["end_utc"])
                        - datetime.fromisoformat(attack["start_utc"])).total_seconds() / 60
            intervals.append((f"{prefix}{i}", duration, attack["delay_seconds"]))
    macros["IncidentLabels"] = ",".join(label for label, _, _ in reversed(intervals))
    bars, alerts = [], []
    for y, (_, duration, delay) in enumerate(reversed(intervals), 1):
        bars.append(r"\draw[palerose,line width=5pt] (axis cs:0,"
                    + str(y) + f") -- (axis cs:{duration:.6g},{y});")
        if delay is None:
            bars.append(r"\node[anchor=west,font=\small,text=charcoal] at "
                        + f"(axis cs:{duration + .3:.6g},{y}) " + r"{missed};")
        else:
            alerts.append((delay / 60, y))
    macros["IncidentBars"] = "\n".join(bars)
    macros["IncidentAlerts"] = coordinates(alerts)

    output = "% Generated by prepare_evidence.py from retained JSON; do not edit.\n"
    output += "\n".join("\\newcommand{\\" + key + "}{" + value + "}"
                         for key, value in macros.items()) + "\n"
    (HERE / "evidence.tex").write_text(output)
    print("Verified final counts, mode agreement, update attempts, and study coverage.")
    print("Wrote", HERE / "evidence.tex")


if __name__ == "__main__":
    main()

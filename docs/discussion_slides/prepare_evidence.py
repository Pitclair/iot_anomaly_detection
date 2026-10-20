"""Regenerate slide values from retained results; run from any directory."""

import json
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
    counts = ["true_positive", "false_positive", "false_negative", "true_negative"]
    for row, windows, attacks in zip(static, [432, 576], [6, 12]):
        assert sum(row[k] for k in counts) == row["window_count"] == windows
        assert row["true_positive"] + row["false_negative"] == attacks
    for result in final["results"]:
        original = static[devices.index(result["dataset"])]
        assert all(result[k] == original[k] for k in counts)
        assert result["threshold_updates"] == result["model_refits"] == 0

    macros = {}
    for short, device in zip(["Chromecast", "Samsung"], devices):
        for name, mode in [("Static", "static"), ("Periodic", "periodic_refit")]:
            row = next(r for r in final["results"]
                       if r["dataset"] == device and r["mode"] == mode)
            macros[f"Final{name}{short}"] = coordinates(
                (i, 100 * row[metric]) for i, metric in enumerate(
                    ["precision", "recall", "f1_score"], 1))
    for key, metric in [("TP", "true_positive"), ("FP", "false_positive"),
                        ("FN", "false_negative")]:
        macros[f"Final{key}"] = coordinates(
            (i, r[metric]) for i, r in enumerate(static, 1))
    for short, device in zip(["Chromecast", "Samsung"], devices):
        alpha = read(f"artifacts/final_evaluation/{device}/model.json")["alpha"]
        macros[f"Profile{short}"] = coordinates(
            (i, 100 * value / sum(alpha)) for i, value in enumerate(alpha, 1))

    study = read("reports/window_size/results.json")["results"]
    for short, device, attacks in zip(["Chromecast", "Samsung"], devices, [18, 36]):
        rows = sorted((r for r in study if r["dataset"] == device
                       and r["partition"] == "development_test"
                       and r["mode"] == "static"), key=lambda r: r["minutes"])
        assert [r["minutes"] for r in rows] == [10, 20, 30, 60]
        assert all(r["attack_intervals"] == attacks for r in rows)
        for key, metric, factor in [
            ("WindowDetected", "detected_attack_intervals", 1),
            ("WindowTimely", "alerts_before_attack_end", 1),
            ("WindowFone", "f1_score", 100),
        ]:
            macros[f"{key}{short}"] = coordinates(
                (r["minutes"], factor * r[metric]) for r in rows)
        macros[f"Delay{short}"] = coordinates(
            (r["minutes"], r["median_delay_seconds"] / 60) for r in rows)

    native = read("reports/D-LinkDayCam5/lm_backend_benchmark.json")["backends"]["lm"]
    macros["LmFit"] = f"{1000 * native['model_fit_median_seconds']:.6g}"
    macros["LmKernelTimes"] = coordinates([
        (1, 1e6 * native["likelihood_matrix_median_seconds"]),
        (2, 1e6 * native["single_window_score_median_seconds"]),
    ])
    edge = read("reports/edge/simulation-20000.json")
    assert edge["config"]["adaptation"]["mode"] == "periodic_refit"
    assert edge["config"]["adaptation"]["policy"] == "conservative"
    assert edge["config"]["ingest"]["device_id"] == "D-LinkDayCam5"
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
    print("Verified final counts, mode agreement, inactive updates, and study coverage.")
    print("Wrote", HERE / "evidence.tex")


if __name__ == "__main__":
    main()

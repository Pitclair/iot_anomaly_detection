#!/usr/bin/env python3
"""Measure the real streaming path on a finite, synthetic window replay."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter, process_time
from unittest.mock import patch

PROCESS_STARTED = perf_counter()

import lm_idnet
from lm_idnet.cli import _configure_logging
from lm_idnet.config import load_edge_config
from lm_idnet.models import scoring_stage
from lm_idnet.runtime import InputRecord, run_windows


def current_rss():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024


def disk_sizes(path):
    return {str(item.relative_to(path)): item.stat().st_size
            for item in sorted(path.rglob("*")) if item.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=10000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.windows <= 0:
        parser.error("--windows must be positive")
    config = load_edge_config(args.config)
    config.state_dir = args.state_dir
    log_path = _configure_logging(False, bounded=True)
    templates = [InputRecord.model_validate(json.loads(line)).model_dump(mode="json")
                 for line in args.fixture.read_text().splitlines()]
    if not templates:
        parser.error("fixture must contain windows")
    samples = []
    stats = {"scored_windows": 0, "latency_total_seconds": 0., "latency_max_seconds": 0.,
             "update_attempts": 0, "promoted_updates": 0, "refit_attempts": 0,
             "estimator_calls": 0, "fit_total_seconds": 0., "fit_max_seconds": 0.,
             "refit_window_total_seconds": 0., "refit_window_max_seconds": 0.}

    class Replay:
        position = 0
        available_at = 0.
        refit = False
        startup_seconds = None

        def readline(self, size):
            if self.position == args.windows:
                return b""
            if self.startup_seconds is None:
                self.startup_seconds = perf_counter() - PROCESS_STARTED
            if self.position % max(1, args.windows // 8) == 0:
                samples.append({"after_input_windows": self.position, "rss_bytes": current_rss(),
                                "state_bytes": sum(disk_sizes(config.state_dir).values()),
                                "log_bytes": sum(disk_sizes(log_path.parent).values())})
            template = templates[self.position % len(templates)]
            start = datetime(2026, 10, 15, tzinfo=timezone.utc) + timedelta(
                minutes=self.position * config.ingest.window_minutes,
            )
            value = {**template, "window": {**template["window"],
                     "start_utc": start.isoformat(),
                     "end_utc": (start + timedelta(minutes=config.ingest.window_minutes)).isoformat()}}
            encoded = (json.dumps(value) + "\n").encode()
            if len(encoded) > size - 1:
                raise ValueError("fixture record exceeds runtime line limit")
            self.position += 1
            self.refit = False
            self.available_at = perf_counter()
            return encoded

    source = Replay()

    class Events:
        def write(self, value):
            return sys.stdout.write(value)

        def flush(self):
            sys.stdout.flush()
            elapsed = perf_counter() - source.available_at
            stats["scored_windows"] += 1
            stats["latency_total_seconds"] += elapsed
            stats["latency_max_seconds"] = max(stats["latency_max_seconds"], elapsed)
            if source.refit:
                stats["refit_window_total_seconds"] += elapsed
                stats["refit_window_max_seconds"] = max(stats["refit_window_max_seconds"], elapsed)

    original_process = scoring_stage.WindowDetector.process
    original_factory = scoring_stage.create_estimator

    def observed_process(detector, *arguments, **keywords):
        result = original_process(detector, *arguments, **keywords)
        _, update, refit = result
        source.refit = refit is not None
        stats["refit_attempts"] += int(source.refit)
        stats["update_attempts"] += int(update is not None)
        stats["promoted_updates"] += int(update is not None and update["promoted"])
        return result

    def observed_factory(*arguments, **keywords):
        estimator = original_factory(*arguments, **keywords)
        original_fit = estimator.fit

        def fit(matrix):
            started = perf_counter()
            try:
                return original_fit(matrix)
            finally:
                elapsed = perf_counter() - started
                stats["estimator_calls"] += 1
                stats["fit_total_seconds"] += elapsed
                stats["fit_max_seconds"] = max(stats["fit_max_seconds"], elapsed)

        estimator.fit = fit
        return estimator

    started, cpu_started = perf_counter(), process_time()
    with patch.object(scoring_stage.WindowDetector, "process", observed_process), \
            patch.object(scoring_stage, "create_estimator", observed_factory):
        run_windows(config, source, Events())
    elapsed, cpu_seconds = perf_counter() - started, process_time() - cpu_started
    if source.position != args.windows:
        raise RuntimeError("benchmark interrupted before the complete replay")
    samples.append({"after_input_windows": source.position, "rss_bytes": current_rss(),
                    "state_bytes": sum(disk_sizes(config.state_dir).values()),
                    "log_bytes": sum(disk_sizes(log_path.parent).values())})
    package_root = Path(lm_idnet.__file__).parent
    digest = hashlib.sha256()
    for file in sorted(package_root.rglob("*.py")):
        digest.update(str(file.relative_to(package_root)).encode() + b"\0" + file.read_bytes())
    report = {
        "scope": "local Docker simulation; supplied windows; no physical gateway claim",
        "environment": {"platform": platform.platform(), "architecture": platform.machine(),
            "container_os": platform.freedesktop_os_release(),
            "cpu_model": next((line.split(":", 1)[1].strip() for line in
                Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), None),
            "cpu_quota": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
            "memory_limit_bytes": int(Path("/sys/fs/cgroup/memory.max").read_text()),
            "python": platform.python_version(), "source_revision": os.environ.get("LM_IDNET_REVISION"),
            "package_python_sha256": digest.hexdigest(),
            "measurement_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "native_library_sha256": hashlib.sha256((package_root / "algorithms/native/LM_time_lib.so").read_bytes()).hexdigest(),
            "dependencies": {item.metadata["Name"]: item.version for item in distributions()}},
        "config": config.model_dump(mode="json"),
        "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "input_windows": source.position, "startup_seconds": source.startup_seconds,
        "elapsed_seconds": elapsed, "cpu_seconds": cpu_seconds,
        "cpu_percent": cpu_seconds / elapsed * 100,
        "throughput_scored_windows_per_second": stats["scored_windows"] / elapsed,
        "mean_processing_seconds": stats["latency_total_seconds"] / stats["scored_windows"],
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "samples": samples, "measurements": stats,
        "state_files_bytes": disk_sizes(config.state_dir), "log_files_bytes": disk_sizes(log_path.parent),
        "method": "Line available to event flush, including validation, scoring, updates, serialization and Docker stdout; excludes traffic collection delay. Counters and at most ten resource samples; no event history retained.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

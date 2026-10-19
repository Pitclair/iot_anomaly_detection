"""Streaming input, event delivery, and CLI failure checks with prepared state."""

from datetime import datetime, timedelta, timezone
import io
import json
import logging
import signal
import os
from pathlib import Path
import select
import subprocess
import sys
from time import sleep

import pytest

from lm_idnet import cli, runtime
from lm_idnet.artifacts import detector_state_lock, load_detector_state, save_detector_state
from lm_idnet.config import load_config
from lm_idnet.exceptions import DataValidationError, DetectorStateCommitError, PolicyRejectionError
from lm_idnet.models.schemas import DetectorStateArtifact
from lm_idnet.models.scoring_stage import WindowDetector
from lm_idnet.processing.schemas import MAX_WINDOW_COUNT, WindowRecord

pytestmark = pytest.mark.unit
def record(config, offset=0, counts=(100, 20, 3, 2)):
    start = datetime(2026, 10, 15, 12, tzinfo=timezone.utc) + timedelta(minutes=offset)
    window = WindowRecord(
        start_utc=start, end_utc=start + timedelta(minutes=config.ingest.window_minutes),
        categories=config.ingest.categories, counts=counts,
        state="missing" if counts is None else "observed" if sum(counts) else "observed-silent",
    )
    return {"device_id": config.ingest.device_id, "source_id": "local-collector",
            "window": window.model_dump(mode="json")}


def line(value):
    return (json.dumps(value) + "\n").encode()


def arguments(path, *extra):
    return [sys.executable, "-m", "lm_idnet.cli", "run", "--config", str(path), *extra]


def environment():
    return dict(os.environ, LM_IDNET_LOG_PATH=os.devnull)


def test_stream_matches_shared_detector_and_distinguishes_silence_missing_and_gaps(edge_runtime):
    config, _ = edge_runtime
    records = [record(config), record(config, 10, (0, 0, 0, 0)),
               record(config, 20, None), record(config, 50)]
    expected = []
    with detector_state_lock(config.state_dir):
        detector = WindowDetector(config)
        for value in records:
            event, _, _ = detector.process(
                WindowRecord.model_validate(value["window"]),
                device_id=value["device_id"], capture_id=value["source_id"],
            )
            if event is not None:
                expected.append(event)
    output = io.StringIO()
    runtime.run_windows(config, io.BytesIO(b"".join(map(line, records))), output)
    assert [json.loads(value) for value in output.getvalue().splitlines()] == expected
    assert len(expected) == 3
    assert set(expected[1]["counts"].values()) == {0}
    with detector_state_lock(config.state_dir):
        pass


@pytest.mark.parametrize("dataset,total", [("unsw_chromecast", 5_000_000),
                                           ("unsw_samsung_camera", 10_000_000)])
@pytest.mark.parametrize("mode", ["static", "adaptive_threshold", "periodic_refit"])
def test_high_volume_window_does_not_interrupt_stream(
    tmp_path, edge_config_factory, dataset, total, mode, caplog,
):
    config = load_config(Path(__file__).resolve().parents[1] / f"configs/{dataset}.json")
    config.adaptation.mode = mode
    directory = tmp_path / "state"
    initial = load_detector_state(directory, config)
    edge = edge_config_factory(config, directory)
    records = [record(edge, counts=(total, 0, 0, 0)), record(edge, 10)]
    output = io.StringIO()

    runtime.run_windows(edge, io.BytesIO(b"".join(map(line, records))), output)

    events = [json.loads(value) for value in output.getvalue().splitlines()]
    assert len(events) == 2
    assert events[0]["counts"]["tcp"] == total
    assert events[1]["counts"] == dict(
        zip(config.ingest.categories, records[1]["window"]["counts"])
    )
    if dataset == "unsw_samsung_camera":
        assert events[0]["is_anomaly"] is True
        assert events[0]["composition_score"] is None
        assert "high-precision DM scoring" not in caplog.text
    else:
        assert events[0]["composition_score"] is not None
        assert "high-precision DM scoring" in caplog.text
    assert load_detector_state(directory, edge).version_id == initial.version_id


@pytest.mark.parametrize(("section", "field", "value"), [
    (None, "device_id", "another-device"),
    (None, "source_id", "   "),
    (None, "unexpected", True),
    ("window", "end_utc", "2026-10-15T12:15:00Z"),
    ("window", "start_utc", "2026-10-15T12:10:00"),
    ("window", "categories", ["udp", "tcp", "ssdp", "arp"]),
    ("window", "counts", [True, 0, 0, 0]),
    ("window", "counts", ["1", 0, 0, 0]),
    ("window", "counts", [1.5, 0, 0, 0]),
    ("window", "counts", [float("nan"), 0, 0, 0]),
    ("window", "counts", [float("inf"), 0, 0, 0]),
    ("window", "counts", [MAX_WINDOW_COUNT, 1, 0, 0]),
])
def test_bad_record_reports_line_and_stops_before_reading_more(edge_runtime, section, field, value):
    config, _ = edge_runtime
    invalid = record(config, 10)
    (invalid if section is None else invalid[section])[field] = value
    payload = line(record(config)) + line(invalid) + line(record(config, 20))
    source, output = io.BytesIO(payload), io.StringIO()
    with pytest.raises(DataValidationError, match="input line 2:"):
        runtime.run_windows(config, source, output)
    assert len(output.getvalue().splitlines()) == 1
    assert source.tell() < len(payload)


@pytest.mark.parametrize("payload", [b"{\n", b"{}\n", b"null\n", b"\n", b"\xff\n",
                                    b'{"window":', b"[" * 2000 + b"]" * 2000])
def test_malformed_incomplete_or_non_utf8_input_has_a_clear_error(edge_runtime, payload):
    config, _ = edge_runtime
    output = io.StringIO()
    with pytest.raises(DataValidationError, match="input line 1:"):
        runtime.run_windows(config, io.BytesIO(payload), output)
    assert output.getvalue() == ""


@pytest.mark.parametrize("offset", [0, -10, 5])
def test_duplicate_backward_and_overlapping_windows_are_rejected(edge_runtime, offset):
    config, _ = edge_runtime
    with pytest.raises(DataValidationError, match="input line 2: window overlaps or precedes"):
        runtime.run_windows(config, io.BytesIO(line(record(config)) + line(record(config, offset))), io.StringIO())


def test_missing_coverage_also_advances_the_stream_boundary(edge_runtime):
    config, _ = edge_runtime
    with pytest.raises(DataValidationError, match="input line 2: window overlaps or precedes"):
        runtime.run_windows(config, io.BytesIO(line(record(config, 10, None)) + line(record(config))), io.StringIO())


def test_window_grid_must_match_the_trained_counting_policy(edge_runtime):
    config, _ = edge_runtime
    with pytest.raises(DataValidationError, match="input line 1: window start must align"):
        runtime.run_windows(config, io.BytesIO(line(record(config, 1))), io.StringIO())


def test_oversized_line_is_bounded_before_decoding_or_parsing(edge_runtime):
    config, _ = edge_runtime

    class BoundedReader(io.BytesIO):
        def readline(self, size=-1):
            assert size == runtime.MAX_INPUT_LINE_BYTES + 1
            return super().readline(size)

    source = BoundedReader(b"x" * (runtime.MAX_INPUT_LINE_BYTES * 2))
    with pytest.raises(DataValidationError, match="input line 1: record exceeds"):
        runtime.run_windows(config, source, io.StringIO())
    assert source.tell() == runtime.MAX_INPUT_LINE_BYTES + 1


def test_line_limit_is_inclusive_and_final_newline_is_optional(edge_runtime):
    config, _ = edge_runtime
    value = json.dumps(record(config)).encode()
    payload = value + b" " * (runtime.MAX_INPUT_LINE_BYTES - len(value))
    output = io.StringIO()
    runtime.run_windows(config, io.BytesIO(payload), output)
    assert json.loads(output.getvalue())["artifact_type"] == "anomaly_event"


def test_empty_input_exits_without_output(edge_runtime):
    config, _ = edge_runtime
    output = io.StringIO()
    runtime.run_windows(config, io.BytesIO(), output)
    assert output.getvalue() == ""


def test_output_is_flushed_before_reading_the_next_record(edge_runtime):
    config, _ = edge_runtime
    flushed = []

    class Reader(io.BytesIO):
        def readline(self, size=-1):
            if self.tell():
                assert flushed
            return super().readline(size)

    class Writer(io.StringIO):
        def flush(self):
            flushed.append(self.getvalue())

    output = Writer()
    runtime.run_windows(config, Reader(line(record(config)) + line(record(config, 10))), output)
    assert len(flushed) == 2


@pytest.mark.parametrize("policy", ["conservative", "legacy"])
def test_stream_uses_adaptive_updates_and_logs_attempts(edge_runtime, tmp_path, caplog, policy):
    config, _ = edge_runtime
    raw = load_detector_state(config.state_dir, config).model_dump(exclude={"version_id", "created_utc"})
    raw["adaptation"].update(mode="adaptive_threshold", policy=policy, update_every=2, buffer_size=3, safe_margin=0.)
    raw["calibration"].update(minimum_samples=2, quantile=.25)
    raw["baseline_threshold"].update(quantile=.25, threshold=-100., score_iqr=100.)
    raw["threshold"].update(threshold=-100., score_iqr=100.)
    if policy == "legacy":
        raw["packet_count_limit"] = None
    state = DetectorStateArtifact.model_validate(raw)
    config.adaptation, config.calibration = state.adaptation, state.calibration
    config.state_dir = tmp_path / "adaptive"
    save_detector_state(config.state_dir, state)
    output = io.StringIO()
    records = [record(config), record(config, 10, (101, 19, 3, 2)), record(config, 20)]
    with caplog.at_level(logging.INFO, logger="lm_idnet.runtime"):
        runtime.run_windows(config, io.BytesIO(b"".join(map(line, records))), output)
    events = [json.loads(value) for value in output.getvalue().splitlines()]
    restored = load_detector_state(config.state_dir, config)
    assert events[0]["threshold"] == events[1]["threshold"] == -100.
    assert events[2]["threshold"] == restored.threshold.threshold != -100.
    assert '"promoted": true' in caplog.text
    assert all(event["artifact_type"] == "anomaly_event" for event in events)


def test_uncertain_commit_stops_with_its_original_failure_type(edge_runtime, monkeypatch):
    config, _ = edge_runtime

    def fail(*args, **kwargs):
        raise DetectorStateCommitError("pointer switched but sync failed")

    monkeypatch.setattr(WindowDetector, "process", fail)
    with pytest.raises(DetectorStateCommitError, match="input line 1:") as error:
        runtime.run_windows(config, io.BytesIO(line(record(config))), io.StringIO())
    assert error.value.committed


def test_rejected_refit_is_logged_and_does_not_replace_the_pair(edge_runtime, tmp_path, caplog):
    config, _ = edge_runtime
    raw = load_detector_state(config.state_dir, config).model_dump(exclude={"version_id", "created_utc"})
    raw["adaptation"]["update_every"] = 1
    state = DetectorStateArtifact.model_validate(raw)
    config.adaptation = state.adaptation
    config.state_dir = tmp_path / "rejected-refit"
    save_detector_state(config.state_dir, state)
    output = io.StringIO()
    with caplog.at_level(logging.INFO, logger="lm_idnet.runtime"):
        runtime.run_windows(config, io.BytesIO(line(record(config))), output)
    assert load_detector_state(config.state_dir, config) == state
    assert len(output.getvalue().splitlines()) == 1
    assert '"promoted": false' in caplog.text
    assert "not enough accepted windows" in caplog.text


@pytest.mark.parametrize("operation", ["read", "write"])
def test_stream_io_errors_stop_with_line_context(edge_runtime, operation):
    config, _ = edge_runtime

    class FailedReader(io.BytesIO):
        def readline(self, size=-1):
            raise OSError("input disconnected")

    class FailedWriter(io.StringIO):
        def write(self, value):
            raise OSError("output disconnected")

    source = FailedReader() if operation == "read" else io.BytesIO(line(record(config)))
    output = FailedWriter() if operation == "write" else io.StringIO()
    with pytest.raises(DataValidationError, match="input line 1: .* disconnected"):
        runtime.run_windows(config, source, output)


@pytest.mark.integration
def test_cli_delivers_event_before_eof_and_holds_state_lock(edge_runtime):
    config, path = edge_runtime
    process = subprocess.Popen(arguments(path), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=environment())
    try:
        process.stdin.write(line(record(config)))
        process.stdin.flush()
        assert select.select([process.stdout], [], [], 10)[0], "first event was not flushed"
        event_line = os.read(process.stdout.fileno(), 65536)
        assert event_line.endswith(b"\n")
        assert json.loads(event_line)["capture_id"] == "local-collector"
        assert process.poll() is None
        with pytest.raises(PolicyRejectionError, match="in use"):
            with detector_state_lock(config.state_dir):
                pytest.fail("streaming writer did not hold the lock")
        process.stdin.close()
        process.stdin = None
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0
        assert stdout == stderr == b""
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)


@pytest.mark.integration
@pytest.mark.parametrize("error_format", ["text", "json"])
def test_cli_validation_errors_use_stderr_and_nonzero_status(edge_runtime, error_format):
    config, path = edge_runtime
    result = subprocess.run(
        [sys.executable, "-m", "lm_idnet.cli", "--error-format", error_format,
         "run", "--config", str(path)],
        input=line(record(config)) + b"{}\n", capture_output=True, env=environment(), check=False,
    )
    assert result.returncode == DataValidationError.exit_code
    assert len(result.stdout.splitlines()) == 1
    assert json.loads(result.stdout)["artifact_type"] == "anomaly_event"
    assert b"input line 2:" in result.stderr
    assert b"Traceback" not in result.stderr
    if error_format == "json":
        assert json.loads(result.stderr)["error"]["code"] == "data_validation_error"


@pytest.mark.integration
def test_cli_state_override_and_dry_run(edge_runtime, tmp_path):
    config, path = edge_runtime
    directory = config.state_dir
    config.state_dir = tmp_path / "unprepared"
    path.write_text(config.model_dump_json())
    dry_run = subprocess.run(arguments(path, "--dry-run"), capture_output=True, env=environment(), check=False)
    assert dry_run.returncode == 0
    assert json.loads(dry_run.stdout)["status"] == "ready"
    assert not config.state_dir.exists()
    result = subprocess.run(arguments(path, "--state-dir", str(directory)), input=line(record(config)),
                            capture_output=True, env=environment(), check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["device_id"] == config.ingest.device_id


@pytest.mark.integration
def test_runtime_log_retention_is_bounded(tmp_path):
    log_path = tmp_path / "runtime.log"
    result = subprocess.run([
        sys.executable, "-c",
        "import logging; from lm_idnet import cli; "
        "cli.RUNTIME_LOG_MAX_BYTES = 512; cli._configure_logging(False, bounded=True); "
        "[logging.getLogger('lm_idnet.runtime').info('test ' + 'x' * 100) for _ in range(50)]",
    ], capture_output=True, env=dict(os.environ, LM_IDNET_LOG_PATH=str(log_path)), check=False)
    assert result.returncode == 0, result.stderr
    logs = list(tmp_path.glob("runtime.log*"))
    assert len(logs) == cli.RUNTIME_LOG_BACKUPS + 1
    assert all(path.stat().st_size <= 512 for path in logs)


@pytest.mark.integration
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_signal_stops_an_idle_reader_cleanly_and_releases_lock(edge_runtime, signum):
    config, path = edge_runtime
    process = subprocess.Popen(arguments(path), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=environment())
    try:
        process.stdin.write(line(record(config)))
        process.stdin.flush()
        assert select.select([process.stdout], [], [], 10)[0]
        assert json.loads(os.read(process.stdout.fileno(), 65536))["artifact_type"] == "anomaly_event"
        process.send_signal(signum)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        assert stdout == stderr == b""
        with detector_state_lock(config.state_dir):
            pass
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)


def test_signal_during_atomic_activation_is_deferred(edge_runtime, tmp_path, monkeypatch):
    from lm_idnet import artifacts

    config, _ = edge_runtime
    raw = load_detector_state(config.state_dir, config).model_dump(exclude={"version_id", "created_utc"})
    raw["adaptation"].update(mode="adaptive_threshold", update_every=2, buffer_size=3, safe_margin=0.)
    raw["calibration"].update(minimum_samples=2, quantile=.25)
    raw["baseline_threshold"].update(quantile=.25, threshold=-100., score_iqr=100.)
    raw["threshold"].update(threshold=-100., score_iqr=100.)
    state = DetectorStateArtifact.model_validate(raw)
    config.adaptation, config.calibration = state.adaptation, state.calibration
    config.state_dir = tmp_path / "signal-state"
    save_detector_state(config.state_dir, state)
    original_replace = artifacts.os.replace
    handler = signal.getsignal(signal.SIGTERM)

    def stop_after_pointer_switch(source, destination):
        original_replace(source, destination)
        if Path(destination).name == "current.json":
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(artifacts.os, "replace", stop_after_pointer_switch)
    output = io.StringIO()
    records = [record(config), record(config, 10, (101, 19, 3, 2)), record(config, 20)]
    runtime.run_windows(config, io.BytesIO(b"".join(map(line, records))), output)
    assert load_detector_state(config.state_dir, config).threshold.threshold != -100.
    assert len(output.getvalue().splitlines()) == 1
    assert signal.getsignal(signal.SIGTERM) == handler


@pytest.mark.integration
def test_edge_rollback_cli_restores_previous_pair(edge_runtime):
    config, path = edge_runtime
    first = load_detector_state(config.state_dir, config)
    raw = first.model_dump(exclude={"version_id", "created_utc"})
    raw["source"] = "adaptive_threshold"
    raw["threshold"]["threshold"] -= .25
    save_detector_state(config.state_dir, DetectorStateArtifact.model_validate(raw))
    result = subprocess.run(
        [sys.executable, "-m", "lm_idnet.cli", "rollback", "--config", str(path),
         "--state-dir", str(config.state_dir)], capture_output=True, env=environment(), check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["active_version"] == first.version_id
    assert load_detector_state(config.state_dir, config) == first


@pytest.mark.parametrize("score_type", ["raw", "normalized"])
@pytest.mark.parametrize("policy", ["conservative", "legacy"])
@pytest.mark.parametrize("mode", ["static", "adaptive_threshold", "periodic_refit"])
@pytest.mark.parametrize("count_tails", [False, True])
def test_streaming_cli_replay_matches_batch_updates(
    tmp_path, config_factory, edge_config_factory, monkeypatch, capsys, score_type, policy, mode,
    count_tails,
):
    import numpy as np
    from lm_idnet.algorithms.dirichlet import DirichletFit
    from lm_idnet.models import scoring_stage
    from lm_idnet.processing.storage import load_processed_dataset
    from test_scoring_stage import write_model, write_threshold, write_processed_captures, read_adaptation_outputs

    config = config_factory(
        ingest={"processed_root": tmp_path / "processed"},
        estimator={"log_likelihood_backend": "scipy"},
        calibration={"score_type": score_type, "minimum_samples": 2, "quantile": .25},
        adaptation={"mode": mode, "policy": policy, "buffer_size": 3,
                    "update_every": 2, "minimum_refit_windows": 2},
        outputs={"model_path": tmp_path / "model.json", "threshold_path": tmp_path / "threshold.json",
                 "events_path": tmp_path / "events.json", "reports_dir": tmp_path / "reports"},
    )
    write_model(config)
    write_threshold(config, score_type, threshold_value=-8. if score_type == "raw" else -.8,
                    score_iqr=2. if score_type == "raw" else .2)
    if count_tails:
        raw_threshold = json.loads(config.outputs.threshold_path.read_text())
        raw_threshold["count_tails"] = {
            name: {"threshold": float(np.log1p(100)), "score_iqr": 1.}
            for name in ("total", "arp")
        }
        config.outputs.threshold_path.write_text(json.dumps(raw_threshold))
    write_processed_captures(config, "calibration", counts=((1, 2, 3, 4), (4, 2, 3, 4)))
    write_processed_captures(config, counts=(
        (1, 2, 3, 4), (2, 2, 3, 4), (10000, 0, 0, 0),
        (0, 0, 0, 0), None, (3, 2, 3, 4), (4, 2, 3, 4),
    ))

    class Estimator:
        calls = 0

        def fit(self, _matrix):
            self.calls += 1
            alpha = np.array([1., 2., 3., 4.]) * 1.01
            return DirichletFit(alpha, alpha, float(alpha.sum()), 1 / float(alpha.sum()),
                                1, self.calls % 2 == 1, -1e9, 1e9)

    monkeypatch.setattr(scoring_stage, "create_estimator", lambda *_: Estimator())
    scoring_stage.adapt_windows(config)
    expected, report = read_adaptation_outputs(config)
    directory = tmp_path / "stream-state"
    load_detector_state(directory, config)
    edge = edge_config_factory(config, directory)
    path = tmp_path / "edge.json"
    path.write_text(edge.model_dump_json())
    records = []
    for capture_id in config.ingest.partitions.development_test:
        dataset = load_processed_dataset(
            config.ingest.processed_root / config.ingest.dataset_folder / f"{capture_id}.json",
        )
        records.extend({"device_id": config.ingest.device_id, "source_id": capture_id,
                        "window": window.model_dump(mode="json")} for window in dataset.windows)
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(b"".join(map(line, records)))))
    monkeypatch.setenv("LM_IDNET_LOG_PATH", str(tmp_path / "stream.log"))
    assert cli.main(["run", "--config", str(path)]) == 0
    output = capsys.readouterr()
    assert [json.loads(value) for value in output.out.splitlines()] == expected
    assert output.err == ""
    state = load_detector_state(directory, edge)
    assert list(state.alpha) == report["final_alpha"]
    assert state.threshold.threshold == report["final_threshold"]
    assert state.threshold.model_fingerprint == report["final_model_fingerprint"]
    if mode != "static":
        assert report["threshold_update_count"] > 0
        history = json.loads((directory / "history.json").read_text())["records"]
        assert any(item["promoted"] for item in history)
        if mode == "periodic_refit":
            assert any(not item["promoted"] for item in history)


@pytest.mark.integration
def test_slow_output_stalls_reads_without_memory_growth_and_can_stop(edge_runtime, tmp_path):
    config, path = edge_runtime
    raw = load_detector_state(config.state_dir, config).model_dump(exclude={"version_id", "created_utc"})
    raw["adaptation"]["mode"] = "static"
    raw["packet_count_limit"] = None
    state = DetectorStateArtifact.model_validate(raw)
    config.adaptation = state.adaptation
    config.state_dir = tmp_path / "static-state"
    save_detector_state(config.state_dir, state)
    path.write_text(config.model_dump_json())
    source_path = tmp_path / "many-windows.jsonl"
    with source_path.open("wb") as source:
        for index in range(10000):
            source.write(line(record(config, index * 10)))
    with source_path.open("rb") as source:
        process = subprocess.Popen(arguments(path), stdin=source, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=dict(environment(), PYTHONUNBUFFERED="1"))
        try:
            assert select.select([process.stdout], [], [], 10)[0]
            sleep(.2)

            def rss():
                return int(next(value.split()[1] for value in
                    Path(f"/proc/{process.pid}/status").read_text().splitlines() if value.startswith("VmRSS:"))) * 1024

            first = rss()
            sleep(.5)
            assert process.poll() is None
            assert rss() <= first + 2 * 1024 * 1024
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=5) == 0
            process.communicate(timeout=5)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)

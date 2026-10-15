"""Streaming input, event delivery, and CLI failure checks with prepared state."""

from datetime import datetime, timedelta, timezone
import io
import json
import logging
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys

import pytest

from lm_idnet import cli, runtime
from lm_idnet.artifacts import detector_state_lock, load_detector_state, save_detector_state
from lm_idnet.config import load_edge_config
from lm_idnet.exceptions import DataValidationError, DetectorStateCommitError, PolicyRejectionError
from lm_idnet.models.schemas import DetectorStateArtifact
from lm_idnet.models.scoring_stage import WindowDetector
from lm_idnet.processing.schemas import MAX_WINDOW_COUNT, WindowRecord

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def edge_runtime(tmp_path):
    config = load_edge_config(ROOT / "configs" / "edge_example.json")
    config.state_dir = tmp_path / "state"
    shutil.copytree(ROOT / "artifacts" / "edge-initial-state", config.state_dir)
    config.outputs.model_path = tmp_path / "absent-model.json"
    config.outputs.threshold_path = tmp_path / "absent-threshold.json"
    path = tmp_path / "edge.json"
    path.write_text(config.model_dump_json())
    return config, path


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

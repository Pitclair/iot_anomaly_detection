"""Opt-in checks against the prepared production image; no builds or downloads."""

import json
import os
from pathlib import Path
import subprocess
from time import monotonic, sleep
from uuid import uuid4

import pytest

from lm_idnet.artifacts import load_detector_state, save_detector_state
from lm_idnet.models.schemas import DetectorStateArtifact
from test_runtime import line, record

ROOT = Path(__file__).resolve().parents[1]
IMAGE = os.environ.get("LM_IDNET_EDGE_IMAGE")
pytestmark = [pytest.mark.integration, pytest.mark.skipif(not IMAGE, reason="set LM_IDNET_EDGE_IMAGE to a built image")]


def command(arguments, *, environment=None, check=True, **kwargs):
    return subprocess.run(arguments, env=environment, capture_output=True, check=check, timeout=40, **kwargs)


def wait_for(operation, timeout=20):
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        result = operation()
        if result:
            return result
        sleep(.1)
    pytest.fail("container did not reach the expected state")


def test_prebuilt_image_runs_offline_without_compiler_or_test_extra(edge_runtime, tmp_path):
    config, _ = edge_runtime
    environment = dict(os.environ, LM_IDNET_IMAGE_NAME=IMAGE, LM_IDNET_STATE_DIR=str(config.state_dir),
                       LM_IDNET_LOG_DIR=str(tmp_path / "logs"))
    fixture = (ROOT / "tests/data/edge/windows.jsonl").read_bytes()
    first = command([str(ROOT / "scripts/run-edge.sh")], environment=environment, input=fixture)
    second = command([str(ROOT / "scripts/run-edge.sh")], environment=environment, input=fixture)
    events = [json.loads(item) for item in first.stdout.splitlines()]
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == b""
    assert len(events) == 4
    assert any(event["is_anomaly"] for event in events)
    assert any(not sum(event["counts"].values()) for event in events)
    command(["docker", "run", "--rm", "--pull", "never", "--network", "none", "--read-only",
             "--entrypoint", "python", IMAGE, "-c",
             "import importlib.util,shutil; from lm_idnet.algorithms.lm_native import NativeLmKernel; "
             "NativeLmKernel(6); assert shutil.which('gcc') is None; "
             "assert importlib.util.find_spec('pytest') is None"])


def test_service_restarts_preserves_state_rotates_output_and_stays_stopped(edge_runtime, tmp_path):
    config, _ = edge_runtime
    initial = load_detector_state(config.state_dir, config)
    raw = initial.model_dump(exclude={"version_id", "created_utc"})
    raw["source"] = "adaptive_threshold"
    raw["threshold"]["threshold"] -= .25
    accepted = DetectorStateArtifact.model_validate(raw)
    save_detector_state(config.state_dir, accepted)
    input_path = tmp_path / "windows.fifo"
    os.mkfifo(input_path)
    fifo = os.open(input_path, os.O_RDWR)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    override = tmp_path / "rotation.json"
    override.write_text(json.dumps({"services": {"detector": {"logging": {
        "driver": "local", "options": {"max-size": "1k", "max-file": "2"},
    }}}}))
    environment = dict(os.environ, HOST_UID=str(os.getuid()), HOST_GID=str(os.getgid()),
        LM_IDNET_IMAGE_NAME=IMAGE, LM_IDNET_STATE_DIR=str(config.state_dir),
        LM_IDNET_LOG_DIR=str(log_dir), LM_IDNET_INPUT=str(input_path))
    compose = ["docker", "compose", "-p", "lm-idnet-check-" + uuid4().hex[:10],
               "-f", str(ROOT / "compose.edge.yaml"), "-f", str(override)]
    try:
        command([*compose, "up", "--detach", "--no-build", "--pull", "never"], environment=environment)
        container = command([*compose, "ps", "-q"], environment=environment).stdout.decode().strip()

        def inspect():
            return json.loads(command(["docker", "inspect", container]).stdout)[0]

        def events():
            return [json.loads(value) for value in command(["docker", "logs", container]).stdout.splitlines()]

        os.write(fifo, line(record(config)))
        first = wait_for(lambda: events() or None)[-1]
        assert first["threshold"] == accepted.threshold.threshold
        sleep(10)
        os.write(fifo, b"invalid JSON\n")
        wait_for(lambda: inspect()["RestartCount"] >= 1 and inspect()["State"]["Running"])
        os.write(fifo, line(record(config, 10)))
        wait_for(lambda: any(item["window_start_utc"].endswith("12:10:00Z") for item in events()))
        assert events()[-1]["threshold"] == accepted.threshold.threshold
        for index in range(100):
            os.write(fifo, line(record(config, 20 + index * 10)))
        wait_for(lambda: any(item["window_start_utc"].endswith("04:50:00Z") for item in events()))
        assert len(events()) < 102
        assert len(command(["docker", "logs", container]).stdout) < 20000
        running = inspect()
        assert running["HostConfig"]["Memory"] == 512 * 1024 * 1024
        assert running["HostConfig"]["NanoCpus"] == 1000000000
        assert running["HostConfig"]["ReadonlyRootfs"]
        assert running["HostConfig"]["NetworkMode"] == "none"
        assert running["HostConfig"]["LogConfig"]["Config"]["max-file"] == "2"
        command([*compose, "stop", "--timeout", "10"], environment=environment)
        sleep(1)
        stopped = inspect()
        assert not stopped["State"]["Running"]
        assert stopped["State"]["ExitCode"] == 0
        assert stopped["RestartCount"] == running["RestartCount"]
        assert load_detector_state(config.state_dir, config) == accepted
    finally:
        os.close(fifo)
        command([*compose, "down", "--timeout", "5"], environment=environment, check=False)

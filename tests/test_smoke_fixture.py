import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lm_idnet.config import load_config
from lm_idnet.randomness import create_named_generators
from lm_idnet.smoke import (
    aggregate_smoke_fixture,
    load_smoke_fixture,
    run_smoke_experiment,
)

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "data" / "smoke" / "synthetic_packet_counts.json"
CONFIG = ROOT / "configs" / "config.json"


def test_smoke_fixture_is_synthetic_payload_free_and_redistributable() -> None:
    fixture_text = FIXTURE.read_text(encoding="utf-8")
    fixture = json.loads(fixture_text)

    assert fixture["metadata"] == {
        "name": "lm-idnet-public-smoke-fixture",
        "synthetic": True,
        "payload_free": True,
        "license": "CC0-1.0",
    }
    assert all(set(event) == {"timestamp", "category"} for event in fixture["events"])
    assert not re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", fixture_text)
    assert not re.search(r"\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b", fixture_text)


def test_smoke_preprocessing_matches_exact_expected_windows() -> None:
    fixture = load_smoke_fixture(FIXTURE)

    actual = aggregate_smoke_fixture(fixture)

    assert actual == fixture["expected_windows"]
    assert actual[1]["state"] == "missing"
    assert actual[1]["counts"] is None


def test_missing_smoke_window_is_excluded_from_model_counts() -> None:
    report = run_smoke_experiment(FIXTURE, CONFIG)

    assert report["windows"][1]["state"] == "missing"
    assert report["windows"][1]["counts"] is None
    assert len(report["windows"]) == 3
    assert report["modeled_window_count"] == 2


def test_named_generators_are_reproducible_and_independent() -> None:
    seeds = load_config(CONFIG).seeds
    first = create_named_generators(seeds)
    second = create_named_generators(seeds)

    assert np.array_equal(
        first.simulation.integers(0, 100, 8),
        second.simulation.integers(0, 100, 8),
    )
    assert np.array_equal(
        first.model_fitting.integers(0, 100, 8),
        second.model_fitting.integers(0, 100, 8),
    )
    assert not np.array_equal(
        create_named_generators(seeds).simulation.integers(0, 100, 8),
        create_named_generators(seeds).model_fitting.integers(0, 100, 8),
    )


def test_smoke_experiment_does_not_mutate_global_numpy_rng() -> None:
    np.random.seed(12345)
    expected = np.random.random(4)
    np.random.seed(12345)

    run_smoke_experiment(FIXTURE, CONFIG)
    actual = np.random.random(4)

    assert np.array_equal(actual, expected)


def run_smoke_process(output_path: Path) -> dict:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lm_idnet.smoke",
            "--fixture",
            str(FIXTURE),
            "--config",
            str(CONFIG),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(output_path.read_text(encoding="utf-8"))


def test_smoke_experiment_reproduces_across_separate_processes(
    tmp_path: Path,
) -> None:
    first = run_smoke_process(tmp_path / "first.json")
    second = run_smoke_process(tmp_path / "second.json")

    assert first == second
    assert first["model_parameters"] == second["model_parameters"]
    assert first["injected_anomaly"] == second["injected_anomaly"]
    assert first["metrics"] == second["metrics"]
    assert first["float_tolerance"] == 1e-12
    assert first["scientific_claim"] == "none"

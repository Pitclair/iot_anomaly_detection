"""Contract tests for shared fixtures and marker selection."""

from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from lm_idnet.config import AppConfig
from lm_idnet.processing.schemas import WindowRecord


@pytest.mark.unit
def test_count_matrix_factory_is_deterministic(
    count_matrix_factory: Callable[..., np.ndarray],
) -> None:
    first = count_matrix_factory(rows=3, categories=4, seed=42)
    second = count_matrix_factory(rows=3, categories=4, seed=42)

    assert first.shape == (3, 4)
    assert first.dtype == np.int64
    assert np.array_equal(first, second)
    assert np.all(first >= 0)


@pytest.mark.unit
def test_window_factory_returns_valid_schema(
    window_factory: Callable[..., WindowRecord],
) -> None:
    window = window_factory(tcp=4, udp=3, ssdp=2, arp=1)

    assert isinstance(window, WindowRecord)
    assert window.counts == (4, 3, 2, 1)
    assert window.total_count == sum(window.counts) == 10


@pytest.mark.unit
def test_config_factory_returns_independent_valid_models(
    config_factory: Callable[..., AppConfig],
) -> None:
    changed = config_factory(calibration={"quantile": 0.05})
    baseline = config_factory()

    assert changed.calibration.quantile == 0.05
    assert baseline.calibration.quantile == 0.01


@pytest.mark.integration
def test_artifact_store_is_isolated_and_complete(artifact_store: Path) -> None:
    assert artifact_store.name == "artifacts"
    assert {path.name for path in artifact_store.iterdir()} == {
        "models",
        "thresholds",
        "events",
        "metrics",
        "logs",
    }


@pytest.mark.numerical
def test_numerical_marker_is_collectable() -> None:
    assert np.isfinite(np.log1p(1.0))


@pytest.mark.slow
def test_slow_marker_is_collectable() -> None:
    assert True


@pytest.mark.pcap
def test_pcap_marker_is_collectable() -> None:
    assert True


@pytest.mark.benchmark
def test_benchmark_marker_is_collectable() -> None:
    assert True

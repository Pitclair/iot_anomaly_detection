"""Shared pytest fixtures for the LM-IDNet test architecture."""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from lm_idnet.config import AppConfig, DetectorInputConfig, EdgeConfig, load_edge_config
from lm_idnet.processing.schemas import WindowRecord


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def count_matrix_factory() -> Callable[..., np.ndarray]:
    """Create deterministic, non-negative integer count matrices."""

    def create(
        rows: int = 4,
        categories: int = 4,
        *,
        seed: int = 1001,
        maximum: int = 20,
    ) -> np.ndarray:
        if rows <= 0 or categories <= 0:
            raise ValueError("rows and categories must be positive")
        if maximum < 0:
            raise ValueError("maximum must be non-negative")
        rng = np.random.default_rng(seed)
        return rng.integers(
            0,
            maximum + 1,
            size=(rows, categories),
            dtype=np.int64,
        )

    return create


@pytest.fixture
def window_factory() -> Callable[..., WindowRecord]:
    """Create valid processed-window schema instances."""
    config_data = json.loads(
        (ROOT / "configs" / "d_link_day_cam5.json").read_text(encoding="utf-8")
    )
    category_order = AppConfig.model_validate(config_data).ingest.categories

    def create(
        *,
        tcp: int = 10,
        udp: int = 5,
        ssdp: int = 1,
        arp: int = 0,
    ) -> WindowRecord:
        counts = (tcp, udp, ssdp, arp)
        total_count = sum(counts)
        return WindowRecord(
            start_utc="2020-01-01T00:00:00Z",
            end_utc="2020-01-01T00:10:00Z",
            categories=category_order,
            counts=counts,
            state="observed" if total_count else "observed-silent",
        )

    return create


@pytest.fixture
def config_factory() -> Callable[..., AppConfig]:
    """Create independently mutable, validated application configurations."""
    baseline = json.loads(
        (ROOT / "configs" / "d_link_day_cam5.json").read_text(encoding="utf-8")
    )

    def create(**section_overrides: dict[str, Any]) -> AppConfig:
        data = deepcopy(baseline)
        for section, values in section_overrides.items():
            if section not in data:
                raise KeyError(f"unknown configuration section: {section}")
            if not isinstance(values, dict) or not isinstance(data[section], dict):
                data[section] = values
            else:
                data[section].update(values)
        return AppConfig.model_validate(data)

    return create


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


@pytest.fixture
def edge_config_factory():
    def create(config, state_dir):
        return EdgeConfig.model_validate({
            "ingest": config.ingest.model_dump(include=set(DetectorInputConfig.model_fields)),
            "outputs": config.outputs.model_dump(include={"model_path", "threshold_path"}),
            "state_dir": state_dir,
            **config.model_dump(include={"precision_digits", "estimator", "calibration", "adaptation"}),
        })

    return create


@pytest.fixture
def artifact_store(tmp_path: Path) -> Path:
    """Create an isolated dataset-specific artifact directory."""
    store = tmp_path / "artifacts" / "D-LinkDayCam5"
    store.mkdir(parents=True)
    return store

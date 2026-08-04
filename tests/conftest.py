"""Shared pytest fixtures for the LM-IDNet test architecture."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from lm_idnet.config import AppConfig
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
        (ROOT / "configs" / "config.json").read_text(encoding="utf-8")
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
            device_id="test-device",
            capture_id="test-capture",
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
        (ROOT / "configs" / "config.json").read_text(encoding="utf-8")
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
def artifact_store(tmp_path: Path) -> Path:
    """Create an isolated artifact directory with the standard run layout."""
    store = tmp_path / "artifacts"
    for child in ("models", "thresholds", "events", "metrics", "logs"):
        (store / child).mkdir(parents=True)
    return store

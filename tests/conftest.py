"""Shared pytest fixtures for the LM-IDNet test architecture."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from lm_idnet.config import AppConfig
from lm_idnet.processing.schemas import WindowCount


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
def window_factory() -> Callable[..., WindowCount]:
    """Create valid processed-window schema instances."""

    def create(
        *,
        tcp: int = 10,
        udp: int = 5,
        ssdp: int = 1,
        arp: int = 0,
    ) -> WindowCount:
        return WindowCount(tcp=tcp, udp=udp, ssdp=ssdp, arp=arp)

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

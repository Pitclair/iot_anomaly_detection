"""Named, explicit random-number generators for reproducible stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from lm_idnet.config import SeedConfig


@dataclass(frozen=True)
class NamedGenerators:
    simulation: Generator
    model_fitting: Generator


def create_named_generators(seeds: SeedConfig) -> NamedGenerators:
    """Create independent generators without reading or changing global RNG state."""
    return NamedGenerators(
        simulation=np.random.default_rng(seeds.simulation),
        model_fitting=np.random.default_rng(seeds.model_fitting),
    )

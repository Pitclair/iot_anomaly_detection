# Testing architecture

LM-IDNet uses pytest markers to keep fast correctness checks separate from
data-heavy and performance work. Marker declarations and the default selection
live in `pyproject.toml`; reusable fixtures live in `tests/conftest.py`.

## Markers

- `unit`: isolated deterministic behavior.
- `integration`: behavior crossing modules, processes, or artifact boundaries.
- `numerical`: numerical correctness, stability, or convergence.
- `slow`: correctness tests unsuitable for the default fast feedback loop.
- `pcap`: tests that require packet-capture fixtures or parsing.
- `benchmark`: performance measurement rather than correctness.

A test may have more than one marker. For example, estimator correctness can be
both `unit` and `numerical`, while a real-PCAP pipeline test can be both
`integration` and `pcap`.

Unknown or misspelled markers are errors because pytest runs with
`--strict-markers`.

## Commands

The default suite excludes only the explicitly slow or data-heavy groups:

```sh
python -m pytest
```

Run one group independently:

```sh
python -m pytest -m unit
python -m pytest -m integration
python -m pytest -m numerical
python -m pytest -m slow
python -m pytest -m pcap
python -m pytest -m benchmark
```

Run absolutely every test:

```sh
python -m pytest -m ""
```

## Shared fixture factories

- `count_matrix_factory`: deterministic non-negative integer matrices with
  configurable shape, seed, and maximum count.
- `window_factory`: valid `WindowCount` instances.
- `config_factory`: independent validated configurations with section-level
  overrides.
- `artifact_store`: an isolated temporary artifact tree automatically removed
  by pytest.

Factories prevent individual tests from sharing mutable objects. New test
modules should use these fixtures rather than maintaining slightly different
local versions of the same setup.

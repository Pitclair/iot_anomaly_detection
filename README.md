# IoT Anomaly Detection Baseline

This repo provides a scaffold for modeling and forecasting IoT network traffic using Dirichlet-based methods.

Structure:
- `src/lm_idnet/`: installable Python package
- configs/config.json: default configuration
- `tests/`: automated tests

Install for local development:

```sh
python -m pip install -e .
```

Run the CLI from any directory without setting `PYTHONPATH`:

```sh
lm-idnet --help
lm-idnet preprocess --config configs/config.json
lm-idnet diagnose --config configs/config.json
lm-idnet train --config configs/config.json
lm-idnet forecast --config configs/config.json
```

The stable command set is `preprocess`, `diagnose`, `train`, `calibrate`,
`score`, `evaluate`, `adapt`, `forecast`, and `benchmark`. Use
`lm-idnet <command> --help` for command-specific options. Commands whose
pipeline stage has not been implemented yet fail explicitly with exit code 9;
they never report a false success. Every command supports `--dry-run` to
validate its configuration and CLI wiring without executing the stage.

The complete command contract and current implementation status are documented
in [`docs/command-line-interface.md`](docs/command-line-interface.md).

`python main.py ...` remains available as a compatibility launcher after the
package has been installed.

Configuration fields and validation rules are documented in
[`docs/configuration.md`](docs/configuration.md).

Test markers, suite commands, and shared fixtures are documented in
[`docs/testing.md`](docs/testing.md).

Domain exceptions, CLI exit codes, and fail-closed artifact handling are
documented in [`docs/failure-semantics.md`](docs/failure-semantics.md).

Persisted artifact types, semantic versions, and migration rules are documented
in [`docs/artifact-schemas.md`](docs/artifact-schemas.md).

Source-capture inventory behavior and duplicate explanations are documented in
[`docs/capture-inventory.md`](docs/capture-inventory.md).

Complete PCAP parsing, warning classification, and corruption handling are
documented in [`docs/pcap-validation.md`](docs/pcap-validation.md).

The frozen chronological research split and purpose-scoped access rules are
documented in [`docs/temporal-partitions.md`](docs/temporal-partitions.md).

Dataset version identifiers are documented in
[`docs/dataset-fingerprints.md`](docs/dataset-fingerprints.md), named random
generators in
[`docs/deterministic-randomness.md`](docs/deterministic-randomness.md), and the
public fixture in [`docs/smoke-fixture.md`](docs/smoke-fixture.md).

# IoT Anomaly Detection Baseline

This repository provides a scaffold for modeling and forecasting IoT network
traffic using Dirichlet-based methods.

Structure:

- `src/lm_idnet/`: installable Python package
- `configs/d_link_day_cam5.json`: default configuration
- `tests/`: automated tests

## Run with Docker

Docker is the recommended way to run the project. The image installs
`lm-idnet`, and every container invocation compiles the required native LM
library before dispatching the requested CLI command.

Use the runner from the repository root:

```sh
./scripts/run.sh --help
./scripts/run.sh preprocess --config configs/d_link_day_cam5.json
./scripts/run.sh diagnose --config configs/d_link_day_cam5.json
./scripts/run.sh train --config configs/d_link_day_cam5.json
./scripts/run.sh forecast --config configs/d_link_day_cam5.json
```

The runner builds `lm-idnet:latest` using Docker's build cache and then starts
a temporary container. It mounts these host directories into `/app` so inputs
and generated artifacts survive after the container exits:

- `configs/` as read-only configuration
- `data/` for raw and processed datasets
- `artifacts/` for dataset-specific models, thresholds, and detector output
- `reports/` for generated reports
- `logs/` for application logs

The container runs commands with the current host user's numeric user and
group IDs, so generated files remain writable outside Docker.

The first argument is always an ordinary `lm-idnet` argument or subcommand;
the runner does not select a pipeline stage for you. For example, inspect a
specific command with:

```sh
./scripts/run.sh train --help
```

Override the image name or Python base version when needed:

```sh
LM_IDNET_IMAGE_NAME=my-lm-idnet PYTHON_VERSION=3.12-slim \
  ./scripts/run.sh diagnose --config configs/d_link_day_cam5.json
```

The equivalent manual build is:

```sh
docker build --tag lm-idnet:latest .
```

The runner is preferred for execution because it supplies all persistent
volume mounts consistently. Docker Compose is not required because this
project currently runs as one command-line application rather than a group of
long-running services.

## Local development

Install the package directly only when the local machine already has all
required build tools:

```sh
python -m pip install -e .
```

Run the CLI without setting `PYTHONPATH`:

```sh
lm-idnet --help
lm-idnet preprocess --config configs/d_link_day_cam5.json
lm-idnet diagnose --config configs/d_link_day_cam5.json
lm-idnet train --config configs/d_link_day_cam5.json
lm-idnet forecast --config configs/d_link_day_cam5.json
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

The distinction between the fitting likelihood kernel and the complete anomaly
score is documented in
[`docs/likelihood-semantics.md`](docs/likelihood-semantics.md).

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

# LM-IDNet

LM-IDNet is a lightweight, unsupervised anomaly detector for IoT traffic. It
learns each device's normal mixture of TCP, ordinary UDP, SSDP, and ARP packet
counts in ten-minute windows using a Dirichlet–multinomial model. Separate
benign windows calibrate likelihood, total-volume, and ARP margins. Fitting,
calibration, and runtime decisions require no labelled attacks, attack
signatures, or neural networks; offline evaluation and method selection use
labels where available. A trusted normal reference is still required.

The native Languasco–Migliardi numerical backend computes likelihood terms
accurately. Static detection and guarded adaptation share the same scoring
path. Every alert records its window, counts, expected profile, residuals,
score, cutoff, and model fingerprint so the decision can be traced.

Retrospective Chromecast and Samsung camera development results show 88.24%
and 92.31% static precision, with 46.88% and 68.57% recall. Results vary across
devices and adaptation policies. Resource measurements cover a local
Linux/amd64 Docker simulation with one CPU and 512 MiB, rather than physical
IoT hardware. The [full project report](docs/lm_idnet_full_project_report.tex)
describes the evidence, rejected alternatives, and limits. The detector emits
alerts; it does not block traffic automatically.

A labelled 10/20/30/60-minute study shows a detection/response-time tradeoff.
Thirty minutes gives the strongest static development results, while ten
minutes alerts sooner; longer-window threshold adaptation can add false alerts.
The production default remains ten minutes. The saved
[results](reports/window_size/results.json) cover static and both conservative
adaptive modes with fresh fits and calibration at each duration.

The completed final-partition evaluation covers 1,008 labelled windows: pooled
precision 82.35%, recall 77.78%, and F1 80.00%. Static and both conservative
adaptive modes agree, with no promoted updates. Fitting and calibration use
normal traffic; final labels are used only for offline evaluation. Starting
pairs, full events, update attempts, and input/code
fingerprints are retained in the [final evidence bundle](reports/final_evaluation/results.json).
From the repository root, reproduce it with the installed package:

```sh
python scripts/evaluate_final.py
```

The detector, evaluation, and full project report complete the agreed scope.
The [complete thesis draft](thesis/README.md) is ready for the author's final
review. Optional research and physical deployment are listed
in [future work](docs/future-work.md). Earlier plans and status reports are
historical records, rather than an active implementation backlog.

Structure:

- `src/lm_idnet/`: installable Python package
- `configs/d_link_day_cam5.json`: default configuration
- `tests/`: automated tests

## Run with Docker

Docker is the recommended way to run the project. The image installs
`lm-idnet` and compiles the native LM library during the image build. Startup
uses that compiled library without a compiler or dependency downloads.

For the one-CPU, 512-MiB gateway simulation, use the
[edge deployment guide](docs/edge-deployment.md). It supplies prepared state,
a JSON-lines replay, an offline runner, a bounded service, rollback, and measurements.

Use the runner from the repository root:

```sh
./scripts/run.sh --help
./scripts/run.sh preprocess --config configs/d_link_day_cam5.json
./scripts/run.sh diagnose --config configs/d_link_day_cam5.json
./scripts/run.sh train --config configs/d_link_day_cam5.json
./scripts/run.sh calibrate --config configs/d_link_day_cam5.json
./scripts/run.sh score --config configs/d_link_day_cam5.json
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

Use this runner for research commands and their persistent volume mounts.
The edge runner launches a prebuilt image; the optional Compose service runs
the supplied-window simulation with bounded output and restart behaviour.

## Compile LaTeX documents

The comprehensive project reference is
[lm_idnet_full_project_report.tex](docs/lm_idnet_full_project_report.tex),
with modular sources in [docs/full_project_report/](docs/full_project_report/).

The LaTeX tool is a separate Docker Compose service and is not included in the
application image. From the repository root, pass it one document:

```sh
./scripts/latex.sh docs/YOUR_DOCUMENT.tex
```

The first run downloads the full TeX Live image. `latexmk` writes the PDF and
all temporary build files to the gitignored `out/` directory.

The modular thesis workspace is in [`thesis/`](thesis/README.md). Run
`./thesis/build.sh` to produce `out/thesis/thesis.pdf` using the same Compose
service with LuaLaTeX.

## Local development

Install the package directly only when the local machine already has all
required build tools:

```sh
gcc -O3 -fPIC -shared src/lm_idnet/algorithms/native/LM_time_lib.c \
  -lm -o src/lm_idnet/algorithms/native/LM_time_lib.so
python -m pip install -e '.[test]'
```

Run the CLI without setting `PYTHONPATH`:

```sh
lm-idnet --help
lm-idnet preprocess --config configs/d_link_day_cam5.json
lm-idnet diagnose --config configs/d_link_day_cam5.json
lm-idnet train --config configs/d_link_day_cam5.json
lm-idnet calibrate --config configs/d_link_day_cam5.json
lm-idnet score --config configs/d_link_day_cam5.json
lm-idnet adapt --config configs/d_link_day_cam5.json
```

The stable command set is `preprocess`, `diagnose`, `train`, `calibrate`,
`score`, `evaluate`, `adapt`, `run`, `rollback`, and `benchmark`. Use
`lm-idnet <command> --help` for command-specific options. Every command supports
`--dry-run` to validate its configuration and CLI wiring without executing the
stage.

Run the adaptive-threshold and both periodic-refit development variants with
the conservative update rules for both labelled devices:

```sh
./scripts/run_adaptation_experiments.sh
```

Run the earlier adaptation methods used in the thesis development results:

```sh
./scripts/run_adaptation_experiments.sh legacy
```

This selects the six configurations in `configs/experiments/legacy/`, including
adaptive thresholding and both periodic-refit schedules. They restore the
pre-Step-2 policy and write to `artifacts/experiments/legacy/` and
`reports/experiments/legacy/`. Historical results and conservative runs have
separate output paths. Both policies use the existing `adapt` CLI and Python
`adapt_windows(config)` entry point; `adaptation.policy` selects `legacy` or the
default `conservative` policy. For one run:

```sh
./scripts/run.sh adapt --config configs/experiments/legacy/unsw_chromecast_periodic_refit_288_240.json
./scripts/run.sh evaluate --config configs/experiments/legacy/unsw_chromecast_periodic_refit_288_240.json
```

Legacy threshold updates use all recent scored windows, including anomalies.
Legacy refits retain the original baseline confidence gate, convergence and
likelihood checks, but omit the Step 2 volume and movement limits; a rejected
refit can still update the threshold. These are reproducible research variants
with the original contamination risk. The selected policy is recorded in the
CLI result and adaptation report.

The runner uses the existing saved model and threshold for each device. It
does not retrain, recalibrate, or overwrite the saved static results. Conservative
runs use the matching files under `configs/experiments/` and write events and
evaluation reports under `artifacts/experiments/conservative/` and
`reports/experiments/conservative/`. Earlier adaptive results remain under
`artifacts/experiments/<device>/<variant>/` and
`reports/experiments/<device>/<variant>/` for comparison. The six configurations
keep the previous capture partitions, seeds, scoring convention, buffer sizes,
and update timing, and explicitly record the new safety limits. Original PCAPs
are not needed; conservative runs need processed calibration and development
counts, while legacy runs need only processed development counts.

If starting models or thresholds are absent, first run `train` and `calibrate`
with each device's configuration. Replacing those artifacts also changes the
starting point of the comparison.

To save accepted updates and resume them on the next invocation, supply a state
directory explicitly. Use the same configuration and directory for rollback:

```sh
lm-idnet adapt --config configs/experiments/legacy/unsw_chromecast_periodic_refit_288_240.json --state-dir artifacts/UNSW-Chromecast/detector-state
lm-idnet rollback --config configs/experiments/legacy/unsw_chromecast_periodic_refit_288_240.json --state-dir artifacts/UNSW-Chromecast/detector-state
```

The saved pair is restored before scoring, and a rejected save keeps the previous
pair active. The directory permits one adaptive run or rollback at a time; another
writer receives a clear error. Omitting `--state-dir` keeps the original experiment
starting point. Resuming still replays the configured development captures;
learning buffers and input positions are not restored.

The state directory retains the active and previous versions plus `history.json`
with the latest 100 update/rollback decisions. Interrupted-save leftovers are
cleaned on restoration. Original experiment artifacts remain separate.

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

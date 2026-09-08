# Command-line interface

The installed executable is `lm-idnet`. It exposes one subcommand per pipeline
stage and never performs another stage implicitly.

```text
preprocess  Read configured captures and write processed windows
diagnose    Report statistics and daily psi for processed windows
train       Fit the normal-traffic model
calibrate   Create a model-bound anomaly threshold
score       Score windows and emit anomaly decisions
evaluate    Evaluate detector outputs
adapt       Assess an adaptive-model candidate
forecast    Forecast traffic using a verified model
benchmark   Measure backend and pipeline performance
```

`diagnose` writes one JSON report for all configured captures. Each entry
includes its capture ID, temporal partition, descriptive category statistics,
and a daily Dirichlet-multinomial fit with `psi`, iteration count, and
convergence status. By default the report is
`reports/D-LinkDayCam5/capture_statistics.json`; use `--output PATH` to choose
a different location.

Run `lm-idnet --help` for the complete list or
`lm-idnet <command> --help` for command-specific arguments.

## Common contract

Every command requires an explicit configuration:

```sh
lm-idnet preprocess --config configs/d_link_day_cam5.json
```

Every command also supports a non-mutating readiness check:

```sh
lm-idnet score --config configs/d_link_day_cam5.json --dry-run
```

A successful dry run means the command name, package wiring, and configuration
are valid. It does not claim that the underlying research stage is implemented
or execute that stage.

Successful command and dry-run status records are JSON on standard output.
Expected failures produce no success record, write to standard error, and
return a non-zero exit code.

Informational application logs are written to `logs/lm_idnet.log`. Add the
global `--verbose` option before the command to also show them in the terminal:

```sh
lm-idnet --verbose preprocess --config configs/d_link_day_cam5.json
```

Log timestamps use ISO 8601 UTC with millisecond precision and an explicit `Z`
suffix, for example `2026-08-06T10:38:04.975Z`. They do not depend on the host
or container timezone.

Use JSON-formatted runtime errors when integrating with scripts:

```sh
lm-idnet --error-format json train --config configs/d_link_day_cam5.json
```

Argument-parser errors, such as a missing required `--config`, use argparse's
standard exit code 2 and usage output.

## Implementation status

At this milestone, every registered command has a callable handler. `train` loads the
fit-partition matrix, estimates Dirichlet parameters with Minka's fixed-point
iteration and the configured likelihood backend, and saves a validated model artifact.
`calibrate` scores the configured calibration captures and saves a validated
lower-quantile threshold artifact. `score` applies that threshold to
non-missing development-test windows and writes a JSON array with one result per
window. `evaluate` compares development-test anomaly decisions with matching
window labels and writes `evaluation_statistics.json` under the configured
reports directory. `benchmark` compares LM and SciPy on the same training,
calibration, and scoring workload and writes `lm_backend_benchmark.json` under
the configured reports directory. `adapt` runs the configured `static` or
`adaptive_threshold` mode over the same chronological development-test stream.
Adaptive-threshold mode keeps alpha fixed, scores each window before updating,
and periodically replaces the threshold and score IQR with estimates from the
bounded buffer of all recent non-missing scores. Labels and the current anomaly
decision do not filter that intentionally naive baseline. An update with a
non-positive score IQR is recorded and rejected, leaving the prior threshold
active. The command writes ordinary anomaly events plus
`adaptation.json` under the configured reports directory. Periodic alpha
refitting is not implemented.

## Stage isolation

`train`, `diagnose`, `calibrate`, `forecast`, and later downstream commands
consume existing processed artifacts. They do not silently invoke PCAP
preprocessing. Preprocessing happens only through `lm-idnet preprocess`.

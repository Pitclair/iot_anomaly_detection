# Command-line interface

The installed executable is `lm-idnet`. It exposes one subcommand per pipeline
stage and never performs another stage implicitly.

```text
preprocess  Read configured captures and write processed windows
diagnose    Report statistics for already processed windows
train       Fit the normal-traffic model
calibrate   Create a model-bound anomaly threshold
score       Score windows and emit anomaly decisions
evaluate    Evaluate detector outputs
adapt       Assess an adaptive-model candidate
forecast    Forecast traffic using a verified model
benchmark   Measure backend and pipeline performance
```

`diagnose` writes one JSON report for all configured captures. By default the
report is `reports/capture_statistics.json`; use `--output PATH` to choose a
different location. Each entry includes its capture ID and temporal partition.

Run `lm-idnet --help` for the complete list or
`lm-idnet <command> --help` for command-specific arguments.

## Common contract

Every command requires an explicit configuration:

```sh
lm-idnet preprocess --config configs/config.json
```

Every command also supports a non-mutating readiness check:

```sh
lm-idnet score --config configs/config.json --dry-run
```

A successful dry run means the command name, package wiring, and configuration
are valid. It does not claim that the underlying research stage is implemented
or execute that stage.

Successful command and dry-run status records are JSON on standard output.
Expected failures produce no success record, write to standard error, and
return a non-zero exit code.

Use JSON-formatted runtime errors when integrating with scripts:

```sh
lm-idnet --error-format json train --config configs/config.json
```

Argument-parser errors, such as a missing required `--config`, use argparse's
standard exit code 2 and usage output.

## Implementation status

At this milestone, `preprocess`, `diagnose`, and `forecast` have callable
handlers. `train`, `calibrate`, `score`, `evaluate`, `adapt`, and `benchmark`
are registered stable interfaces but intentionally return
`command_unavailable_error` (exit code 9) when executed. Their `--help` and
`--dry-run` paths work normally.

This explicit failure prevents automation from mistaking an empty placeholder
for successful model training or anomaly detection. Each later roadmap task
replaces its unavailable handler only when that stage has a real implementation
and acceptance tests.

## Stage isolation

`train`, `diagnose`, `forecast`, and later downstream commands consume existing
processed artifacts. They do not silently invoke PCAP preprocessing.
Preprocessing happens only through `lm-idnet preprocess`.

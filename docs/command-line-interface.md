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
rollback    Restore the previous accepted detector version
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
the configured reports directory. `adapt` runs the configured `static`,
`adaptive_threshold`, or `periodic_refit` mode over the same chronological
development-test stream. `adaptation.policy` selects `conservative` (default)
or the earlier `legacy` policy. The following rules describe conservative runs.
Adaptive-threshold mode keeps alpha fixed, scores each
window before updating, and periodically replaces the threshold and score IQR
with estimates from a bounded buffer of high-confidence accepted windows.
Flagged and near-threshold windows are excluded. Windows above the configured
quantile of benign calibration packet totals are also excluded from learning,
even when their protocol proportions look normal. Both adaptive modes use these
checks; labels are never consulted for admission or updates.

Periodic-refit mode uses the same accepted buffer. Windows must clear the
high-confidence gate under both the active and original static detectors. On each
update interval, it fits a candidate alpha when enough safe windows are
available. One one-hot smoothing row per category keeps an absent protocol from
forcing its alpha to zero. The candidate is promoted only if fitting converges,
its likelihood on that same guarded matrix is not worse than the active model,
its concentration and category proportions stay within configured movement
limits, and rescoring the accepted window buffer produces finite scores and a
positive finite score IQR. The matching candidate threshold must also stay
within its movement limit. Alpha,
expected profile, model fingerprint, threshold, and score IQR are then replaced
together. A rejected refit preserves the model, threshold, and score IQR;
there is no separate threshold fallback. Every scheduled attempt, including
insufficient evidence, is recorded with its acceptance or rejection reason in
`adaptation.json`. The report also records the packet-count limit, movement
limits, and counts of windows excluded from learning. See
[`configuration.md`](configuration.md#conservative-update-limits) for defaults
and units. Existing processed calibration JSON supplies the volume limit;
PCAPs are not needed for this stage.

## Earlier adaptation methods

The same `adapt` handler exposes the original adaptive-threshold and
periodic-refit methods when the configuration sets `adaptation.policy` to
`legacy`. These methods retain all-recent-window threshold calibration and
the threshold fallback after a rejected refit, without the Step 2 volume or
movement limits. Periodic fitting still uses the original static confidence
gate, smoothing, convergence, parameter validation, and likelihood comparison.
Legacy runs need processed development counts and the saved starting model
and threshold; they do not load calibration counts or attack labels.

```sh
lm-idnet adapt --config configs/experiments/legacy/unsw_chromecast_periodic_refit_288_240.json
lm-idnet evaluate --config configs/experiments/legacy/unsw_chromecast_periodic_refit_288_240.json
```

Use `./scripts/run_adaptation_experiments.sh legacy` to run and evaluate all six
earlier device/variant configurations with Docker. Outputs go under
`artifacts/experiments/legacy/` and `reports/experiments/legacy/`; historical,
static, and conservative outputs remain separate. The runner defaults to
`conservative` if its policy argument is omitted. CLI results include
`adaptation_policy`, and `adaptation.json` includes `policy` so results can be
attributed to their method. Legacy results retain the earlier contamination
risk and should be identified as that research policy in the thesis.

## Saved detector state

`adapt --state-dir DIRECTORY` initializes or restores an accepted pair and saves
each permitted update before using it. The same option is available to Python
callers as `adapt_windows(config, state_dir=directory)`. Omitting the directory
preserves fresh-start experiment behavior for both adaptation policies.

```sh
lm-idnet adapt --config configs/experiments/legacy/unsw_chromecast_periodic_refit_288_240.json --state-dir artifacts/UNSW-Chromecast/detector-state
lm-idnet rollback --config configs/experiments/legacy/unsw_chromecast_periodic_refit_288_240.json --state-dir artifacts/UNSW-Chromecast/detector-state
```

`rollback` requires `--state-dir`, validates the previous snapshot against the
configuration, and atomically swaps the active and previous identifiers. It does
not retrain or require the initial artifacts or calibration counts. Missing
previous versions and busy state directories fail with exit code 8; incompatible
snapshots fail with exit code 7. No success JSON is emitted on failure.

Persistent adaptation and rollback use a nonblocking Linux file lock. Only one
can modify a directory at a time, including when paths refer to the same directory
through a symlink. Locks release when the process exits. Python callers modifying
state through low-level helpers must hold `detector_state_lock(directory)`.
`--dry-run` validates arguments and configuration without opening a state directory.
Resuming a batch run replays its configured captures with the restored pair;
learning buffers, update counters, and input positions are not restored.

## Stage isolation

`train`, `diagnose`, `calibrate`, `forecast`, and later downstream commands
consume existing processed artifacts. They do not silently invoke PCAP
preprocessing. Preprocessing happens only through `lm-idnet preprocess`.

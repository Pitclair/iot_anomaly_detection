# Artifact schemas

LM-IDNet stores models and results as readable JSON. Each artifact has an
`artifact_type` that selects its Pydantic schema. Unknown fields and invalid
values are rejected when an artifact is loaded.

Detector snapshots carry a schema version. Optional scoring fields preserve
loading of older artifacts; existing snapshots retain their embedded thresholds.

## Registered artifact types

| Artifact type | Purpose |
|---|---|
| `model` | Fitted Dirichlet parameters, training provenance, and convergence diagnostics |
| `threshold` | Model fingerprint, score convention, quantile, threshold, and calibration IQR |
| `anomaly_event` | Device/window context, decision, severity, category residuals, and model fingerprint |
| `experiment_manifest` | Run, command, data, configuration, and seed provenance |
| `detector_state` | Active alpha and matching threshold, original safety references, and learning settings |
| `detector_state_index` | Identifiers of the active and previous detector snapshots |
| `detector_state_history` | The latest 100 adaptive decisions and operator rollbacks |

Schemas live with the application layer that owns their data:

- processed capture schemas are in `processing/schemas.py`;
- model, calibration, and scoring schemas are in `models/schemas.py`;
- experiment-level schemas are in `evaluation/schemas.py`.

The shared `artifacts.py` module only routes artifact types and handles JSON
loading and saving.

A model stores alpha, concentration, mean probabilities, and psi together with
the training capture IDs and likelihood backend. Its fitting diagnostics
include convergence settings, likelihoods, and runtime.

Processed captures use their own validated JSON format in `processing.storage`.

New threshold artifacts include `count_tails` for `total` and `arp`. Each entry
contains a finite, nonnegative log-count cutoff and a positive finite reference
IQR. Both entries are required together. Events add `composition_score` and
`feature_margins`; their combined score is the minimum margin and alerts below
zero. DM-only artifacts omit these features. The threshold `score_iqr` is a fit
reference scale for the joint detector and a calibration IQR for older artifacts.
See [score semantics](likelihood-semantics.md#joint-detector-score).

## Loading artifacts

Consumers load artifacts through the shared validation boundary:

```python
from lm_idnet.artifacts import load_artifact

model = load_artifact(
    "artifacts/D-LinkDayCam5/model.json", expected_type="model"
)
```

Missing, unreadable, mistyped, or invalid artifacts raise
`ArtifactCompatibilityError`.

## Detector snapshots

`DetectorStateArtifact` keeps learned alpha and its fingerprint-bound threshold
and score IQR together. It embeds the original model and threshold without
rewriting their training or calibration provenance. Window duration, device
MAC, calibration/estimator/adaptation settings, and the calibration-derived
packet-count limit are saved for compatible restoration. Legacy and static
states have no packet-count limit.

Snapshot validation rejects invalid parameters, non-finite threshold statistics,
mismatched fingerprints, incompatible calibration settings, and unusable buffer
sizes. Each snapshot has its own version identifier and timestamp. A threshold
change receives a new version even when the model fingerprint stays the same.

`save_detector_version(directory, state)` writes one complete snapshot to
`versions/<version_id>.json` and rereads it through validation. Existing version
files are never overwritten. `load_detector_version(directory, version_id)`
rejects unsafe identifiers and a snapshot whose embedded identifier differs
from its filename. The index format reserves `current.json` for the active and
previous identifiers; saving an individual version does not activate it.

## Committing an active version

`save_detector_state(directory, state)` saves a new immutable snapshot and then
atomically replaces `current.json`, retaining the previous active identifier.
Temporary files are created alongside their destinations, flushed to disk, and
reread through validation before publication. File and directory synchronization
targets a local Linux filesystem. Updating a state directory requires one writer.

The pointer replacement is the commit point. A failure before replacement leaves
the previous pair active. A failure to synchronize the directory after replacement
raises `DetectorStateCommitError` with `committed=True`; callers must reload or stop
instead of reporting a rejected update. Both referenced pairs remain complete.
Temporary-file cleanup failures are logged without undoing a committed switch.
After a successful save, unreferenced managed snapshots and abandoned temporary
files are removed. Only the active and previous versions are retained. The same
cleanup runs on restoration after both referenced snapshots have been validated.
Cleanup failures are logged and retried on a later restoration or successful save;
they never undo a committed pointer switch. Unrelated files are left alone.

## Restoring a detector

`load_detector_state(directory, config)` follows `current.json`, validates its
snapshot against the configured device, dataset, category order, window duration,
capture provenance, numerical backend/precision, and learning settings, and returns
the accepted state. An existing corrupt or incompatible state raises a clear error;
it never silently resets to the initial model.

If the directory is absent or completely empty, the loader initializes and commits
a snapshot using the configured initial model and threshold. Conservative adaptive
initialization derives its packet-count limit from calibration windows using the
same helper as batch scoring. Legacy and static initialization do not need those
windows. Subsequent loads use only the snapshot and configuration, so the initial
files and calibration data can be absent. Changing filesystem paths does not change
compatibility.

```python
from lm_idnet.artifacts import load_detector_state
from lm_idnet.config import load_config

config = load_config("configs/experiments/unsw_chromecast_periodic_refit.json")
state = load_detector_state("artifacts/UNSW-Chromecast/detector-state", config)
```

The returned state includes the active alpha/threshold/IQR and the original safety
references. Traffic buffers and update scheduling counters start empty when a
caller resumes; no traffic history or input position is restored.

`adapt_windows(config, state_dir=directory)` restores this state before scoring
and commits each accepted update before activating it in memory. Storage failures
before the pointer switch reject the update and preserve the whole active pair;
they do not trigger the legacy threshold fallback. Legacy threshold fallbacks
caused by fitting failures are themselves saved before activation. A post-switch
`DetectorStateCommitError` stops the run so its next invocation can reload the
committed pair. The report records starting and final version identifiers.

Omitting `state_dir` keeps existing experiment behavior. Resuming a batch run
replays its configured captures with the restored state; this is not input-offset
recovery. A single writer must own the directory while running adaptive scoring.
`adapt_windows` holds `detector_state_lock(directory)` for the entire persistent
run. `rollback_detector_state(directory, config)` takes the same lock, validates
the previous snapshot, and atomically swaps the active and previous identifiers.
Direct callers of the lower-level storage helpers must hold the lock themselves.
The `.lock` file remains in place so processes always lock the same inode.

## Bounded update history

Persistent adaptation and rollback write `history.json` atomically. It retains the
latest 100 decisions, recording timestamps, method, old/new version identifiers,
acceptance or rejection, rejection reason, and the batch window position when
applicable. Reasons are capped at 1024 characters. A rejected legacy refit followed
by an accepted threshold fallback produces separate records.

The active pointer and validated snapshot are authoritative. History is diagnostic:
an interrupted process can commit an update before recording it. History failures
are logged without rejecting an already committed update or rollback. Standalone
storage helpers save versions without synthesizing adaptive decision records.
The batch event array and experiment report remain available for full development
results; this retention policy bounds the state directory and its recent audit.

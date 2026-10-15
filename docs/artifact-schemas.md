# Artifact schemas

LM-IDNet stores models and results as readable JSON. Each artifact has an
`artifact_type` that selects its Pydantic schema. Unknown fields and invalid
values are rejected when an artifact is loaded.

Detector snapshots carry a schema version. Existing training and result formats
remain unchanged; there is no automatic migration of older artifacts.

## Registered artifact types

| Artifact type | Purpose |
|---|---|
| `model` | Fitted Dirichlet parameters, training provenance, and convergence diagnostics |
| `threshold` | Model fingerprint, score convention, quantile, threshold, and calibration IQR |
| `anomaly_event` | Device/window context, decision, severity, category residuals, and model fingerprint |
| `experiment_manifest` | Run, command, data, configuration, and seed provenance |
| `detector_state` | Active alpha and matching threshold, original safety references, and learning settings |
| `detector_state_index` | Identifiers of the active and previous detector snapshots |

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
Unreferenced snapshots can remain after interrupted saves; retention and automatic
promotion are separate from these storage helpers.

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

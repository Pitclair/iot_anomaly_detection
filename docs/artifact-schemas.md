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

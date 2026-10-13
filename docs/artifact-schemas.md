# Artifact schemas

LM-IDNet stores models and results as readable JSON. Each artifact has an
`artifact_type` that selects its Pydantic schema. Unknown fields and invalid
values are rejected when an artifact is loaded.

Artifacts do not currently carry schema versions, checksums, or migration
logic. Those features can be added when the formats need long-term stability.

## Registered artifact types

| Artifact type | Purpose |
|---|---|
| `model` | Fitted Dirichlet parameters, training provenance, and convergence diagnostics |
| `threshold` | Model fingerprint, score convention, quantile, threshold, and calibration IQR |
| `anomaly_event` | Device/window context, decision, severity, category residuals, and model fingerprint |
| `experiment_manifest` | Run, command, data, configuration, and seed provenance |

Schemas live with the application layer that owns their data:

- processed capture schemas are in `processing/schemas.py`;
- model, calibration, and scoring schemas are in `models/schemas.py`;
- experiment-level schemas are in `evaluation/schemas.py`.

The shared `artifacts.py` module only routes artifact types and handles JSON
loading and saving.

A model stores alpha, concentration, mean probabilities, and psi together with
the training capture IDs, likelihood backend, and required `input_policy`.
That policy records the configured device MAC and window duration, plus the
processing code's SSDP classification and window-boundary rules. It is also
stored in processed-capture metadata and included in the model fingerprint.
Changing the policy invalidates the old model-bound threshold.

Training, calibration, and scoring share checks for capture identity, device,
partition, category order, recorded input policy, and actual window duration.
Counts must be non-negative integers; booleans and floating-point counts are
rejected. Packet totals must fit the detector's integer representation.
Missing observations remain distinct from genuine zero-count windows.
Threshold values and their score summaries must be finite.

Fitting diagnostics include convergence settings, likelihoods, and runtime.

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

Direct callers of `score_window()` must provide `model_window_minutes` from
`model.input_policy.window_minutes`; the scorer checks the actual window duration.

## Recreating older files

Processed captures and models without `input_policy` are rejected. Their policy
is not inferred from today's configuration. For each affected configuration,
run `preprocess`, then `train` and `calibrate`, and finally `score` or `adapt`
to recreate dependent outputs. Keep historical results separately if needed.
Existing files are not silently migrated or modified during loading.

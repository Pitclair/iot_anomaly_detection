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
| `threshold` | Score convention, quantile, and threshold |
| `anomaly_event` | Window decision and score |
| `experiment_manifest` | Run, command, data, configuration, and seed provenance |

The Pydantic definitions are in `src/lm_idnet/artifact_schemas.py`.

A model stores alpha, concentration, mean probabilities, and psi together with
the training capture IDs and likelihood backend. Its fitting diagnostics
include convergence settings, likelihoods, and runtime.

Processed captures use their own validated JSON format in `processing.storage`.

## Loading artifacts

Consumers load artifacts through the shared validation boundary:

```python
from lm_idnet.artifacts import load_artifact

model = load_artifact("artifacts/model.json", expected_type="model")
```

Missing, unreadable, mistyped, or invalid artifacts raise
`ArtifactCompatibilityError`.

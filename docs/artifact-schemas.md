# Artifact schema versioning

Every versioned LM-IDNet model or result artifact has:

- `artifact_type`, identifying the schema family;
- `schema_version`, written as semantic `MAJOR.MINOR.PATCH`;
- `checksum`, containing the canonical SHA-256 integrity digest.

The current schema version is `1.2.0`. The explicitly supported previous
versions are `1.1.0` and `1.0.0`.

## Registered artifact types

| Artifact type | Purpose |
|---|---|
| `model` | Fitted Dirichlet parameters, training provenance, and convergence diagnostics |
| `threshold` | Model-bound score convention, quantile, and threshold |
| `anomaly_event` | Window decision, score, threshold, and model identity |
| `experiment_manifest` | Run, command, code, dataset, configuration, and seed provenance |

The Pydantic definitions and migration registry are in
`src/lm_idnet/artifact_schemas.py`.

A current model artifact stores alpha, concentration, mean probabilities, and
psi together with the training capture IDs and likelihood backend. Its fitting
diagnostics include initial alpha, iteration and convergence state, configured
tolerance and iteration limit, initial and final log-likelihoods, and runtime.

Processed captures deliberately use a simpler boundary: one validated JSON file
per capture, loaded and saved through `processing.storage`. They do not have a
second matrix export, schema migration, or per-file checksum.

## Compatibility rules

- The exact current version loads directly.
- Versions `1.1.0` and `1.0.0` are upgraded through explicit migrations.
- A same-major version with no registered migration is rejected.
- A newer version than the software supports is rejected.
- An unknown major version is always rejected.
- Missing or malformed semantic versions are rejected.
- An artifact whose declared type differs from the requested type is rejected.
- Unknown fields are rejected by every current schema.

Migration is never used to bypass integrity. The loader first verifies the
checksum of the original artifact, applies the registered migration, validates
the resulting current schema, and calculates a checksum for the migrated
representation.

Models written before `1.2.0` did not contain training provenance or fitting
diagnostics. Their concentration, mean probabilities, and psi are derived from
alpha during migration. Unavailable capture IDs, likelihood backend, and
diagnostics remain explicitly empty rather than being inferred.

## Loading artifacts

Consumers should not parse artifact JSON directly:

```python
from lm_idnet.artifacts import load_typed_artifact

model = load_typed_artifact("artifacts/model.json", expected_type="model")
```

The scoring boundary uses `load_model_for_scoring`, which applies the same
integrity, migration, and schema checks. Any failure raises
`ArtifactCompatibilityError` or `ArtifactIntegrityError`; consumers must stop.

## Changing a schema

For a backward-compatible addition, increment the minor version and add an
explicit migration from the previous supported version. For a breaking change,
increment the major version and do not silently interpret old or future
artifacts as the new structure. Add round-trip, migration, wrong-type, and
unsupported-version tests for every affected artifact family.

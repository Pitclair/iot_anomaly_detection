# Artifact schema versioning

Every persisted LM-IDNet artifact has:

- `artifact_type`, identifying the schema family;
- `schema_version`, written as semantic `MAJOR.MINOR.PATCH`;
- `checksum`, containing the canonical SHA-256 integrity digest.

The current schema version is `1.1.0`. The explicitly supported previous
version is `1.0.0`.

## Registered artifact types

| Artifact type | Purpose |
|---|---|
| `processed_dataset` | Processed capture metadata and validated count windows |
| `model` | Category ordering and fitted Dirichlet parameters |
| `threshold` | Model-bound score convention, quantile, and threshold |
| `anomaly_event` | Window decision, score, threshold, and model identity |
| `experiment_manifest` | Run, command, code, dataset, configuration, and seed provenance |

The Pydantic definitions and migration registry are in
`src/lm_idnet/artifact_schemas.py`.

## Compatibility rules

- The exact current version loads directly.
- Version `1.0.0` is upgraded through an explicit migration.
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

# Exception and failure semantics

LM-IDNet treats expected operational failures as typed domain exceptions.
Library code raises an exception; the command-line boundary converts it to one
stable non-zero exit code and one concise message on standard error.

| Exit | Error code | Meaning |
|---:|---|---|
| 1 | `application_error` | Generic typed application failure |
| 2 | `configuration_error` | Missing, unreadable, or invalid configuration |
| 3 | `ingestion_error` | Capture reading or preprocessing failed |
| 4 | `data_validation_error` | Input data violates its schema or invariants |
| 5 | `numerical_precision_error` | Required numerical precision was not achieved |
| 6 | `convergence_error` | Estimation did not converge acceptably |
| 7 | `artifact_compatibility_error` | Artifact is missing, unreadable, or incompatible |
| 7 | `artifact_integrity_error` | Artifact checksum is missing or incorrect |
| 8 | `policy_rejection_error` | A safety or promotion policy rejected an action |

Messages use this machine-readable prefix:

```text
ERROR [configuration_error]: config not found: configs/missing.json
```

Expected failures do not print a success message or continue with partial
results. The original exception is retained as the Python exception cause for
debugging and tests.

## Artifact integrity

`lm_idnet.artifacts.load_verified_artifact` reads JSON artifacts and verifies a
SHA-256 checksum over canonical JSON, excluding the `checksum` field itself.
Comparison uses a constant-time digest comparison.

Model consumers must use `load_model_for_scoring`. A missing checksum, changed
parameter, malformed artifact, or mismatched checksum raises a typed exception
before the model is returned. Scoring must never catch this exception and
continue with an unverified model.

Artifact schema/version compatibility will be expanded with the versioned
artifact work. Integrity checking already fails closed rather than treating a
corrupt model as a warning.

## Raising failures

Use the narrowest domain exception that describes the failed contract:

```python
from lm_idnet.exceptions import NumericalPrecisionError

if achieved_digits < requested_digits:
    raise NumericalPrecisionError(
        f"achieved {achieved_digits} digits; requested {requested_digits}"
    )
```

Do not call `sys.exit()` from library modules and do not convert an integrity,
precision, convergence, or validation failure into a normal return value.

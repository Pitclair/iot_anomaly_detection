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
| 8 | `policy_rejection_error` | A safety or promotion policy rejected an action |

Messages use this machine-readable prefix:

```text
ERROR [configuration_error]: config not found: configs/missing.json
```

Expected failures do not print a success message or continue with partial
results. The original exception is retained as the Python exception cause for
debugging and tests.

## Artifact validation

`lm_idnet.artifacts.load_artifact` reads JSON artifacts and validates them
against the requested Pydantic schema. Missing, malformed, mistyped, or invalid
artifacts raise `ArtifactCompatibilityError` before their values are used.

## Raising failures

Use the narrowest domain exception that describes the failed contract:

```python
from lm_idnet.exceptions import NumericalPrecisionError

if achieved_digits < requested_digits:
    raise NumericalPrecisionError(
        f"achieved {achieved_digits} digits; requested {requested_digits}"
    )
```

Do not call `sys.exit()` from library modules and do not convert a precision,
convergence, or validation failure into a normal return value.

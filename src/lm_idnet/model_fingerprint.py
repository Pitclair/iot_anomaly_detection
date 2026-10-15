"""Stable identity for the parameters used when scoring windows."""

import hashlib
import json


def model_parameter_fingerprint(
    *,
    alpha: tuple[float, ...],
    categories: tuple[str, ...],
    log_likelihood_backend: str,
    precision_digits: int,
) -> str:
    """Fingerprint only the parameter values used while scoring."""
    canonical = json.dumps(
        {
            "alpha": alpha,
            "categories": categories,
            "log_likelihood_backend": log_likelihood_backend,
            "precision_digits": precision_digits,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


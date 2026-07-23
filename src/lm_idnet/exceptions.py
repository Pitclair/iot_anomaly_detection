"""Domain exceptions and stable process exit codes."""

from __future__ import annotations


class LMIDNetError(Exception):
    """Base class for expected, user-facing application failures."""

    exit_code = 1
    error_code = "application_error"


class ConfigurationError(LMIDNetError):
    exit_code = 2
    error_code = "configuration_error"


class IngestionError(LMIDNetError):
    exit_code = 3
    error_code = "ingestion_error"


class DataValidationError(LMIDNetError):
    exit_code = 4
    error_code = "data_validation_error"


class NumericalPrecisionError(LMIDNetError):
    exit_code = 5
    error_code = "numerical_precision_error"


class ConvergenceError(LMIDNetError):
    exit_code = 6
    error_code = "convergence_error"


class ArtifactCompatibilityError(LMIDNetError):
    exit_code = 7
    error_code = "artifact_compatibility_error"


class ArtifactIntegrityError(ArtifactCompatibilityError):
    """An artifact failed an integrity check and must not be consumed."""

    error_code = "artifact_integrity_error"


class PolicyRejectionError(LMIDNetError):
    exit_code = 8
    error_code = "policy_rejection_error"

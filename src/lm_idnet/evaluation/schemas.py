"""Schemas for experiment-level evaluation artifacts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExperimentManifestArtifact(BaseModel):
    """Record the inputs required to identify an experiment run."""

    model_config = ConfigDict(extra="forbid")

    artifact_type: Literal["experiment_manifest"] = "experiment_manifest"
    run_id: str = Field(min_length=1)
    command: str = Field(min_length=1)
    code_commit: str = Field(min_length=1)
    configuration_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int = Field(ge=0)

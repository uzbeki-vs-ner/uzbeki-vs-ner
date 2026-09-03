"""Run JSON schema for local eval comparison (matches train_ner.py)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from uzbek_ner.labels import ENTITY_LABELS

RUN_SCHEMA_VERSION = 1


class LabelMetrics(BaseModel):
    """Precision / recall / F1 plus confusion counts for one label (or micro)."""

    model_config = ConfigDict(extra="ignore")

    precision: float
    recall: float
    f1: float
    tp: int = 0
    fp: int = 0
    fn: int = 0
    gold: int = 0
    predicted: int = 0


class MacroMetrics(BaseModel):
    """Unweighted mean of per-label precision / recall / F1."""

    model_config = ConfigDict(extra="ignore")

    precision: float
    recall: float
    f1: float


class ExactSpanMetrics(BaseModel):
    """Organizer-compatible exact-span payload from ``write_metrics``."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int = 1
    matching: str = "same hash and exact label/start/end"
    records: int
    by_label: dict[str, LabelMetrics]
    micro: LabelMetrics
    macro: MacroMetrics

    @field_validator("by_label")
    @classmethod
    def require_entity_labels(cls, value: dict[str, LabelMetrics]) -> dict[str, LabelMetrics]:
        missing = [label for label in ENTITY_LABELS if label not in value]
        if missing:
            raise ValueError(f"metrics.by_label missing {missing}")
        return value


class EvalPaths(BaseModel):
    model_config = ConfigDict(extra="ignore")

    gold: str = ""
    predictions: str = ""
    metrics: str = ""


class BoundaryMetrics(BaseModel):
    """Type-agnostic exact ``(start, end)`` P/R/F1."""

    model_config = ConfigDict(extra="ignore")

    precision: float
    recall: float
    f1: float
    tp: int = 0
    fp: int = 0
    fn: int = 0
    gold: int = 0
    predicted: int = 0


class TypeGivenBoundary(BaseModel):
    """Label accuracy on spans whose character offsets already match."""

    model_config = ConfigDict(extra="ignore")

    aligned_spans: int
    type_correct: int
    accuracy: float
    confusion: dict[str, dict[str, int]] = Field(default_factory=dict)


class ErrorBuckets(BaseModel):
    """Partition of gold/pred entities (counts, not the official score)."""

    model_config = ConfigDict(extra="ignore")

    exact_match: int = 0
    type_mismatch: int = 0
    partial_same_type: int = 0
    partial_diff_type: int = 0
    missed: int = 0
    spurious: int = 0


class SpanDiagnostics(BaseModel):
    """Why official micro-F1 is low. Not used for the leaderboard."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int = 1
    matching: str = "diagnostic only; organizer score remains exact label/start/end"
    boundary_exact: BoundaryMetrics
    type_given_boundary: TypeGivenBoundary
    buckets: ErrorBuckets
    reading_id: str = "mixed"
    reading: str = ""


class EvalRun(BaseModel):
    """One scored model run on a gold split (usually official_dev)."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int = RUN_SCHEMA_VERSION
    run_id: str
    model: str
    checkpoint: str = ""
    created_at: str
    split: str = "official_dev"
    hyperparams: dict[str, Any] = Field(default_factory=dict)
    metrics: ExactSpanMetrics
    diagnostics: SpanDiagnostics | None = None
    paths: EvalPaths = Field(default_factory=EvalPaths)

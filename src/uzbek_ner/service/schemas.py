"""Pydantic v2 models for the organizer HTTP contract."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

EntityLabel = Literal["ORG", "NAME", "GEO"]


class PredictItem(BaseModel):
    """One document in a predict batch."""

    model_config = ConfigDict(extra="ignore")

    hash: str
    text: str


class EntitySpan(BaseModel):
    """One exact-span entity in Unicode Python indices [start, end)."""

    model_config = ConfigDict(extra="forbid")

    label: EntityLabel
    start: int
    end: int

    @model_validator(mode="after")
    def check_offsets(self) -> Self:
        if not self.start < self.end:
            raise ValueError("entity offsets must satisfy start < end")
        return self


def _require_unique_hashes(items: list[PredictItem]) -> list[PredictItem]:
    hashes = [item.hash for item in items]
    if len(hashes) != len(set(hashes)):
        raise ValueError("hashes must be unique within the request")
    return items


class PredictResult(BaseModel):
    """Prediction for one input document (same hash, same batch order)."""

    hash: str
    entities: list[EntitySpan] = Field(default_factory=list)


class PredictResponse(BaseModel):
    """Successful POST /api/v1/predict envelope."""

    data: list[PredictResult]


class CanonicalEntitySpan(EntitySpan):
    """Scored span plus a sibling canon. Offsets still point at the model surface."""

    canon: str


class CanonicalPredictResult(BaseModel):
    hash: str
    entities: list[CanonicalEntitySpan] = Field(default_factory=list)


class CanonicalPredictResponse(BaseModel):
    """POST /internal/v1/predict — extra CASE field, not the organizer contract."""

    data: list[CanonicalPredictResult]


class HealthResponse(BaseModel):
    status: Literal["ok"]


PredictBatch = Annotated[
    list[PredictItem],
    Field(min_length=1),
    AfterValidator(_require_unique_hashes),
]

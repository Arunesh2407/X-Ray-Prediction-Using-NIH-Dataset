from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    label: str
    probability: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    selected: bool = False


class RegionAttribution(BaseModel):
    label: str
    laterality: str | None = None
    lung_zone: str | None = None
    description: str
    heatmap_path: str | None = None
    coordinates: list[float] = Field(default_factory=list)


class EvidenceSnippet(BaseModel):
    source: str
    title: str
    snippet: str
    url: str | None = None
    score: float | None = None


class GraphRelation(BaseModel):
    source: str
    relation: str
    target: str
    explanation: str


class ReportBundle(BaseModel):
    study_id: str
    predictions: list[PredictionItem]
    regions: list[RegionAttribution]
    evidence: list[EvidenceSnippet]
    graph_relations: list[GraphRelation]
    clinician_report: str
    patient_summary: str
    metadata: dict[str, Any] = Field(default_factory=dict)

from __future__ import annotations

from src.schema import EvidenceSnippet, GraphRelation, PredictionItem, RegionAttribution, ReportBundle


def build_report(
    study_id: str,
    predictions: list[PredictionItem],
    regions: list[RegionAttribution],
    evidence: list[EvidenceSnippet],
    graph_relations: list[GraphRelation],
) -> ReportBundle:
    positive = [item for item in predictions if item.selected and item.label != "No finding"]
    finding_lines = [f"- {item.label}" for item in positive]

    if not finding_lines:
        finding_lines = ["- No finding: no dominant abnormality selected by the stub model."]

    region_lines = [f"- {region.label}: {region.description}" for region in regions]
    evidence_lines = [f"- {item.source}: {item.title} - {item.snippet}" for item in evidence]
    relation_lines = [f"- {item.source} {item.relation} {item.target}: {item.explanation}" for item in graph_relations]

    clinician_report = "\n".join(
        [
            f"Study ID: {study_id}",
            "",
            "Impression:",
            *finding_lines,
            "",
            "Image explanation:",
            *region_lines,
            "",
            "Evidence:",
            *evidence_lines,
            "",
            "Graph reasoning:",
            *relation_lines,
        ]
    )

    if positive:
        patient_summary = (
            f"The image suggests possible findings in: {', '.join(item.label for item in positive[:3])}. "
            "A clinician should review the image, the evidence, and the final report before any decision."
        )
    else:
        patient_summary = (
            "The current stub workflow does not show a dominant abnormality. "
            "A clinician should review the image and the final report before any decision."
        )

    metadata = {
        "positive_labels": [item.label for item in positive],
        "evidence_count": len(evidence),
        "graph_relation_count": len(graph_relations),
    }

    return ReportBundle.model_construct(
        study_id=study_id,
        predictions=predictions,
        regions=regions,
        evidence=evidence,
        graph_relations=graph_relations,
        clinician_report=clinician_report,
        patient_summary=patient_summary,
        metadata=metadata,
    )

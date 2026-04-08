from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib import error as urllib_error
from urllib import request as urllib_request

from src.config import get_no_finding_label, get_reporting_messages
from src.runtime_secrets import get_runtime_secret
from src.schema import EvidenceSnippet, GraphRelation, PredictionItem, RegionAttribution, ReportBundle


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


@dataclass(frozen=True)
class StructuredReportDraft:
    clinician_report: str
    patient_summary: str


@dataclass(frozen=True)
class LLMCallResult:
    draft: StructuredReportDraft | None
    error_type: str | None = None
    error_detail: str | None = None


def _selected_findings(predictions: list[PredictionItem]) -> list[PredictionItem]:
    no_finding_label = get_no_finding_label()
    return [item for item in predictions if item.selected and item.label != no_finding_label]


def _build_template_report(
    study_id: str,
    predictions: list[PredictionItem],
    regions: list[RegionAttribution],
    evidence: list[EvidenceSnippet],
    graph_relations: list[GraphRelation],
) -> StructuredReportDraft:
    reporting_messages = get_reporting_messages()
    positive = _selected_findings(predictions)

    finding_lines = [f"- {item.label}: {item.probability:.2f}" for item in positive]
    if not finding_lines:
        finding_lines = [f"- {reporting_messages['no_predictions']}"]

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
        summary_prefix = reporting_messages["summary_prefix"]
        summary_suffix = reporting_messages["summary_suffix"]
        patient_summary = f"{summary_prefix} {', '.join(item.label for item in positive[:3])}. {summary_suffix}"
    else:
        patient_summary = reporting_messages["summary_no_findings"]

    return StructuredReportDraft(clinician_report=clinician_report, patient_summary=patient_summary)


def _build_llm_prompt(
    study_id: str,
    predictions: list[PredictionItem],
    regions: list[RegionAttribution],
    evidence: list[EvidenceSnippet],
    graph_relations: list[GraphRelation],
) -> str:
    selected = _selected_findings(predictions)
    payload = {
        "study_id": study_id,
        "selected_findings": [
            {
                "label": item.label,
                "probability": round(item.probability, 4),
                "threshold": round(item.threshold, 4),
            }
            for item in selected
        ],
        "regions": [region.model_dump() for region in regions],
        "evidence": [item.model_dump() for item in evidence],
        "graph_relations": [item.model_dump() for item in graph_relations],
    }

    return (
        "You are generating a chest X-ray decision-support report for a clinician and a patient. "
        "Use only the provided data. Do not invent findings, diagnoses, or confidence values. "
        "Write concise, well structured output in JSON with exactly these keys: "
        '{"clinician_report": string, "patient_summary": string}. '
        "The clinician report must use short sections with headings such as Impression, Image explanation, Evidence, and Graph reasoning. "
        "The patient summary must be a short plain-language paragraph. "
        "Return JSON only, no markdown fences.\n\n"
        f"INPUT DATA:\n{json.dumps(payload, ensure_ascii=True, indent=2)}"
    )


def _parse_json_object(text: str) -> dict[str, str]:
    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            return {str(key): str(value) for key, value in loaded.items()}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        loaded = json.loads(match.group(0))
        if isinstance(loaded, dict):
            return {str(key): str(value) for key, value in loaded.items()}

    raise ValueError("LLM response did not contain a JSON object")


def _call_groq_report(prompt: str) -> LLMCallResult:
    api_key = get_runtime_secret("GROQ_API_KEY")
    if not api_key:
        return LLMCallResult(draft=None, error_type="missing_key", error_detail="GROQ_API_KEY not found")

    model = get_runtime_secret("GROQ_MODEL", DEFAULT_GROQ_MODEL) or DEFAULT_GROQ_MODEL

    try:
        temperature = float(get_runtime_secret("GROQ_TEMPERATURE", "0.2"))
        timeout_seconds = float(get_runtime_secret("GROQ_TIMEOUT_SECONDS", "30"))
    except ValueError as exc:
        return LLMCallResult(draft=None, error_type="invalid_config", error_detail=str(exc))

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    }

    data = json.dumps(payload).encode("utf-8")
    request = urllib_request.Request(
        GROQ_API_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "xray-decision-support/0.1",
        },
        method="POST",
    )

    try:
        with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        error_body = ""
        try:
            error_body = exc.read().decode("utf-8")
        except Exception:
            error_body = str(exc)
        return LLMCallResult(draft=None, error_type=f"http_{exc.code}", error_detail=error_body[:240])
    except TimeoutError as exc:
        return LLMCallResult(draft=None, error_type="timeout", error_detail=str(exc))
    except urllib_error.URLError as exc:
        return LLMCallResult(draft=None, error_type="network_error", error_detail=str(exc))
    except ValueError as exc:
        return LLMCallResult(draft=None, error_type="invalid_response", error_detail=str(exc))

    choices = response_payload.get("choices", [])
    if not choices:
        return LLMCallResult(draft=None, error_type="invalid_response", error_detail="choices missing")

    message = choices[0].get("message", {})
    content = str(message.get("content", "")).strip()
    if not content:
        return LLMCallResult(draft=None, error_type="empty_content", error_detail="message content empty")

    try:
        parsed = _parse_json_object(content)
    except (json.JSONDecodeError, ValueError) as exc:
        return LLMCallResult(draft=None, error_type="parse_error", error_detail=str(exc))

    clinician_report = parsed.get("clinician_report", "").strip()
    patient_summary = parsed.get("patient_summary", "").strip()
    if not clinician_report or not patient_summary:
        return LLMCallResult(draft=None, error_type="invalid_schema", error_detail="missing clinician_report or patient_summary")

    return LLMCallResult(
        draft=StructuredReportDraft(clinician_report=clinician_report, patient_summary=patient_summary),
        error_type=None,
        error_detail=None,
    )


def build_report(
    study_id: str,
    predictions: list[PredictionItem],
    regions: list[RegionAttribution],
    evidence: list[EvidenceSnippet],
    graph_relations: list[GraphRelation],
) -> ReportBundle:
    llm_model = get_runtime_secret("GROQ_MODEL", DEFAULT_GROQ_MODEL) or DEFAULT_GROQ_MODEL
    llm_call = _call_groq_report(
        _build_llm_prompt(
            study_id=study_id,
            predictions=predictions,
            regions=regions,
            evidence=evidence,
            graph_relations=graph_relations,
        )
    )
    draft = llm_call.draft

    reporting_mode = "llm" if draft is not None else "template"
    if draft is None:
        draft = _build_template_report(study_id, predictions, regions, evidence, graph_relations)

    positive = _selected_findings(predictions)

    metadata = {
        "positive_labels": [item.label for item in positive],
        "evidence_count": len(evidence),
        "graph_relation_count": len(graph_relations),
        "reporting_mode": reporting_mode,
        "llm_model": llm_model,
        "llm_error": llm_call.error_type,
        "llm_error_detail": llm_call.error_detail,
    }

    return ReportBundle.model_construct(
        study_id=study_id,
        predictions=predictions,
        regions=regions,
        evidence=evidence,
        graph_relations=graph_relations,
        clinician_report=draft.clinician_report,
        patient_summary=draft.patient_summary,
        metadata=metadata,
    )

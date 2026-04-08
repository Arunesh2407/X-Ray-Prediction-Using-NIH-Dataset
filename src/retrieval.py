from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from src.schema import EvidenceSnippet, PredictionItem, RegionAttribution

DEFAULT_RETRIEVAL_PATH = Path(os.getenv("RETRIEVAL_DATA_PATH", "data/retrieval"))


@lru_cache(maxsize=4)
def load_corpus(source_path: str) -> dict[str, list[EvidenceSnippet]]:
    path = Path(source_path)
    if not path.exists():
        return {}

    files: list[Path]
    if path.is_dir():
        files = sorted(path.glob("*.json"))
    else:
        files = [path]

    corpus: dict[str, list[EvidenceSnippet]] = {}
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        corpus.update(_coerce_corpus(payload))
    return corpus


def retrieve_evidence(
    predictions: list[PredictionItem],
    regions: list[RegionAttribution],
    corpus_path: str | None = None,
) -> list[EvidenceSnippet]:
    corpus = load_corpus(corpus_path or str(DEFAULT_RETRIEVAL_PATH))
    labels = [item.label for item in predictions if item.selected and item.label in corpus]
    if not labels:
        labels = [region.label for region in regions if region.label in corpus]

    evidence: list[EvidenceSnippet] = []
    for label in labels[:3]:
        evidence.extend(corpus[label])
    return evidence


def _coerce_corpus(payload: object) -> dict[str, list[EvidenceSnippet]]:
    corpus: dict[str, list[EvidenceSnippet]] = {}

    if isinstance(payload, dict):
        for label, items in payload.items():
            corpus[str(label)] = [_coerce_snippet(item) for item in items or []]
        return corpus

    if isinstance(payload, list):
        for item in payload:
            snippet = _coerce_snippet(item)
            if isinstance(item, dict):
                label = str(item.get("label", snippet.title))
            else:
                label = snippet.title
            if label:
                corpus.setdefault(label, []).append(snippet)
        return corpus

    return corpus


def _coerce_snippet(item: object) -> EvidenceSnippet:
    if isinstance(item, EvidenceSnippet):
        return item
    if not isinstance(item, dict):
        return EvidenceSnippet(source="Unknown", title="Unknown", snippet=str(item))

    return EvidenceSnippet(
        source=str(item.get("source", "Unknown")),
        title=str(item.get("title", "Unknown")),
        snippet=str(item.get("snippet", "")),
        url=item.get("url"),
        score=item.get("score"),
    )

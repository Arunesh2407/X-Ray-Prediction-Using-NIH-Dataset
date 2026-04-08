from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from src.schema import GraphRelation, PredictionItem

DEFAULT_GRAPH_PATH = Path(os.getenv("GRAPH_DATA_PATH", "data/graph/relations.json"))


@lru_cache(maxsize=4)
def load_graph(source_path: str) -> dict[str, list[GraphRelation]]:
    path = Path(source_path)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _coerce_graph(payload)


def query_graph(predictions: list[PredictionItem], graph_path: str | None = None) -> list[GraphRelation]:
    graph = load_graph(graph_path or str(DEFAULT_GRAPH_PATH))
    relations: list[GraphRelation] = []
    for prediction in predictions:
        if prediction.selected and prediction.label in graph:
            relations.extend(graph[prediction.label])
    return relations


def _coerce_graph(payload: object) -> dict[str, list[GraphRelation]]:
    graph: dict[str, list[GraphRelation]] = {}

    if isinstance(payload, dict):
        for label, relations in payload.items():
            graph[str(label)] = [_coerce_relation(item) for item in relations or []]
        return graph

    if isinstance(payload, list):
        for item in payload:
            relation = _coerce_relation(item)
            graph.setdefault(relation.source, []).append(relation)
        return graph

    return graph


def _coerce_relation(item: object) -> GraphRelation:
    if isinstance(item, GraphRelation):
        return item
    if not isinstance(item, dict):
        return GraphRelation(source="", relation="", target="", explanation=str(item))

    return GraphRelation(
        source=str(item.get("source", "")),
        relation=str(item.get("relation", "")),
        target=str(item.get("target", "")),
        explanation=str(item.get("explanation", "")),
    )

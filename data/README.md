# Data Layout

This project expects real, local data files for the post-CNN pipeline.

## Retrieval
- Place retrieval JSON files under `data/retrieval/`
- Or point `RETRIEVAL_DATA_PATH` to a JSON file or directory

## GraphRAG
- Place the disease relation graph at `data/graph/relations.json`
- Or point `GRAPH_DATA_PATH` to a JSON file

## Format
- Retrieval data should map each label to a list of evidence snippets, or provide a list of labeled snippet objects.
- Graph data should map each label to a list of relation objects, or provide a flat list of relation objects.

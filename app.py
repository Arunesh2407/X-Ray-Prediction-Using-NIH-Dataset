from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.explainability import generate_gradcam_regions
from src.graph_rag import query_graph
from src.model_stub import RealCNNModel
from src.reporting import build_report
from src.retrieval import retrieve_evidence


st.set_page_config(page_title="Chest X-ray Decision Support", layout="wide")

st.title("Chest X-ray Decision Support Prototype")
st.caption("The CNN now loads the uploaded PyTorch weights and the remaining stages read from real data files.")

weights_path = Path(os.getenv("MODEL_WEIGHTS_PATH", "weights/best_model.pth"))
retrieval_path = Path(os.getenv("RETRIEVAL_DATA_PATH", "data/retrieval"))
graph_path = Path(os.getenv("GRAPH_DATA_PATH", "data/graph/relations.json"))

st.sidebar.header("Runtime paths")
st.sidebar.write(f"Weights: {weights_path}")
st.sidebar.write(f"Retrieval data: {retrieval_path}")
st.sidebar.write(f"Graph data: {graph_path}")

uploaded = st.file_uploader("Upload a chest X-ray", type=["png", "jpg", "jpeg"])

if uploaded is None:
    st.info("Upload an image to run the end-to-end workflow.")
    st.stop()

workspace_image = Path("uploaded_xray.png")
workspace_image.write_bytes(uploaded.getbuffer())

@st.cache_resource
def load_model(weight_file: str) -> RealCNNModel:
    return RealCNNModel(weights_path=weight_file)


model = load_model(str(weights_path))
predictions = model.predict(str(workspace_image))
regions = generate_gradcam_regions(model, str(workspace_image), predictions)
evidence = retrieve_evidence(predictions, regions, corpus_path=str(retrieval_path))
graph_relations = query_graph(predictions, graph_path=str(graph_path))
report = build_report(
    study_id=uploaded.name,
    predictions=predictions,
    regions=regions,
    evidence=evidence,
    graph_relations=graph_relations,
)

left, right = st.columns(2)

with left:
    st.subheader("Predictions")
    st.dataframe(
        [
            {
                "label": item.label,
                "probability": item.probability,
                "threshold": item.threshold,
                "selected": item.selected,
            }
            for item in predictions
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Grad-CAM Regions")
    st.dataframe(
        [region.model_dump() for region in regions],
        use_container_width=True,
        hide_index=True,
    )

    st.caption("Each row describes one selected finding and the overlay image saved for that region.")

    st.subheader("Grad-CAM Visualizations")
    for index, region in enumerate(regions, start=1):
        st.markdown(f"#### {index}. {region.label}")
        st.write(region.description)
        if region.laterality or region.lung_zone:
            location_parts = [part for part in [region.laterality, region.lung_zone] if part]
            st.caption(f"Location: {', '.join(location_parts)}")

        heatmap_path = Path(region.heatmap_path) if region.heatmap_path else None
        if heatmap_path and heatmap_path.exists():
            st.image(str(heatmap_path), caption=f"Grad-CAM overlay for {region.label}", use_container_width=True)
        else:
            st.info("No overlay image was found for this region. The metadata table above still captures the explanation.")

with right:
    st.subheader("Clinician Report")
    st.text(report.clinician_report)

    st.subheader("Patient Summary")
    st.write(report.patient_summary)

st.subheader("Retrieved Evidence")
for item in evidence:
    st.markdown(f"**{item.source}** - {item.title}")
    st.write(item.snippet)
    if item.url:
        st.link_button("Open source", item.url)

st.subheader("GraphRAG Relations")
for relation in graph_relations:
    st.write(f"{relation.source} {relation.relation} {relation.target}: {relation.explanation}")

st.subheader("Audit Metadata")
st.json(report.metadata)

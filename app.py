from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src import explainability
from src.graph_rag import query_graph
from src.model_stub import RealCNNModel
from src.reporting import build_report
from src.retrieval import retrieve_evidence


st.set_page_config(page_title="Chest X-ray Decision Support", layout="wide")

st.title("Chest X-ray Decision Support Prototype")
st.caption("The CNN now loads the uploaded PyTorch weights and the remaining stages read from real data files.")

st.markdown(
    """
    <style>
    .finding-card {
        border: 1px solid #cdd6e3;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.75rem;
        background: linear-gradient(140deg, #f8fbff 0%, #edf4ff 100%);
        box-shadow: 0 2px 8px rgba(40, 63, 95, 0.08);
    }
    .finding-title {
        color: #111827;
        font-weight: 700;
        font-size: 1.05rem;
        line-height: 1.3;
    }
    .finding-chip {
        display: inline-block;
        margin-top: 0.4rem;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .chip-positive {
        color: #0a5d2b;
        background: #d9f7e7;
    }
    .chip-negative {
        color: #6c7280;
        background: #eef1f6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

weights_path = Path(os.getenv("MODEL_WEIGHTS_PATH", "weights/best_model.pth"))
retrieval_path = Path(os.getenv("RETRIEVAL_DATA_PATH", "data/retrieval"))
graph_path = Path(os.getenv("GRAPH_DATA_PATH", "data/graph/relations.json"))

with st.sidebar.expander("Runtime paths", expanded=False):
    st.write(f"Weights: {weights_path}")
    st.write(f"Retrieval data: {retrieval_path}")
    st.write(f"Graph data: {graph_path}")

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
regions = explainability.generate_gradcam_regions(model, str(workspace_image), predictions)
build_gradcam_montage = getattr(explainability, "build_gradcam_montage", None)
montage_candidate = build_gradcam_montage(str(workspace_image), regions) if callable(build_gradcam_montage) else None
montage_path = montage_candidate if isinstance(montage_candidate, Path) else None
evidence = retrieve_evidence(predictions, regions, corpus_path=str(retrieval_path))
graph_relations = query_graph(predictions, graph_path=str(graph_path))
report = build_report(
    study_id=uploaded.name,
    predictions=predictions,
    regions=regions,
    evidence=evidence,
    graph_relations=graph_relations,
)

selected_findings = [item for item in predictions if item.selected and item.label != "No finding"]

summary_tab, gradcam_tab, reports_tab, details_tab = st.tabs(
    ["Summary", "Grad-CAM", "Reports", "Evidence & Graph"]
)

with summary_tab:
    st.subheader("Findings at a Glance")

    selected_col, table_col = st.columns([1.25, 1])

    with selected_col:
        if selected_findings:
            st.caption("Detected findings")
            for finding in selected_findings:
                st.markdown(
                    (
                        f"<div class='finding-card'>"
                        f"<span class='finding-title'>{finding.label}</span><br/>"
                        f"<span class='finding-chip chip-positive'>Detected</span>"
                        f"</div>"
                    ),
                    unsafe_allow_html=True,
                )
        else:
            st.info("No dominant finding selected for this study.")

    with table_col:
        st.caption("All labels")
        st.dataframe(
            [
                {
                    "label": item.label,
                    "probability": item.probability,
                    "threshold": item.threshold,
                    "status": "Detected" if item.selected and item.label != "No finding" else "Not selected",
                }
                for item in predictions
                if item.label != "No finding"
            ],
            use_container_width=True,
            hide_index=True,
        )

with gradcam_tab:
    st.subheader("Paper-style Grad-CAM Montage")
    if isinstance(montage_path, Path) and montage_path.exists():
        st.image(str(montage_path), caption="Combined figure with rows (A), (B), (C)", width=900)
    else:
        st.info("No montage was produced for this image.")

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
            st.image(
                str(heatmap_path),
                caption=f"Grad-CAM panel (original, heatmap, overlay) for {region.label}",
                width=720,
            )
        else:
            st.info("No overlay image was found for this region. The metadata table above still captures the explanation.")

with reports_tab:
    st.subheader("Clinician Report")
    st.text(report.clinician_report)

    st.subheader("Patient Summary")
    st.write(report.patient_summary)

with details_tab:
    st.subheader("Retrieved Evidence")
    for item in evidence:
        with st.expander(f"{item.source} - {item.title}"):
            st.write(item.snippet)
            if item.url:
                st.link_button("Open source", item.url)

    st.subheader("GraphRAG Relations")
    for relation in graph_relations:
        st.write(f"{relation.source} {relation.relation} {relation.target}: {relation.explanation}")

    st.subheader("Audit Metadata")
    st.json(report.metadata)

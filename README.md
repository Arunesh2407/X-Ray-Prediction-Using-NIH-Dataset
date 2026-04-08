# Chest X-ray Decision Support Prototype

This project scaffolds the post-CNN pipeline for chest X-ray interpretation.

## What is implemented
- A shared prediction schema for multi-label chest X-ray classification
- A PyTorch CNN loader for `weights/best_model.pth`
- Grad-CAM extraction on the last convolutional block
- File-backed retrieval and GraphRAG loaders
- Report synthesis for clinician and patient outputs
- A Streamlit UI entrypoint

## Current status
The uploaded model weights are a PyTorch `state_dict`. The app now loads them and expects the retrieval and graph inputs to come from local JSON files.

## Deployment note
This repository is a Streamlit application. It is safe to push to GitHub as-is, but Vercel does not run a long-lived Streamlit server directly. If you need a Vercel deployment, you will need to split the app into a separate frontend plus API service. For a direct Python app deployment, Streamlit Community Cloud or a container host is a better fit.

## Data locations
- Model weights: `weights/best_model.pth`
- Retrieval data: `data/retrieval/` or `data/retrieval.json`
- Graph data: `data/graph/relations.json`

## Safe git setup
Before pushing, initialize the repository and publish only tracked files:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/Arunesh2407/X-Ray-Prediction-Using-NIH-Dataset.git
git push -u origin main
```

The `.gitignore` file excludes local environments, generated Grad-CAM outputs, and the uploaded image artifact so they are not committed by accident.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

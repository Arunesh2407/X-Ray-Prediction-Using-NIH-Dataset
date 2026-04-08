# Chest X-ray Decision Support Prototype

This project scaffolds the post-CNN pipeline for chest X-ray interpretation.

## At a glance

This app takes an uploaded chest X-ray, runs the CNN, generates Grad-CAM overlays, pulls supporting evidence, and assembles a clinician-facing report plus a short patient summary.

## What is implemented

- A shared prediction schema for multi-label chest X-ray classification
- A PyTorch CNN loader for `weights/best_model.pth`
- Grad-CAM extraction on the last convolutional block
- File-backed retrieval and GraphRAG loaders
- Report synthesis for clinician and patient outputs
- A Streamlit UI entrypoint

## Workflow

1. Upload a chest X-ray image.
2. Review the prediction table and the Grad-CAM overlays.
3. Read the retrieved evidence, graph relations, and generated report.
4. Use the outputs as a decision-support aid, not as a final diagnosis.

## Current status

The uploaded model weights are a PyTorch `state_dict`. The app now loads them and expects the retrieval and graph inputs to come from local JSON files.

## Deployment

This repository is a Streamlit application. The recommended host is Streamlit Community Cloud.

### Streamlit Community Cloud (recommended)

1. Push this repository to GitHub (the project already includes `app.py` at the root and a `requirements.txt`).
2. Go to Streamlit Community Cloud and click **Create app**.
3. Select this GitHub repository and branch.
4. Set **Main file path** to `app.py`.
5. Deploy.

If the app restarts or sleeps after inactivity, open it again and wait for the warm-up. That is normal on the free tier.

### Why not Vercel for this codebase

Vercel does not run a long-lived Streamlit server directly. To use Vercel, you would need to re-architect into a separate frontend and API.

## Data locations

- Model weights: `weights/best_model.pth`
- Retrieval data: `data/retrieval/` or `data/retrieval.json`
- Graph data: `data/graph/relations.json`
- Business config: `config/business_config.json`

## Configuration

Business constants are now loaded from `config/business_config.json` (labels, thresholds, report messages, Grad-CAM settings, and UI options). You can tune behavior by editing this file without changing Python code.

## LLM report generation

The clinician and patient reports can be generated through Groq when `GROQ_API_KEY` is set in the environment. If the key is missing or the request fails, the app falls back to the deterministic template report.

The app also checks a local `.env` file and `.streamlit/secrets.toml`, so you can keep the key out of the terminal session if you prefer.

Optional environment variables:

- `GROQ_API_KEY`: your Groq API key
- `GROQ_MODEL`: model name to use, default `llama-3.3-70b-versatile`
- `GROQ_TEMPERATURE`: sampling temperature, default `0.2`
- `GROQ_TIMEOUT_SECONDS`: request timeout, default `30`

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

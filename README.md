# AI for Oil & Gas: Production Forecasting POC

This repo contains a small, end-to-end Proof of Concept to demonstrate how AI can create value in oil & gas through production forecasting and insight generation. It is designed to be run live in a session.

## What this shows
- Download monthly production data for multiple fields from public sources (configurable URL, with a synthetic fallback).
- Exploratory data analysis (seasonality, decline, missing data).
- Baseline forecasting using SARIMAX with walk-forward backtesting.
- Simple explainability and decision support: feature attribution (where applicable), forecast vs. actual plots, and actionable levers.

## Quickstart

1. Install dependencies (in a virtual environment is recommended):
```bash
pip install -r requirements.txt
```

2. Launch Jupyter:
```bash
jupyter lab
```

3. Open `notebooks/production_forecasting.ipynb` and run all cells.

## Data sources
- Primary: configurable HTTP CSV endpoint for monthly production by field. You can drop in a CSV with columns like: `date, field, oil, gas, water`.
- Fallback: the notebook can synthesize a realistic decline curve dataset so the demo always runs.

## Talking points for the session
- AI value levers in Integrated Asset Management:
  - Production forecasting for lift planning and deferment minimization.
  - Opportunity ranking (well workovers, choke optimization) using what-if scenarios.
  - Reliability and predictive maintenance for critical rotating equipment.
  - HSE analytics and anomaly detection.
- Execution considerations:
  - Data plumbing and quality governance are the long poles.
  - Start with narrow, high-signal POCs; build trust via backtesting and operational KPIs.
  - Human-in-the-loop and explainability are key for adoption.

## Repository layout
- `requirements.txt` – Python dependencies
- `notebooks/production_forecasting.ipynb` – All-in-one demo notebook

## Notes
- This POC is intentionally lightweight, focused on clarity and speed to value.

## Anomaly detection POC (asset integrity)

Use this when you want to showcase asset integrity monitoring with sensor anomalies.

How to run:
- Ensure the environment is set up as above.
- Optionally set `SENSOR_DATA_URL` to a CSV with columns like: `timestamp, equipment_id, pressure, temperature, vibration` (more sensors welcome). Hourly data recommended.
- Launch Jupyter and open `notebooks/anomaly_detection_asset_integrity.ipynb`.

What it does:
- Downloads CSV or synthesizes realistic data for multiple equipment IDs with injected faults (spikes, drifts, stuck sensors).
- EDA plots per sensor and equipment.
- Feature engineering (rolling means, z-scores, diffs).
- Two detectors:
  - STL residual z-scores per sensor (seasonality-robust univariate anomalies)
  - IsolationForest on multivariate feature space
- Flags anomalies and visualizes them; produces a summary table with anomaly counts and rates.

## Interactive anomaly dashboard (Streamlit)

Run locally:
- python -m venv .venv && source .venv/bin/activate
- pip install -r requirements.txt
- Optional: export SENSOR_DATA_URL="<https CSV with columns: timestamp, equipment_id, pressure, temperature, vibration>"
- streamlit run app/streamlit_app.py

Features:
- URL upload or file upload, with synthetic fallback ensuring a live demo.
- Controls: resample frequency, STL period, IsolationForest contamination.
- Filters: equipment and sensors.
- Plots: interactive time series with anomaly markers.
- Tables: anomaly summary and event list, downloadable CSV.

## Installation options

- Dashboard only (fast path, Python 3.13 OK):
  - python -m venv .venv && source .venv/bin/activate
  - pip install --upgrade pip setuptools wheel
  - pip install -r requirements-dashboard.txt
  - streamlit run app/streamlit_app.py

- Full notebooks (recommend Python 3.11 or 3.12 for smooth wheels):
  - python3.11 -m venv .venv && source .venv/bin/activate
  - pip install --upgrade pip setuptools wheel
  - pip install -r requirements.txt
  - jupyter lab

Troubleshooting (macOS):
- If a package tries to compile freetype or similar, prefer Python 3.11/3.12, or `brew install freetype pkg-config`.
- If pmdarima wheel fails, use the dashboard-only requirements or install `cmdstanpy` alternatives, or skip pmdarima (not needed for anomaly dashboard).

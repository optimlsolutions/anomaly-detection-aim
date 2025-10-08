import io
import math
import os
import sys
import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

@dataclass
class Cfg:
	time_col: str = "timestamp"
	id_col: str = "equipment_id"
	sensors: List[str] = None
	freq: str = "H"

CFG = Cfg(sensors=["pressure", "temperature", "vibration"])  # can be overridden from UI

# ---------- Data utilities ----------

def try_download_csv(url: str, timeout: int = 30) -> Optional[pd.DataFrame]:
	if not url:
		return None
	try:
		r = requests.get(url, timeout=timeout)
		r.raise_for_status()
		return pd.read_csv(io.StringIO(r.text))
	except Exception:
		return None


def synthesize(equipments=3, start="2024-01-01", periods=24*120, freq=CFG.freq, seed=7) -> pd.DataFrame:
	rng = np.random.default_rng(seed)
	ts = pd.date_range(start=start, periods=periods, freq=freq)
	rows = []
	for i in range(equipments):
		equip = f"EQ{i+1:02d}"
		base_p = rng.uniform(60, 120)
		base_t = rng.uniform(50, 90)
		base_v = rng.uniform(1.0, 3.0)
		drift_t = rng.uniform(-0.002, 0.003)
		for t_idx, t in enumerate(ts):
			cyc = 1 + 0.05 * math.sin(2*math.pi*(t_idx%24)/24.0)
			pressure = base_p * cyc + rng.normal(0, 0.8)
			temperature = base_t + drift_t*t_idx + 0.5*math.sin(2*math.pi*(t_idx%168)/168.0) + rng.normal(0, 0.5)
			vibration = base_v + 0.2*math.sin(2*math.pi*(t_idx%24)/24.0) + rng.normal(0, 0.1)
			rows.append({CFG.time_col: t, CFG.id_col: equip, "pressure": pressure, "temperature": temperature, "vibration": vibration})
		# anomalies
		inj_idx = rng.choice(len(ts), size=6, replace=False)
		for j in inj_idx:
			rows[j + i*len(ts)]["pressure"] += rng.uniform(10, 25)
		start_d = rng.integers(24, len(ts)-72)
		for j in range(start_d, start_d+48):
			rows[j + i*len(ts)]["temperature"] += (j-start_d) * 0.15
		stuck_start = rng.integers(24, len(ts)-48)
		stuck_val = rows[stuck_start + i*len(ts)]["vibration"]
		for j in range(stuck_start, stuck_start+36):
			rows[j + i*len(ts)]["vibration"] = stuck_val
	return pd.DataFrame(rows)


def normalize_and_coerce(df: pd.DataFrame) -> pd.DataFrame:
	df = df.rename(columns={c: c.strip().lower() for c in df.columns})
	if CFG.time_col in df.columns:
		df[CFG.time_col] = pd.to_datetime(df[CFG.time_col])
		df = df.sort_values(CFG.time_col)
	if CFG.id_col in df.columns:
		df[CFG.id_col] = df[CFG.id_col].astype(str)
	for s in CFG.sensors:
		if s in df.columns:
			df[s] = pd.to_numeric(df[s], errors="coerce")
	df = df.dropna(subset=[CFG.time_col])
	return df


def add_features(df: pd.DataFrame, freq: str) -> pd.DataFrame:
	df = df.copy()
	df = df.set_index(CFG.time_col)
	parts = []
	for eq, g in df.groupby(CFG.id_col):
		g = g.asfreq(freq).interpolate(limit_direction="both")
		for s in CFG.sensors:
			if s in g.columns:
				g[f"{s}_diff1"] = g[s].diff(1)
				g[f"{s}_z"] = (g[s] - g[s].rolling(24, min_periods=6).mean()) / (g[s].rolling(24, min_periods=6).std() + 1e-6)
				g[f"{s}_ma12"] = g[s].rolling(12).mean()
				g[f"{s}_ma24"] = g[s].rolling(24).mean()
		g[CFG.id_col] = eq
		parts.append(g)
	return pd.concat(parts).reset_index()


def stl_residual_scores(g: pd.DataFrame, column: str, period: int) -> pd.Series:
	x = g[column].asfreq(CFG.freq).interpolate(limit_direction="both")
	if x.isna().all():
		return pd.Series(index=g.index, dtype=float)
	try:
		stl = STL(x, period=period)
		res = stl.fit()
		resid = res.resid
		z = (resid - resid.rolling(2*period, min_periods=period//2).mean()) / (resid.rolling(2*period, min_periods=period//2).std() + 1e-6)
		return z.reindex(g.index)
	except Exception:
		return pd.Series(index=g.index, dtype=float)

# ---------- UI ----------

st.set_page_config(page_title="Asset Integrity: Anomaly Detection", layout="wide")
st.title("Asset Integrity: Anomaly Detection Dashboard")

with st.sidebar:
	st.header("Data")
	url = st.text_input("CSV URL (optional)", value=os.environ.get("SENSOR_DATA_URL", ""))
	uploaded = st.file_uploader("Or upload CSV", type=["csv"])  # expects columns timestamp, equipment_id, sensors
	st.caption("Columns: timestamp, equipment_id, pressure, temperature, vibration")

	st.header("Configuration")
	freq = st.selectbox("Resample frequency", options=["H","30min","15min","D"], index=0)
	stl_period = st.number_input("STL period (samples)", min_value=6, max_value=240, value=24, step=6)
	contam = st.slider("IsolationForest contamination", min_value=0.001, max_value=0.1, value=0.02, step=0.001)
	seed = st.number_input("Random seed (synthetic)", min_value=0, max_value=10_000, value=7, step=1)
	synth_equip = st.slider("Synthetic equipments", 1, 10, 3)
	period_hours = st.slider("Synthetic hours", 24*24, 24*180, 24*120, step=24)

# Load data
if uploaded is not None:
	df = pd.read_csv(uploaded)
elif url:
	df = try_download_csv(url) or synthesize(equipments=synth_equip, periods=period_hours, seed=seed)
else:
	df = synthesize(equipments=synth_equip, periods=period_hours, seed=seed)

df = normalize_and_coerce(df)

# Sensors present
sensors_present = [s for s in CFG.sensors if s in df.columns]
if not sensors_present:
	st.error("No expected sensor columns found. Include at least one of: pressure, temperature, vibration.")
	st.stop()

# Filters
left, right = st.columns([2,1])
with left:
	st.subheader("Raw preview")
	st.dataframe(df.head(20), use_container_width=True)
with right:
	equipments = sorted(df[CFG.id_col].unique())
	selected_eq = st.multiselect("Equipment filter", options=equipments, default=equipments[: min(5, len(equipments))])
	selected_sensors = st.multiselect("Sensors", options=sensors_present, default=sensors_present)

if selected_eq:
	df = df[df[CFG.id_col].isin(selected_eq)]

# Feature engineering
feat = add_features(df, freq=freq)

# STL residual scores
score_parts = []
for eq, g in feat.set_index(CFG.time_col).groupby(CFG.id_col):
	g = g.sort_index()
	sc = pd.DataFrame(index=g.index)
	for s in selected_sensors:
		sc[f"{s}_stl_z"] = stl_residual_scores(g, s, period=int(stl_period))
	sc[CFG.id_col] = eq
	score_parts.append(sc)
score_df = pd.concat(score_parts).reset_index()

# IsolationForest
features = []
for s in selected_sensors:
	features += [f"{s}", f"{s}_diff1", f"{s}_z", f"{s}_ma12", f"{s}_ma24"]

merged = pd.merge(feat, score_df, on=[CFG.time_col, CFG.id_col], how="left")

anom_rows = []
for eq, g in merged.groupby(CFG.id_col):
	X = g[features].copy()
	X = X.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill").fillna(0)
	scaler = StandardScaler()
	Xs = scaler.fit_transform(X)
	iso = IsolationForest(n_estimators=300, contamination=float(contam), random_state=42)
	iso.fit(Xs)
	scores = -iso.decision_function(Xs)
	preds = iso.predict(Xs)
	gg = g[[CFG.time_col, CFG.id_col] + selected_sensors].copy()
	gg["if_score"] = scores
	gg["if_flag"] = (preds == -1)
	stl_cols = [c for c in g.columns if c.endswith("_stl_z")]
	if stl_cols:
		gg["stl_flag"] = g[stl_cols].abs().max(axis=1) > 3
	else:
		gg["stl_flag"] = False
	gg["anomaly"] = gg["if_flag"] | gg["stl_flag"]
	anom_rows.append(gg)

final = pd.concat(anom_rows, ignore_index=True).sort_values(CFG.time_col)

# ---------- Charts ----------

st.subheader("Sensor traces with anomalies")
for eq, g in final.groupby(CFG.id_col):
	for s in selected_sensors:
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=g[CFG.time_col], y=g[s], name=s, mode="lines"))
		mask = g["anomaly"] == True
		fig.add_trace(go.Scatter(x=g.loc[mask, CFG.time_col], y=g.loc[mask, s], name="anomaly", mode="markers", marker=dict(color="red", size=7)))
		fig.update_layout(title=f"{eq} - {s}", height=350, margin=dict(l=10,r=10,t=40,b=10))
		st.plotly_chart(fig, use_container_width=True)

st.subheader("Anomaly summary")
sum_table = (final.groupby(CFG.id_col)["anomaly"].agg(["sum","mean"]).rename(columns={"sum":"anomaly_count","mean":"anomaly_rate"}).reset_index())
st.dataframe(sum_table, use_container_width=True)

st.subheader("Anomaly events")
st.dataframe(final[final["anomaly"]==True].sort_values([CFG.id_col, CFG.time_col]).reset_index(drop=True), use_container_width=True)

# Download
csv = final.to_csv(index=False).encode("utf-8")
st.download_button("Download scored dataset (CSV)", data=csv, file_name="anomaly_scored.csv", mime="text/csv")

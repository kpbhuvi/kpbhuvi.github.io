"""
AI Predictive Maintenance Dashboard
-------------------------------------
Turns raw equipment sensor telemetry into a fleet-wide risk ranking: "which
3 machines should the technician check this week?" instead of a wall of
charts. Grounded in real field-service and energy-forecasting product work
(Cintas Corp field service mobile, Xcel Energy demand forecasting — see
case studies on aiwithbhuvi.blog).

V1 deliberately uses a statistical/Isolation-Forest anomaly baseline instead
of an LSTM forecaster: it needs far less historical data to be useful, it's
easier to explain to a non-technical field ops manager, and it establishes
the "does this even help" signal before justifying the cost of a deep model
— the same sequencing used on the Xcel Energy engagement.

Drop this file into the `pages/` folder of the existing bhuvi-ai-lab
Streamlit multipage app.

Author: Bhuvaneswari Kuduva Premkumar
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="AI Predictive Maintenance Dashboard", page_icon="🛠️", layout="wide")

st.title("🛠️ AI Predictive Maintenance Dashboard")
st.caption(
    "From raw sensor telemetry to a ranked list of which assets need a look this week — "
    "with an explainable anomaly score, not a black box."
)

# ---------------------------------------------------------------------------
# Synthetic fleet telemetry generator (stands in for real IoT/SCADA export)
# ---------------------------------------------------------------------------

ASSETS = ["Compressor-A1", "Compressor-A2", "Chiller-B1", "Pump-C3", "Motor-D2", "Generator-E1"]


@st.cache_data
def generate_fleet_data(days=60, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="D")
    for asset in ASSETS:
        base_temp = rng.uniform(60, 75)
        base_vib = rng.uniform(0.2, 0.5)
        runtime_hours = rng.uniform(3000, 9000)
        degrade = rng.choice([0, 1], p=[0.6, 0.4])  # some assets are drifting toward failure
        for i, d in enumerate(dates):
            drift = (i / days) * rng.uniform(8, 18) if degrade else rng.uniform(-1, 1)
            temp = base_temp + drift + rng.normal(0, 1.5)
            vib = base_vib + (drift / 20) + rng.normal(0, 0.05)
            runtime_hours += rng.uniform(6, 18)
            rows.append({
                "date": d, "asset": asset,
                "temperature_f": round(temp, 2),
                "vibration_g": round(max(vib, 0), 3),
                "runtime_hours": round(runtime_hours, 1),
            })
    return pd.DataFrame(rows)


with st.sidebar:
    st.header("Data Source")
    source = st.radio("Choose input", ["Use synthetic fleet data", "Upload CSV"], index=0)
    st.caption("CSV needs columns: `date`, `asset`, `temperature_f`, `vibration_g`, `runtime_hours`")
    uploaded = None
    if source == "Upload CSV":
        uploaded = st.file_uploader("Upload sensor CSV", type=["csv"])
    days_back = st.slider("Days of history (synthetic data only)", 30, 120, 60, step=10)

if source == "Upload CSV" and uploaded is not None:
    df = pd.read_csv(uploaded, parse_dates=["date"])
else:
    df = generate_fleet_data(days=days_back)
    if source == "Upload CSV":
        st.info("No file uploaded yet — showing synthetic fleet telemetry below.")

# ---------------------------------------------------------------------------
# Anomaly scoring per asset (Isolation Forest on rolling sensor features)
# ---------------------------------------------------------------------------

def score_asset(asset_df: pd.DataFrame) -> pd.DataFrame:
    asset_df = asset_df.sort_values("date").copy()
    features = asset_df[["temperature_f", "vibration_g"]].copy()
    features["temp_roll_mean"] = features["temperature_f"].rolling(7, min_periods=1).mean()
    features["vib_roll_mean"] = features["vibration_g"].rolling(7, min_periods=1).mean()
    model = IsolationForest(n_estimators=150, contamination=0.15, random_state=42)
    features_filled = features.bfill().ffill()
    raw_scores = model.fit_predict(features_filled)
    # decision_function: higher = more normal. Flip & normalize to a 0-100 "risk score".
    decision = model.decision_function(features_filled)
    risk = (decision.max() - decision) / (decision.max() - decision.min() + 1e-9) * 100
    asset_df["anomaly_flag"] = raw_scores == -1
    asset_df["risk_score"] = risk.round(1)
    return asset_df


scored = df.groupby("asset", group_keys=False).apply(score_asset)

def risk_tier(row):
    if row["risk_score"] >= 70 or row["runtime_hours"] > 8000:
        return "High"
    if row["risk_score"] >= 40:
        return "Medium"
    return "Low"

scored["risk_tier"] = scored.apply(risk_tier, axis=1)

latest = scored.sort_values("date").groupby("asset").tail(1).sort_values("risk_score", ascending=False)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

tier_color = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#00c2a8"}

c1, c2, c3, c4 = st.columns(4)
c1.metric("Assets monitored", len(ASSETS))
c2.metric("High risk", int((latest["risk_tier"] == "High").sum()))
c3.metric("Medium risk", int((latest["risk_tier"] == "Medium").sum()))
c4.metric("Avg. fleet risk score", f"{latest['risk_score'].mean():.0f} / 100")

st.divider()
st.subheader("Fleet risk ranking — which asset needs a look first?")
display = latest[["asset", "risk_tier", "risk_score", "temperature_f", "vibration_g", "runtime_hours"]].reset_index(drop=True)
st.dataframe(
    display.style.apply(
        lambda row: [f"background-color: {tier_color[row['risk_tier']]}22"] * len(row), axis=1
    ),
    use_container_width=True,
)

st.divider()
st.subheader("Drill down: sensor trend + anomaly flags for one asset")
pick = st.selectbox("Choose an asset", ASSETS, index=0)
asset_hist = scored[scored["asset"] == pick].sort_values("date")

fig = go.Figure()
fig.add_trace(go.Scatter(x=asset_hist["date"], y=asset_hist["temperature_f"], name="Temperature (°F)", line=dict(color="#1a4fff")))
fig.add_trace(go.Scatter(x=asset_hist["date"], y=asset_hist["vibration_g"] * 100, name="Vibration (g ×100)", line=dict(color="#00c2a8")))
flagged = asset_hist[asset_hist["anomaly_flag"]]
fig.add_trace(go.Scatter(
    x=flagged["date"], y=flagged["temperature_f"], mode="markers",
    name="Anomaly flagged", marker=dict(color="#ef4444", size=9, symbol="x"),
))
fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=380, legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig, use_container_width=True)

st.download_button(
    "⬇ Download fleet risk report (CSV)",
    display.to_csv(index=False),
    file_name="fleet_risk_report.csv",
    mime="text/csv",
)

st.divider()
st.caption(
    "Built by Bhuvaneswari Kuduva Premkumar · V1 uses Isolation Forest on rolling sensor "
    "features — explainable and data-light. LSTM-based demand/failure forecasting is the "
    "planned V2 upgrade once enough historical failure-labeled data is collected."
)

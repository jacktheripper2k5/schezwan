# =============================
# STREAMLIT APP: HIGH-END UI (CARBON & COPPER THEME)
# =============================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Copper Loss Dashboard", layout="wide")

# -----------------------------
# CUSTOM DARK UI (GLASS + GLOW)
# -----------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: radial-gradient(circle at 20% 20%, #1c1c1c, #0f0f0f 60%);
    color: #eaeaea;
    font-family: 'Segoe UI', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(20,20,20,0.9);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255,140,0,0.2);
}

/* Glass Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    padding: 25px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,140,0,0.3);
    box-shadow: 0 0 20px rgba(255,140,0,0.15);
}

/* Copper Glow Metric */
.metric-glow {
    font-size: 60px;
    font-weight: 700;
    color: #ff8c00;
    text-align: center;
    text-shadow: 0 0 10px rgba(255,140,0,0.7),
                 0 0 20px rgba(255,140,0,0.4);
}

/* Titles */
h1, h2, h3 {
    color: #ffb347;
}

/* Sliders */
.stSlider > div > div {
    color: #ff8c00;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("copper_loss_synthetic_dataset.csv")


df = load_data()

# -----------------------------
# MODEL
# -----------------------------
X = df.drop("Copper Loss in Slag (% Cu)", axis=1)
y = df["Copper Loss in Slag (% Cu)"]

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X, y)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("⚙️ Process Controls")

matte_grade = st.sidebar.slider("Matte Grade (% Cu)", 20.0, 80.0, 50.0)
slag_temp = st.sidebar.slider("Slag Temperature (°C)", 1150.0, 1300.0, 1200.0)
silica_flux = st.sidebar.slider("Silica Flux (% SiO2)", 20.0, 45.0, 30.0)
settling_time = st.sidebar.slider("Settling Time (min)", 10.0, 90.0, 40.0)

# -----------------------------
# PREDICTION
# -----------------------------
input_data = pd.DataFrame({
    "Matte Grade (% Cu in Matte)": [matte_grade],
    "Slag Temperature (°C)": [slag_temp],
    "Silica Flux Added (% SiO2 in Slag)": [silica_flux],
    "Settling Time (Minutes)": [settling_time]
})

prediction = model.predict(input_data)[0]

# -----------------------------
# HEADER
# -----------------------------
st.title("🔥 Copper Loss Intelligence Dashboard")

st.markdown("""
Advanced metallurgical prediction system combining **thermodynamics + machine learning**.

Copper loss occurs due to:
- Chemical dissolution into slag
- Mechanical entrainment of matte droplets
""")

# -----------------------------
# MAIN METRIC (CENTER)
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### Predicted Copper Loss")
st.markdown(f'<div class="metric-glow">{prediction:.3f} %</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Feature Influence")

importance = model.feature_importances_
features = X.columns

fig, ax = plt.subplots()
ax.barh(features, importance)
ax.set_facecolor('#111111')
fig.patch.set_facecolor('#111111')

ax.set_title("Drivers of Copper Loss", color='white')
ax.tick_params(colors='white')
ax.xaxis.label.set_color('white')

st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# DATA PREVIEW
# -----------------------------
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.subheader("Dataset Snapshot")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("""
---
High-performance industrial ML dashboard • Carbon & Copper Theme
""")

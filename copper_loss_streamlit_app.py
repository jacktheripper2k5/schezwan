# =============================
# STREAMLIT APP: COPPER LOSS PREDICTION
# =============================

# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(page_title="Copper Loss Predictor", layout="wide")

# Custom Yellow Theme Styling
st.markdown("""
    <style>
    .main {
        background-color: #fff9db;
    }
    h1, h2, h3 {
        color: #b08900;
    }
    .stSidebar {
        background-color: #fff3bf;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------
# Make sure the CSV is in the same directory or provide full path
@st.cache_data
def load_data():
    return pd.read_csv("copper_loss_synthetic_dataset.csv")

df = load_data()

# -----------------------------
# TRAIN MODEL
# -----------------------------
X = df.drop("Copper Loss in Slag (% Cu)", axis=1)
y = df["Copper Loss in Slag (% Cu)"]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# -----------------------------
# TITLE & DESCRIPTION
# -----------------------------
st.title("🔥 Copper Loss in Slag Predictor")

st.markdown("""
This interactive tool predicts **Copper Loss in Slag (%)** during matte smelting operations.

### Metallurgical Insight:
Copper is lost through two main mechanisms:
- **Chemical Dissolution**: Copper dissolves into slag at high matte grades and oxidizing conditions.
- **Mechanical Entrainment**: Matte droplets get trapped in viscous slag, especially when silica is low.

Use the sliders to simulate real plant conditions and observe how losses change.
""")

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("⚙️ Process Parameters")

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

st.subheader("📊 Predicted Copper Loss")
st.metric(label="Copper Loss (%)", value=f"{prediction:.3f}")

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.subheader("📈 Feature Importance Analysis")

importance = model.feature_importances_
features = X.columns

fig, ax = plt.subplots()
ax.barh(features, importance)
ax.set_xlabel("Importance")
ax.set_title("Which Parameters Drive Copper Loss?")

st.pyplot(fig)

# -----------------------------
# DATA PREVIEW
# -----------------------------
st.subheader("📂 Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("""
---
Built for metallurgical process understanding using Machine Learning.
""")

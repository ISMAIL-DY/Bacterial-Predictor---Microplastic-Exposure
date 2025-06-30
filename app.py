
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

# --- Paths ---
MODEL_PATH = "rf_model_cetobacterium.pkl"
FEATURES_PATH = "rf_model_features.pkl"

# --- Load model and features ---
if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    st.error("❌ Model or feature file is missing. Please upload them.")
    st.stop()

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)
explainer = shap.TreeExplainer(model)

# --- Streamlit UI ---
st.title("🧬 Cetobacterium Predictor - Microplastic Exposure")
st.markdown("Predict the presence of *Cetobacterium* based on microplastic exposure conditions.")

st.sidebar.header("🧪 Exposure Inputs")
mp_type_pe = st.sidebar.selectbox("MP Type: PE", [0, 1])
mp_type_ps = st.sidebar.selectbox("MP Type: PS", [0, 1])
mp_conc = st.sidebar.slider("MP Concentration (µg/mL)", 0, 2000, 1000)
mp_size = st.sidebar.slider("MP Size (µm)", 0, 1000, 300)
exposure_time = st.sidebar.slider("Exposure Time (days)", 1, 30, 14)

# --- Create full-length input vector ---
input_df = pd.DataFrame(columns=feature_names)
input_df.loc[0] = 0  # All features set to 0

# --- Fill in values from UI
user_inputs = {
    "MP_Concentration": mp_conc,
    "MP_Size": mp_size,
    "Exposure_Time": exposure_time,
    "MP_Type_PE": mp_type_pe,
    "MP_Type_PS": mp_type_ps
}

# Safely update any matching fields
for key, value in user_inputs.items():
    if key in input_df.columns:
        input_df.loc[0, key] = value

# --- Prediction ---
pred = model.predict(input_df)[0]
pred_label = "✅ Present" if pred == 1 else "❌ Absent"
st.subheader(f"Prediction: *Cetobacterium* is **{pred_label}**")

# --- SHAP Explanation ---
st.subheader("🔍 SHAP Explanation")
shap_values = explainer.shap_values(input_df)

# Handle SHAP output shape
if isinstance(shap_values, list):
    shap_vector = shap_values[1][0]
else:
    shap_vector = shap_values[0]

shap_1d = shap_vector.flatten()
features = input_df.columns.tolist()

if len(shap_1d) != len(features):
    st.error(f"❌ SHAP vector length ({len(shap_1d)}) does not match input features ({len(features)}).")
    st.stop()

shap_df = pd.DataFrame({"Feature": features, "SHAP Value": shap_1d})
shap_df = shap_df.sort_values(by="SHAP Value", key=abs, ascending=False)

top_shap = shap_df.head(10)
st.subheader("📊 Top SHAP Contributions to Prediction")
st.dataframe(top_shap)

plt.figure(figsize=(6, 4))
plt.barh(top_shap["Feature"], top_shap["SHAP Value"])
plt.title("Top Features Driving the Prediction")
plt.gca().invert_yaxis()
st.pyplot(plt)

# --- Global Feature Importance ---
st.subheader("📈 Global Feature Importance")
importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
plt.figure(figsize=(6, 5))
importances.tail(10).plot(kind='barh')
plt.title("Most Important Features (Global)")
st.pyplot(plt)

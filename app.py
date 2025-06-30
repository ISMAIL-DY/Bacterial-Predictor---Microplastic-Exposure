
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Load model and feature names ---
MODEL_PATH = "rf_model_cetobacterium.pkl"
FEATURES_PATH = "rf_model_features.pkl"
DATA_PATH = "data/final_selected_features_dataset.csv"

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    st.warning("⚠️ Model not found. Please retrain it.")
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
input_df.loc[0] = 0  # initialize all features to zero

# Update user input values
if 'MP_Concentration' in input_df.columns:
    input_df.at[0, 'MP_Concentration'] = mp_conc
if 'MP_Size' in input_df.columns:
    input_df.at[0, 'MP_Size'] = mp_size
if 'Exposure_Time' in input_df.columns:
    input_df.at[0, 'Exposure_Time'] = exposure_time
if 'MP_Type_PE' in input_df.columns:
    input_df.at[0, 'MP_Type_PE'] = mp_type_pe
if 'MP_Type_PS' in input_df.columns:
    input_df.at[0, 'MP_Type_PS'] = mp_type_ps

# --- Prediction ---
pred = model.predict(input_df)[0]
pred_label = "✅ Present" if pred == 1 else "❌ Absent"
st.subheader(f"Prediction: *Cetobacterium* is **{pred_label}**")

# --- SHAP Explanation ---
st.subheader("🔍 SHAP Explanation")
shap_values = explainer.shap_values(input_df)

# Handle SHAP output structure
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

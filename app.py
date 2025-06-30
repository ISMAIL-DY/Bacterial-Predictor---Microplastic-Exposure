
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib

# --- Load trained model and feature names ---
model = joblib.load("rf_model_cetobacterium.pkl")
feature_names = joblib.load("rf_model_features.pkl")  # List of features used during training
explainer = shap.TreeExplainer(model)

# --- App title ---
st.title("🧬 Cetobacterium Predictor - Microplastic Exposure")
st.markdown("Predict the presence of *Cetobacterium* based on microplastic exposure conditions using a trained ML model.")

# --- Sidebar inputs ---
st.sidebar.header("🧪 Exposure Inputs")
mp_type_pe = st.sidebar.selectbox("MP Type: PE", [0, 1])
mp_type_ps = st.sidebar.selectbox("MP Type: PS", [0, 1])
mp_conc = st.sidebar.slider("MP Concentration (µg/mL)", 0, 2000, 1000)
mp_size = st.sidebar.slider("MP Size (µm)", 0, 1000, 300)
exposure_time = st.sidebar.slider("Exposure Time (days)", 1, 30, 14)

# --- Construct input row ---
user_input = {
    "MP_Concentration": mp_conc,
    "MP_Size": mp_size,
    "Exposure_Time": exposure_time,
    "MP_Type_PE": mp_type_pe,
    "MP_Type_PS": mp_type_ps
}

input_df = pd.DataFrame([user_input])

# --- Reindex to match expected model features ---
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# --- Prediction ---
pred = model.predict(input_df)[0]
pred_label = "✅ Present" if pred == 1 else "❌ Absent"
st.subheader(f"Prediction: *Cetobacterium* is **{pred_label}**")

# SHAP Summary Plot (bar format – most robust)
# Get SHAP values
shap_values = explainer.shap_values(input_df)

# Handle both versions safely
try:
    shap_vector = shap_values[0]  # new SHAP versions
except:
    shap_vector = shap_values[1][0]  # older SHAP fallback

# Build SHAP bar chart
# Flatten column names and SHAP values
features = input_df.columns.tolist()
shap_1d = shap_vector.flatten()

# Now safely build the DataFrame
shap_df = pd.DataFrame({
    "Feature": features,
    "SHAP Value": shap_1d
}).sort_values(by="SHAP Value", key=abs, ascending=False)

top_shap = shap_df.head(10)

st.subheader("🔍 Top SHAP Contributions to This Prediction")
st.dataframe(top_shap)

# Plot
plt.figure(figsize=(6, 4))
plt.barh(top_shap["Feature"], top_shap["SHAP Value"])
plt.title("Top Features Driving the Prediction")
plt.gca().invert_yaxis()
st.pyplot(plt)

# --- Feature importance plot ---
st.subheader("📊 Global Feature Importance")
importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
plt.figure(figsize=(6, 5))
importances.tail(10).plot(kind='barh')
plt.title("Top Features Driving the Prediction")
st.pyplot(plt)

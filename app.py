import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- Paths ---
MODEL_PATH = "rf_model_cetobacterium.pkl"
FEATURES_PATH = "rf_model_features.pkl"
DATA_PATH = "data/final_selected_features_dataset.csv"

# --- Train model if missing ---
if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    st.warning("⚠️ Model not found. Training now...")
    df = pd.read_csv(DATA_PATH)

    mp_type_cols = [col for col in df.columns if col.startswith("MP_Type_")]
    species_cols = [col for col in df.columns if col.startswith("Species_")]
    features = mp_type_cols + ['MP_Concentration', 'MP_Size', 'Exposure_Time'] + species_cols

    X = df[features]
    y = df['Cetobacterium_Present']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(X_train.columns.tolist(), FEATURES_PATH)
else:
    model = joblib.load(MODEL_PATH)

# --- Load features and SHAP ---
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

# --- Input Vector Construction ---
# Step 1: Create a full-zero row with correct feature names
# Step 1: Create a blank input row with all features set to 0
input_data = {feature: 0 for feature in feature_names}

# Step 2: Update only user-controlled fields
input_data["MP_Concentration"] = mp_conc
input_data["MP_Size"] = mp_size
input_data["Exposure_Time"] = exposure_time

# Only update if the model was trained with these
if "MP_Type_PE" in feature_names:
    input_data["MP_Type_PE"] = mp_type_pe
if "MP_Type_PS" in feature_names:
    input_data["MP_Type_PS"] = mp_type_ps

# Step 3: Convert to DataFrame with correct feature order
input_df = pd.DataFrame([input_data])[feature_names]

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

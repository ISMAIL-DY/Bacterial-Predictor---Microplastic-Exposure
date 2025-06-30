import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
import numpy as np

# --- Load trained model and feature names ---
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# === Train model if not yet saved ===
model_path = "rf_model_cetobacterium.pkl"
feature_path = "rf_model_features.pkl"

if not os.path.exists(model_path) or not os.path.exists(feature_path):
    st.warning("Training model from scratch...")

    # Load your CSV dataset
    df = pd.read_csv("data/final_selected_features_dataset.csv")

    # Define features
    mp_type_cols = [col for col in df.columns if col.startswith("MP_Type_")]
    species_cols = [col for col in df.columns if col.startswith("Species_")]
    features = mp_type_cols + ['MP_Concentration', 'MP_Size', 'Exposure_Time'] + species_cols
    X = df[features]
    y = df['Cetobacterium_Present']

    # Train model
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    # Save model and feature list
    joblib.dump(model, model_path)
    joblib.dump(X_train.columns.tolist(), feature_path)

else:
    model = joblib.load(model_path)

# === Load features and SHAP ===
feature_names = joblib.load(feature_path)
explainer = shap.TreeExplainer(model)


# --- App title ---
st.title("🧬 Cetobacterium Predictor - Microplastic Exposure")
st.markdown("Predict the presence of *Cetobacterium* in fish microbiomes based on microplastic exposure conditions.")

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

# Build input DataFrame and reindex to match trained features
input_df = pd.DataFrame([user_input])
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# --- Make prediction ---
pred = model.predict(input_df)[0]
pred_label = "✅ Present" if pred == 1 else "❌ Absent"
st.subheader(f"Prediction: *Cetobacterium* is **{pred_label}**")

# --- SHAP Explanation ---
st.subheader("🔍 SHAP Explanation")
shap_values = explainer.shap_values(input_df)

# Safely extract SHAP vector
if isinstance(shap_values, list):
    shap_vector = shap_values[1][0]  # class 1 explanation
else:
    shap_vector = shap_values[0]     # fallback for newer SHAP versions

# Prepare SHAP bar chart
shap_1d = shap_vector.flatten()
features = input_df.columns.tolist()

if len(shap_1d) != len(features):
    st.error(f"❌ SHAP vector length ({len(shap_1d)}) does not match input features ({len(features)}).")
    st.stop()

shap_df = pd.DataFrame({
    "Feature": features,
    "SHAP Value": shap_1d
}).sort_values(by="SHAP Value", key=abs, ascending=False)

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

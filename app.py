import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, f1_score, accuracy_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split, learning_curve

# --- Paths ---
MODEL_PATHS = {
    "Cetobacterium": "rf_model_cetobacterium.pkl",
    "Rhizobiales": "rf_model_rhizobiales.pkl"
}
FEATURE_PATHS = {
    "Cetobacterium": "rf_model_features.pkl",
    "Rhizobiales": "rf_model_features.pkl"
}
DATA_PATH = "final_selected_features_dataset.csv"
from PIL import Image

# --- Display Logos ---
col1, col2 = st.columns([1, 1])

with col1:
    faculty_logo = Image.open("meakxa38.png")
    st.image(faculty_logo, use_column_width=True, caption="Faculty Of Medecine and Pharmacie Rabat")

with col2:
    master_logo = Image.open("cbbda.jpg")
    st.image(master_logo, use_column_width=True, caption="master of computational biology bioinformatics and data_analysis")

st.image("faculty_logo.png", width=200)
st.image("master_logo.png", width=200)
st.title("🧬 Bacterial Predictor - Microplastic Exposure")

# --- Select target bacterium ---
st.title("🧬 Bacterial Predictor - Microplastic Exposure")
target = st.radio("Select Bacterium to Predict:", ["Cetobacterium", "Rhizobiales"])

# --- Description of Selected Bacterium ---
if target == "Cetobacterium":
    st.info("Cetobacterium* is a key gut microbe in fish, often studied in response to microplastics.")
else:
    st.info("Rhizobiales* are bacteria with environmental relevance and potential as indicators in microbiome shifts.")

# --- Load model and features ---
model_path = MODEL_PATHS[target]
features_path = FEATURE_PATHS[target]

if not os.path.exists(model_path) or not os.path.exists(features_path):
    st.error(f"❌ Missing model or features for {target}")
    st.stop()

model = joblib.load(model_path)
feature_names = joblib.load(features_path)

# --- UI Inputs ---
st.sidebar.header("Exposure Inputs")
mp_conc = st.sidebar.slider("MP Concentration (µg/mL)", 0, 2000, 1000)
mp_size = st.sidebar.slider("MP Size (µm)", 0, 1000, 300)
exposure_time = st.sidebar.slider("Exposure Time (days)", 1, 30, 14)


# --- App Info Sidebar ---
with st.sidebar.expander(" About this app"):
    st.write("""
    This app predicts the presence of *Cetobacterium* or *Rhizobiales* in fish gut microbiota
    based on microplastic exposure conditions.

    -  Predicts binary presence using Random Forest
    -  Includes evaluation plots, batch CSV uploads, and feature analysis
    -  Input features: MP_Concentration, MP_Size, Exposure_Time

    Created by: Mohamed Ismail Drissi Yahyaoui  
    
    """)

# --- Create input vector ---
input_df = pd.DataFrame(columns=feature_names)
input_df.loc[0] = 0
input_df.loc[0, "MP_Concentration"] = mp_conc
input_df.loc[0, "MP_Size"] = mp_size
input_df.loc[0, "Exposure_Time"] = exposure_time

# --- Prediction ---
proba = model.predict_proba(input_df)[0][1]
pred = model.predict(input_df)[0]
pred_label = "✅ Present" if pred == 1 else "❌ Absent"

st.subheader(f"Prediction: *{target}* is **{pred_label}**")
st.metric(label="Probability of Presence", value=f"{proba:.2%}")

# --- Download input row ---
st.download_button(
    label="Download This Input",
    data=input_df.to_csv(index=False),
    file_name=f"{target.lower()}_input.csv",
    mime="text/csv"
)

# --- Top Feature Importances ---
st.subheader("Most Important Features")
importances = pd.Series(model.feature_importances_, index=feature_names)
top_features = importances.sort_values(ascending=False).head(3)
st.write(top_features)

# --- Upload for Batch Prediction ---
st.subheader("Batch Prediction from Uploaded File")
uploaded = st.file_uploader("Upload CSV with same features", type="csv")
if uploaded:
    df_upload = pd.read_csv(uploaded)
    df_upload = df_upload.reindex(columns=feature_names, fill_value=0)
    preds = model.predict(df_upload)
    df_upload["Prediction"] = ["Present" if p == 1 else "Absent" for p in preds]
    st.dataframe(df_upload)
    st.download_button(f"Download {target} Predictions", df_upload.to_csv(index=False), f"{target.lower()}_predictions.csv", "text/csv")

# --- Evaluation Plots ---
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    X = df[feature_names]
    y = df[f"{target}_Present"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    st.subheader("ROC Curve (AUC)")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    st.pyplot(fig)

    # --- PR Curve ---
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    st.subheader("Precision-Recall Curve")
    fig2, ax2 = plt.subplots()
    ax2.plot(recall, precision, color='purple', lw=2)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    st.pyplot(fig2)

    # --- F1 Score ---
    st.subheader("Classification Metrics")
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"**F1 Score:** {f1:.2f}")
    st.write(f"**Accuracy:** {acc:.2f}")
    st.text(classification_report(y_test, y_pred))

    # --- Confusion Matrix ---
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    # --- Learning Curve ---
    st.subheader("Learning Curve")
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring="f1", n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    fig3, ax3 = plt.subplots()
    ax3.plot(train_sizes, train_mean, 'o-', label="Training Score")
    ax3.plot(train_sizes, test_mean, 'o-', label="Cross-Validation Score")
    ax3.set_xlabel("Training Set Size")
    ax3.set_ylabel("F1 Score")
    ax3.set_title("Learning Curve")
    ax3.legend()
    st.pyplot(fig3)

    # --- Feature Importance ---
    st.subheader("Full Feature Importance")
    fig4, ax4 = plt.subplots()
    importances.sort_values().plot(kind="barh", ax=ax4)
    ax4.set_title("Feature Importance")
    st.pyplot(fig4)

else:
    st.warning("Training data not found. Evaluation plots skipped.")

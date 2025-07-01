import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, f1_score, accuracy_score,
    classification_report
)
from sklearn.model_selection import train_test_split, learning_curve

# --- Paths ---
MODEL_PATH = "rf_model_cetobacterium.pkl"
FEATURES_PATH = "rf_model_features.pkl"
DATA_PATH = "final_selected_features_dataset.csv"

# --- Load model and features ---
if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    st.error("❌ Model or feature file is missing. Please upload them.")
    st.stop()

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

# --- Streamlit UI ---
st.title("🧬 Cetobacterium Predictor - Microplastic Exposure")
st.markdown("Predict the presence of *Cetobacterium* based on microplastic exposure conditions.")

st.sidebar.header("🧪 Exposure Inputs")
mp_conc = st.sidebar.slider("MP Concentration (µg/mL)", 0, 2000, 1000)
mp_size = st.sidebar.slider("MP Size (µm)", 0, 1000, 300)
exposure_time = st.sidebar.slider("Exposure Time (days)", 1, 30, 14)

# --- Create input vector ---
input_df = pd.DataFrame(columns=feature_names)
input_df.loc[0] = 0
input_df.loc[0, "MP_Concentration"] = mp_conc
input_df.loc[0, "MP_Size"] = mp_size
input_df.loc[0, "Exposure_Time"] = exposure_time

# --- Prediction ---
pred = model.predict(input_df)[0]
pred_label = "✅ Present" if pred == 1 else "❌ Absent"
st.subheader(f"Prediction: *Cetobacterium* is **{pred_label}**")

# --- Load training data for evaluation plots ---
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    X = df[feature_names]
    y = df["Cetobacterium_Present"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    st.subheader("📈 ROC Curve (AUC)")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)

    # --- Precision-Recall Curve ---
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    st.subheader("📉 Precision-Recall Curve")
    fig2, ax2 = plt.subplots()
    ax2.plot(recall, precision, color='purple', lw=2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    st.pyplot(fig2)

    # --- F1 Score and Report ---
    st.subheader("📊 Classification Metrics")
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"**F1 Score:** {f1:.2f}")
    st.write(f"**Accuracy:** {acc:.2f}")
    st.text(classification_report(y_test, y_pred))

    # --- Learning Curve ---
    st.subheader("📚 Learning Curve")
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='f1', n_jobs=-1)
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
    st.subheader("📌 Feature Importance")
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    fig4, ax4 = plt.subplots()
    importances.plot(kind='barh', ax=ax4)
    ax4.set_title("Feature Importance")
    st.pyplot(fig4)

else:
    st.info("Training data not found – evaluation plots skipped.")

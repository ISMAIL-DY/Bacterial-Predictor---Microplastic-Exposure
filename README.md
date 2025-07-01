# Bacterial Predictor - Microplastic Exposure FMPR UM6P

This Streamlit web application predicts the presence of **Cetobacterium** or **Rhizobiales** in fish gut microbiota based on **microplastic (MP) exposure conditions**.

---

## Purpose

The project was developed as part of the **FINAL Master THESIS PROJECT in Computational Biology, Bioinformatics and Data Analysis at the Faculty of Medicine and Pharmacy, Rabat** / **University Mohammed VI Polytechnic UM6P(AGBS DEPARTMENT)**.

It supports:
- Research on gut microbiome responses to environmental stressors
- Modeling microbial shifts due to plastic pollution

---

## What It Does

Users can:
- Select a target bacterium (Cetobacterium or Rhizobiales)
- Input exposure conditions (MP concentration, size, and exposure time)
- Upload a CSV for batch predictions
- View:
  - Prediction and probability
  - Evaluation metrics (F1, ROC, PR curve, confusion matrix)
  - Feature importances

---

## Features Used in Model

- `MP_Concentration (µg/mL)`
- `MP_Size (µm)`
- `Exposure_Time (days)`

---

## Technologies

- Python
- Streamlit
- scikit-learn
- Pandas, NumPy, Matplotlib, Seaborn

---

## Getting Started

### Run Locally:

```bash
pip install -r requirements.txt
streamlit run app.py

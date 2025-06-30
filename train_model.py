import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 👇 TRAIN model if it doesn't exist
if not os.path.exists("rf_model_cetobacterium.pkl"):
    df = pd.read_csv("data/final_selected_features_dataset.csv")

    mp_type_cols = [col for col in df.columns if col.startswith("MP_Type_")]
    species_cols = [col for col in df.columns if col.startswith("Species_")]
    features = mp_type_cols + ['MP_Concentration', 'MP_Size', 'Exposure_Time'] + species_cols

    X = df[features]
    y = df['Cetobacterium_Present']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    # Save the model and features
    joblib.dump(model, "rf_model_cetobacterium.pkl")
    joblib.dump(X_train.columns.tolist(), "rf_model_features.pkl")

else:
    model = joblib.load("rf_model_cetobacterium.pkl")

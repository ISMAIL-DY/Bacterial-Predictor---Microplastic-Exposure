import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("data/final_selected_features_dataset.csv")

# Define features and target
mp_type_cols = [col for col in df.columns if col.startswith("MP_Type_")]
species_cols = [col for col in df.columns if col.startswith("Species_")]
features = mp_type_cols + ['MP_Concentration', 'MP_Size', 'Exposure_Time'] + species_cols
X = df[features]
y = df['Cetobacterium_Present']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

# Save model and feature names
joblib.dump(rf, "rf_model_cetobacterium.pkl")
joblib.dump(X_train.columns.tolist(), "rf_model_features.pkl")

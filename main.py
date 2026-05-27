# ======================================
# TRAIN MODEL SCRIPT
# File: train_model.py
# Project: Employee Retention Prediction
# ======================================

# ========== STEP 1: IMPORT LIBRARIES ==========
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# ========== STEP 2: LOAD DATA ==========
print("📥 Loading dataset...")
df = pd.read_csv("aug_train.csv")


# ========== STEP 3: DROP UNNECESSARY COLUMNS ==========
if 'enrollee_id' in df.columns:
    df.drop(columns=['enrollee_id'], inplace=True)


# ========== STEP 4: SPLIT FEATURES & TARGET ==========
X = df.drop(columns=['target'])
y = df['target']


# ========== STEP 5: HANDLE MISSING VALUES ==========
categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(exclude='object').columns

for col in categorical_cols:
    X[col].fillna(X[col].mode()[0], inplace=True)

for col in numerical_cols:
    X[col].fillna(X[col].median(), inplace=True)


# ========== STEP 6: LABEL ENCODING ==========
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le


# ========== STEP 7: FEATURE SCALING ==========
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])


# ========== STEP 8: TRAIN-TEST SPLIT ==========
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ========== STEP 9: HANDLE CLASS IMBALANCE ==========
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# ========== STEP 10: TRAIN XGBOOST MODEL ==========
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

print("🚀 Training model...")
model.fit(X_train_resampled, y_train_resampled)


# ========== STEP 11: MODEL EVALUATION ==========
y_pred = model.predict(X_test)

print("\n📊 Model Performance")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ========== STEP 12: SAVE MODEL & OBJECTS ==========
joblib.dump(model, "final_xgboost_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("\n✅ Model training completed")
print("💾 Files saved:")
print("   - final_xgboost_model.pkl")
print("   - scaler.pkl")
print("   - label_encoders.pkl")

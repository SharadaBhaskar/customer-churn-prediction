import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import joblib
import os
import json

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'telco_churn.csv')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')

# ── Step 1: Load Data ─────────────────────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"   Shape: {df.shape}")

# ── Step 2: Clean Data ────────────────────────────────────────────────────────
print("🧹 Cleaning data...")

# Fix TotalCharges — it has spaces instead of numbers
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Drop customerID — not needed for training
df.drop('customerID', axis=1, inplace=True)

# Convert target — Yes=1, No=0
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

print(f"   Churn distribution:\n{df['Churn'].value_counts()}")

# ── Step 3: Encode Categorical Columns ───────────────────────────────────────
print("🔤 Encoding categorical columns...")

categorical_cols = df.select_dtypes(include='object').columns.tolist()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save label encoders
joblib.dump(label_encoders,
            os.path.join(MODEL_DIR, 'label_encoders.pkl'))
print(f"   Encoded columns: {categorical_cols}")

# ── Step 4: Split Features and Target ────────────────────────────────────────
X = df.drop('Churn', axis=1)
y = df['Churn']

# Save feature names
feature_names = X.columns.tolist()
with open(os.path.join(MODEL_DIR, 'feature_names.json'), 'w') as f:
    json.dump(feature_names, f)
print(f"   Features: {feature_names}")

# ── Step 5: Scale Features ────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

# ── Step 6: Train Test Split ──────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"\n📊 Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ── Step 7: Train All 3 Models ────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000,
                                               random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100,
                                                   random_state=42),
    'XGBoost':             XGBClassifier(n_estimators=100,
                                          random_state=42,
                                          eval_metric='logloss')
}

results = {}
best_model      = None
best_model_name = ''
best_f1         = 0

print("\n🤖 Training models...")
for name, model in models.items():
    print(f"\n   Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    results[name] = {
        'accuracy':  round(acc  * 100, 2),
        'precision': round(prec * 100, 2),
        'recall':    round(rec  * 100, 2),
        'f1_score':  round(f1   * 100, 2)
    }

    print(f"   ✅ Accuracy: {acc*100:.2f}%  "
          f"F1: {f1*100:.2f}%")

    if f1 > best_f1:
        best_f1         = f1
        best_model      = model
        best_model_name = name

# ── Step 8: Save Best Model ───────────────────────────────────────────────────
print(f"\n🏆 Best Model: {best_model_name} (F1: {best_f1*100:.2f}%)")
joblib.dump(best_model,
            os.path.join(MODEL_DIR, 'best_model.pkl'))

# Save results
with open(os.path.join(MODEL_DIR, 'model_results.json'), 'w') as f:
    json.dump(results, f, indent=4)

# Save feature importances (XGBoost)
if hasattr(best_model, 'feature_importances_'):
    importances = dict(zip(feature_names,
                           best_model.feature_importances_.tolist()))
    importances = dict(sorted(importances.items(),
                               key=lambda x: x[1], reverse=True))
    with open(os.path.join(MODEL_DIR, 'feature_importances.json'), 'w') as f:
        json.dump(importances, f, indent=4)

print("\n✅ All models trained and saved successfully!")
print("📁 Saved files:")
print("   - models/best_model.pkl")
print("   - models/scaler.pkl")
print("   - models/label_encoders.pkl")
print("   - models/feature_names.json")
print("   - models/model_results.json")
print("   - models/feature_importances.json")
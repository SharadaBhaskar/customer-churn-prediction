import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "dataset", "churn_data.csv")

df = pd.read_csv(DATA_PATH)

# CLEAN COLUMN NAMES
df.columns = df.columns.str.strip()

# REMOVE EMPTY STRINGS
df.replace(" ", pd.NA, inplace=True)
df.dropna(inplace=True)

# CONVERT NUMERIC COLUMNS SAFELY
df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# DROP ROWS THAT FAILED CONVERSION
df.dropna(inplace=True)

# TARGET CLEANING
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# FEATURES
X = df[["tenure", "MonthlyCharges", "TotalCharges"]]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, os.path.join(BASE_DIR, "..", "models", "churn_model.pkl"))

print("Model trained successfully!")
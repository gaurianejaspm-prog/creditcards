"""
train_model.py
--------------
Trains a fraud-detection ML pipeline on data/dataset.csv,
compares multiple classifiers, selects the best by ROC-AUC,
and saves the complete pipeline to models/model.pkl.
"""

import os
import warnings
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ── 1. Paths ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "dataset.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 2. Load dataset ─────────────────────────────────────────────────────────
print("Loading dataset …")
df = pd.read_csv(DATA_PATH)
print(f"  Shape  : {df.shape}")
print(f"  Columns: {list(df.columns)}")

TARGET = "is_fraud"
DROP_COLS = ["transaction_id"]

df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
print(f"  Fraud rate: {df[TARGET].mean():.4%}")

# ── 3. Feature type identification ──────────────────────────────────────────
CATEGORICAL_COLS = [
    "merchant_category", "card_type", "auth_method", "channel", "device_type"
]

BOOLEAN_COLS = [
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]

# Convert booleans to int (handles string 'True'/'False' or real bools)
for col in BOOLEAN_COLS:
    df[col] = df[col].astype(str).str.lower().map({"true": 1, "false": 0})

NUMERICAL_COLS = [
    c for c in df.columns
    if c not in CATEGORICAL_COLS + BOOLEAN_COLS + [TARGET]
]

print(f"\n  Numerical  ({len(NUMERICAL_COLS)}): {NUMERICAL_COLS}")
print(f"  Categorical({len(CATEGORICAL_COLS)}): {CATEGORICAL_COLS}")
print(f"  Boolean    ({len(BOOLEAN_COLS)}): {BOOLEAN_COLS}")

# ── 4. Train / test split ────────────────────────────────────────────────────
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape}  Test: {X_test.shape}")

# ── 5. Preprocessing pipeline ───────────────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERICAL_COLS + BOOLEAN_COLS),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
         CATEGORICAL_COLS),
    ],
    remainder="drop",
)

# ── 6. Candidate models ──────────────────────────────────────────────────────
CANDIDATES = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        n_jobs=-1, random_state=42
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=5, random_state=42
    ),
}

# ── 7. Cross-validated comparison ───────────────────────────────────────────
print("\nCross-validating candidates (5-fold, ROC-AUC) …")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, clf in CANDIDATES.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    results[name] = scores.mean()
    print(f"  {name:25s}  ROC-AUC = {scores.mean():.4f} ± {scores.std():.4f}")

best_name = max(results, key=results.get)
print(f"\nBest model: {best_name}  (AUC={results[best_name]:.4f})")

# ── 8. Final fit on full training set ───────────────────────────────────────
best_pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", CANDIDATES[best_name]),
])
best_pipeline.fit(X_train, y_train)

# ── 9. Evaluate on held-out test set ────────────────────────────────────────
y_pred  = best_pipeline.predict(X_test)
y_proba = best_pipeline.predict_proba(X_test)[:, 1]

print("\n--- Test-set evaluation ---")
print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
print(f"ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}")

cm = confusion_matrix(y_test, y_pred)
print(f"Confusion matrix:\n{cm}")

# ── 10. Save pipeline + metadata ────────────────────────────────────────────
artifact = {
    "pipeline":         best_pipeline,
    "model_name":       best_name,
    "feature_columns":  list(X.columns),
    "categorical_cols": CATEGORICAL_COLS,
    "boolean_cols":     BOOLEAN_COLS,
    "numerical_cols":   NUMERICAL_COLS,
    "target":           TARGET,
    "category_values": {
        col: sorted(df[col].dropna().unique().tolist())
        for col in CATEGORICAL_COLS
    },
    "test_roc_auc":  roc_auc_score(y_test, y_proba),
    "cv_results":    results,
}

joblib.dump(artifact, MODEL_PATH)
print(f"\nModel saved → {MODEL_PATH}")
print("Training complete ✓")

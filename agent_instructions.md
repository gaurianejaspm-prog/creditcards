# Agent Instructions

## Project: Credit Card Fraud Detection ML App

### Overview
This project detects fraudulent credit card transactions using a machine-learning pipeline served via a Streamlit web application.

### Dataset
- **Location:** `data/dataset.csv`
- **Source:** Credit Card Fraud 2026 (synthetic)
- **Rows:** 20,000 | **Columns:** 26
- **Target:** `is_fraud` (0 = Legitimate, 1 = Fraud)
- **Class imbalance:** ~1.7% fraud rate → models use `class_weight="balanced"` or equivalent

### Training Pipeline (`train_model.py`)
1. Load CSV from `data/dataset.csv`
2. Drop `transaction_id` (identifier, not a feature)
3. Convert boolean columns from string `"True"`/`"False"` to integer 0/1
4. Apply `ColumnTransformer`:
   - `StandardScaler` → numerical + boolean columns
   - `OrdinalEncoder` → categorical columns
5. Compare 3 classifiers via 5-fold stratified cross-validation (ROC-AUC)
6. Re-fit best classifier on full training set
7. Evaluate on held-out 20% test set
8. Save full `sklearn.Pipeline` + metadata as `models/model.pkl` via `joblib`

### Inference App (`app.py`)
- Loads `models/model.pkl` with `@st.cache_resource`
- Builds a single-row `pd.DataFrame` from user inputs in the **exact column order** stored in `artifact["feature_columns"]`
- Calls `pipeline.predict()` and `pipeline.predict_proba()` — **no hard-coded predictions**
- Displays fraud probability, risk level, and input summary

### Key Conventions
- All paths use `os.path.join(BASE_DIR, ...)` — **no absolute paths**
- No API keys or secrets required
- `requirements.txt` covers all runtime dependencies

### Extending the Project
- To add new models: add an entry to `CANDIDATES` in `train_model.py` and re-run
- To add XGBoost/LightGBM: install the package, import, add to `CANDIDATES`
- To handle Excel input: `pd.read_excel("data/dataset.xlsx", engine="openpyxl")`

### Deployment Checklist
- [ ] `data/dataset.csv` committed to repo
- [ ] `models/model.pkl` committed (or train on first deploy)
- [ ] `requirements.txt` up to date
- [ ] No absolute paths, no hard-coded secrets
- [ ] `streamlit run app.py` works locally

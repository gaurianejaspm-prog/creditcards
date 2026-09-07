"""
app.py
------
Streamlit web application for Credit Card Fraud Detection.
Loads the trained ML pipeline from models/model.pkl and serves
real-time predictions based on user-provided transaction features.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

# ── Load model artifact ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_artifact():
    if not os.path.exists(MODEL_PATH):
        st.error(
            "Model file not found. "
            "Please run `python train_model.py` first."
        )
        st.stop()
    return joblib.load(MODEL_PATH)

artifact = load_artifact()

pipeline         = artifact["pipeline"]
feature_columns  = artifact["feature_columns"]
categorical_cols = artifact["categorical_cols"]
boolean_cols     = artifact["boolean_cols"]
numerical_cols   = artifact["numerical_cols"]
category_values  = artifact["category_values"]
model_name       = artifact["model_name"]
test_roc_auc     = artifact["test_roc_auc"]

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🛡️",
    layout="wide",
)

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🛡️ Credit Card Fraud Detection")
st.markdown(
    f"Real-time fraud prediction powered by **{model_name}**  |  "
    f"Test ROC-AUC: **{test_roc_auc:.4f}**"
)
st.divider()

# ── Sidebar – model info ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Model Info")
    st.write(f"**Algorithm:** {model_name}")
    st.write(f"**Test ROC-AUC:** {test_roc_auc:.4f}")
    st.write(f"**Features used:** {len(feature_columns)}")
    st.markdown("---")
    st.write("**CV Scores by Model:**")
    for m, s in artifact["cv_results"].items():
        st.write(f"- {m}: {s:.4f}")
    st.markdown("---")
    st.caption("Dataset: Credit Card Fraud 2026")

# ── Input form ───────────────────────────────────────────────────────────────
st.subheader("Enter Transaction Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**💳 Transaction Info**")
    amount_usd = st.number_input(
        "Transaction Amount (USD)", min_value=0.01, max_value=100000.0,
        value=50.0, step=0.01
    )
    merchant_category = st.selectbox(
        "Merchant Category", category_values["merchant_category"]
    )
    card_type = st.selectbox("Card Type", category_values["card_type"])
    auth_method = st.selectbox(
        "Authentication Method", category_values["auth_method"]
    )
    channel = st.selectbox("Transaction Channel", category_values["channel"])
    device_type = st.selectbox("Device Type", category_values["device_type"])

with col2:
    st.markdown("**📊 Behavioural Signals**")
    hours_since_last_txn = st.number_input(
        "Hours Since Last Transaction", min_value=0.0, max_value=720.0,
        value=5.0, step=0.1
    )
    txn_count_last_24h = st.number_input(
        "Transactions in Last 24 h", min_value=0, max_value=200,
        value=3, step=1
    )
    distance_from_home_km = st.number_input(
        "Distance from Home (km)", min_value=0.0, max_value=20000.0,
        value=10.0, step=0.1
    )
    velocity_score = st.number_input(
        "Velocity Score", min_value=0.0, max_value=100.0,
        value=20.0, step=0.1
    )
    merchant_risk_score = st.number_input(
        "Merchant Risk Score", min_value=0.0, max_value=100.0,
        value=30.0, step=0.1
    )
    cvv_retry_count = st.number_input(
        "CVV Retry Count", min_value=0, max_value=10,
        value=0, step=1
    )

with col3:
    st.markdown("**👤 Customer & Card Info**")
    customer_age = st.number_input(
        "Customer Age", min_value=18, max_value=100, value=35, step=1
    )
    card_age_months = st.number_input(
        "Card Age (months)", min_value=0, max_value=600, value=24, step=1
    )
    account_balance_usd = st.number_input(
        "Account Balance (USD)", min_value=0.0, max_value=500000.0,
        value=2000.0, step=1.0
    )
    prior_disputes = st.number_input(
        "Prior Disputes", min_value=0, max_value=50, value=0, step=1
    )
    time_of_day_hour = st.slider(
        "Hour of Day (0–23)", min_value=0, max_value=23, value=14
    )
    day_of_week = st.slider(
        "Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=2
    )

    st.markdown("**🚨 Risk Flags**")
    is_foreign_transaction     = st.checkbox("Foreign Transaction")
    is_new_merchant            = st.checkbox("New Merchant")
    used_vpn                   = st.checkbox("VPN Used")
    ip_country_mismatch        = st.checkbox("IP / Country Mismatch")
    billing_shipping_mismatch  = st.checkbox("Billing / Shipping Mismatch")
    is_ai_generated_scam_attempt = st.checkbox("AI-Generated Scam Attempt")

st.divider()

# ── Predict ──────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Fraud", type="primary", use_container_width=True):

    # Build input dict in the same column order as training
    input_data = {
        "amount_usd":                   amount_usd,
        "merchant_category":            merchant_category,
        "card_type":                    card_type,
        "auth_method":                  auth_method,
        "channel":                      channel,
        "device_type":                  device_type,
        "is_foreign_transaction":       int(is_foreign_transaction),
        "hours_since_last_txn":         hours_since_last_txn,
        "txn_count_last_24h":           txn_count_last_24h,
        "distance_from_home_km":        distance_from_home_km,
        "card_age_months":              card_age_months,
        "customer_age":                 customer_age,
        "account_balance_usd":          account_balance_usd,
        "is_new_merchant":              int(is_new_merchant),
        "used_vpn":                     int(used_vpn),
        "ip_country_mismatch":          int(ip_country_mismatch),
        "billing_shipping_mismatch":    int(billing_shipping_mismatch),
        "cvv_retry_count":              cvv_retry_count,
        "velocity_score":               velocity_score,
        "time_of_day_hour":             time_of_day_hour,
        "day_of_week":                  day_of_week,
        "is_ai_generated_scam_attempt": int(is_ai_generated_scam_attempt),
        "merchant_risk_score":          merchant_risk_score,
        "prior_disputes":               prior_disputes,
    }

    # Guarantee column order matches training
    input_df = pd.DataFrame([input_data])[feature_columns]

    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0]

    fraud_prob  = probability[1] * 100
    legit_prob  = probability[0] * 100

    st.subheader("🔎 Prediction Result")

    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        if prediction == 1:
            st.error("⚠️ **FRAUDULENT TRANSACTION**")
        else:
            st.success("✅ **LEGITIMATE TRANSACTION**")

    with res_col2:
        st.metric("Fraud Probability",   f"{fraud_prob:.2f}%")

    with res_col3:
        st.metric("Legitimate Probability", f"{legit_prob:.2f}%")

    # Risk bar
    st.markdown("**Risk Level**")
    st.progress(int(fraud_prob))

    # Colour-coded risk band
    if fraud_prob < 20:
        st.info("🟢 Low Risk")
    elif fraud_prob < 50:
        st.warning("🟡 Moderate Risk — review recommended")
    elif fraud_prob < 75:
        st.warning("🟠 High Risk — manual verification advised")
    else:
        st.error("🔴 Critical Risk — block transaction immediately")

    # Show input summary
    with st.expander("📋 Input Summary"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "This application uses a machine-learning model trained on the "
    "Credit Card Fraud 2026 dataset. "
    "Predictions are probabilistic and should be used as decision support only."
)

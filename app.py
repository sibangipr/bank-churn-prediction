import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="wide"
)

# ------------------ LOAD MODEL ------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ------------------ TITLE ------------------
st.title("🏦 Bank Customer Churn Prediction Dashboard")
st.markdown("Predict customer churn risk using Machine Learning")

# ------------------ SIDEBAR INPUT ------------------
st.sidebar.header("📥 Customer Details")

credit_score = st.sidebar.slider("Credit Score", 300, 900, 600)
age = st.sidebar.slider("Age", 18, 100, 35)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 3)
balance = st.sidebar.number_input("Balance", value=50000)
num_products = st.sidebar.slider("Number of Products", 1, 4, 2)
has_card = st.sidebar.selectbox("Has Credit Card", [0, 1])
is_active = st.sidebar.selectbox("Is Active Member", [0, 1])
salary = st.sidebar.number_input("Estimated Salary", value=50000)

# ------------------ FEATURE ENGINEERING ------------------
balance_salary_ratio = balance / (salary + 1)
product_density = num_products / (tenure + 1)
age_tenure = age * tenure

# ------------------ CREATE INPUT DATA ------------------
input_df = pd.DataFrame([[
    credit_score, age, tenure, balance, num_products,
    has_card, is_active, salary,
    balance_salary_ratio, product_density, age_tenure
]])

# Apply scaling
input_scaled = scaler.transform(input_df)

# ------------------ PREDICTION ------------------
prob = model.predict_proba(input_scaled)[0][1]

# Risk category
if prob < 0.3:
    risk = "🟢 Low Risk"
elif prob < 0.7:
    risk = "🟡 Medium Risk"
else:
    risk = "🔴 High Risk"

# ------------------ LAYOUT ------------------
col1, col2 = st.columns(2)

# ------------------ OUTPUT ------------------
with col1:
    st.subheader("📊 Prediction Result")

    st.metric("Churn Probability", f"{prob:.2f}")

    st.progress(int(prob * 100))

    st.markdown(f"### Risk Level: {risk}")

# ------------------ WHAT-IF ANALYSIS ------------------
with col2:
    st.subheader("🔄 What-if Analysis")
    st.info("Change inputs from sidebar to see real-time effect")

# ------------------ FEATURE IMPORTANCE ------------------
st.subheader("📌 Feature Importance")

try:
    importances = model.feature_importances_

    features = [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary",
        "BalanceSalaryRatio", "ProductDensity", "AgeTenure"
    ]

    feat_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots()
    ax.barh(feat_df["Feature"], feat_df["Importance"])
    ax.invert_yaxis()

    st.pyplot(fig)

except:
    st.warning("Feature importance not available for this model")

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("👨‍💻 Developed by Ashutosh Panda")
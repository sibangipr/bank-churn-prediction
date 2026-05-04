import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ------------------ PAGE CONFIG ------------------
features = pickle.load(open("features.pkl", "rb"))
st.set_page_config(
    page_title="Churn Intelligence Dashboard",
    page_icon="🏦",
    layout="wide"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}
.stMetric {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 10px;
}
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.5);
}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ------------------ HEADER ------------------
st.title("🏦 Churn Intelligence Dashboard")
st.markdown("### Predict customer churn risk with advanced analytics")

# ------------------ SIDEBAR ------------------
st.sidebar.header("📥 Customer Profile")

credit_score = st.sidebar.slider("Credit Score", 300, 900, 650)
age = st.sidebar.slider("Age", 18, 100, 35)
tenure = st.sidebar.slider("Tenure", 0, 10, 5)
balance = st.sidebar.number_input("Account Balance", value=60000)
products = st.sidebar.slider("Products", 1, 4, 2)
has_card = st.sidebar.selectbox("Credit Card", [0, 1])
active = st.sidebar.selectbox("Active Member", [0, 1])
salary = st.sidebar.number_input("Estimated Salary", value=50000)

# ✅ FIXED (Added Missing Inputs)
geography = st.sidebar.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

# ------------------ FEATURE ENGINEERING ------------------
balance_salary_ratio = balance / (salary + 1)
product_density = products / (tenure + 1)
age_tenure = age * tenure

# ------------------ INPUT DATA ------------------
input_dict = {
    'CreditScore': credit_score,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': products,
    'HasCrCard': has_card,
    'IsActiveMember': active,
    'EstimatedSalary': salary,
    'BalanceSalaryRatio': balance_salary_ratio,
    'ProductDensity': product_density,
    'AgeTenure': age_tenure,

    # Dummy variables
    'Geography_Germany': 1 if geography == "Germany" else 0,
    'Geography_Spain': 1 if geography == "Spain" else 0,
    'Gender_Male': 1 if gender == "Male" else 0
}

input_df = pd.DataFrame([input_dict])

# ------------------ FEATURE ORDER MATCH ------------------
model_features = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'BalanceSalaryRatio', 'ProductDensity', 'AgeTenure',
    'Geography_Germany', 'Geography_Spain', 'Gender_Male'
]

input_df = input_df.reindex(columns=features, fill_value=0)

# ------------------ SCALING ------------------
input_scaled = scaler.transform(input_df)

# ------------------ PREDICTION ------------------
prob = model.predict_proba(input_scaled)[0][1]

# ------------------ RISK LEVEL ------------------
if prob < 0.3:
    risk = "🟢 LOW RISK"
elif prob < 0.7:
    risk = "🟡 MEDIUM RISK"
else:
    risk = "🔴 HIGH RISK"

# ------------------ KPI CARDS ------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Churn Probability", f"{prob:.2f}")

with col2:
    st.metric("Risk Level", risk)

with col3:
    st.metric("Customer Score", f"{(1-prob)*100:.0f}/100")

# ------------------ PROGRESS BAR ------------------
st.markdown("### 📊 Risk Gauge")
st.progress(int(prob * 100))

# ------------------ WHAT-IF ------------------
st.markdown("### 🔄 What-if Simulation")
st.info("Adjust inputs in sidebar to simulate churn probability.")

# ------------------ FEATURE IMPORTANCE ------------------
st.markdown("### 📈 Feature Importance")

try:
    importances = model.feature_importances_

    features = [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary",
        "BalanceSalaryRatio", "ProductDensity", "AgeTenure",
        "Geography_Germany", "Geography_Spain", "Gender_Male"
    ]

    feat_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots()
    ax.barh(feat_df["Feature"], feat_df["Importance"])
    ax.set_title("Top Features Influencing Churn")
    ax.invert_yaxis()

    st.pyplot(fig)

except:
    st.warning("Feature importance available only for tree-based models")

# ------------------ INSIGHTS ------------------
st.markdown("### 🧠 Insights")

if prob > 0.7:
    st.error("⚠️ Customer is highly likely to churn. Immediate action needed.")
elif prob > 0.4:
    st.warning("⚡ Moderate churn risk. Engage with offers.")
else:
    st.success("✅ Customer is stable.")

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("🚀 Developed by Sibangi Tripathy | Customer Engagement and retention strategy")
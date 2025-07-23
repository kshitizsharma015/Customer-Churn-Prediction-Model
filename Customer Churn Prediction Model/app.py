
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
model = pickle.load(open("logistic_model.pkl", "rb"))

# Page config
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📊 Customer Churn Prediction App")

st.markdown("Fill the customer's details to predict churn likelihood.")

# User input widgets
gender = st.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
MonthlyCharges = st.slider("Monthly Charges", 0.0, 120.0, 70.0)
TotalCharges = st.slider("Total Charges", 0.0, 9000.0, 2500.0)

# Predict button
if st.button("Predict"):
    df = pd.DataFrame({
        "gender": [0 if gender == "Female" else 1],
        "SeniorCitizen": [SeniorCitizen],
        "Partner": [1 if Partner == "Yes" else 0],
        "Dependents": [1 if Dependents == "Yes" else 0],
        "tenure": [tenure],
        "PhoneService": [1 if PhoneService == "Yes" else 0],
        "PaperlessBilling": [1 if PaperlessBilling == "Yes" else 0],
        "MonthlyCharges": [(MonthlyCharges - 64.76)/30.09],
        "TotalCharges": [(TotalCharges - 2283.3)/2266.77]
    })

    # One-hot columns (manually encoded)
    multi = {
        "MultipleLines_No phone service": 0, "MultipleLines_Yes": 0,
        "InternetService_Fiber optic": 0, "InternetService_No": 0,
        "OnlineSecurity_No": 0, "OnlineSecurity_No internet service": 0,
        "OnlineBackup_No": 0, "OnlineBackup_No internet service": 0,
        "DeviceProtection_No": 0, "DeviceProtection_No internet service": 0,
        "TechSupport_No": 0, "TechSupport_No internet service": 0,
        "StreamingTV_No": 0, "StreamingTV_No internet service": 0,
        "StreamingMovies_No": 0, "StreamingMovies_No internet service": 0,
        "Contract_One year": 0, "Contract_Two year": 0,
        "PaymentMethod_Credit card (automatic)": 0,
        "PaymentMethod_Electronic check": 0,
        "PaymentMethod_Mailed check": 0
    }
    keys = list(multi.keys())
    values = [0]*len(keys)

    col_map = {
        f"MultipleLines_{MultipleLines}",
        f"InternetService_{InternetService}",
        f"OnlineSecurity_{OnlineSecurity}",
        f"OnlineBackup_{OnlineBackup}",
        f"DeviceProtection_{DeviceProtection}",
        f"TechSupport_{TechSupport}",
        f"StreamingTV_{StreamingTV}",
        f"StreamingMovies_{StreamingMovies}",
        f"Contract_{Contract}",
        f"PaymentMethod_{PaymentMethod}"
    }

    for key in keys:
        multi[key] = 1 if key in col_map else 0

    for k in keys:
        df[k] = [multi[k]]

    prediction = model.predict(df)[0]
    result = "🟡 This customer is likely to churn." if prediction else "🟢 This customer is likely to stay."

    st.subheader("Prediction:")
    st.success(result)

import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("churn_model.pkl")

st.set_page_config(page_title="📉 Customer Churn Predictor", layout="centered")
st.title("📊 Customer Churn Prediction App")

st.markdown("Enter customer details to predict the likelihood of churn.")

# Input fields (must match encoded model input)

gender = st.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
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
    "Electronic check", 
    "Mailed check", 
    "Bank transfer (automatic)", 
    "Credit card (automatic)"
])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, value=800.0)

# Encode inputs same as LabelEncoder (match label order used during training)
def encode(val, options):
    return options.index(val)

# Encoding in exact order used in notebook
input_data = np.array([[
    encode(gender, ["Female", "Male"]),
    SeniorCitizen,
    encode(Partner, ["Yes", "No"]),
    encode(Dependents, ["Yes", "No"]),
    tenure,
    encode(PhoneService, ["Yes", "No"]),
    encode(MultipleLines, ["No", "Yes", "No phone service"]),
    encode(InternetService, ["DSL", "Fiber optic", "No"]),
    encode(OnlineSecurity, ["Yes", "No", "No internet service"]),
    encode(OnlineBackup, ["Yes", "No", "No internet service"]),
    encode(DeviceProtection, ["Yes", "No", "No internet service"]),
    encode(TechSupport, ["Yes", "No", "No internet service"]),
    encode(StreamingTV, ["Yes", "No", "No internet service"]),
    encode(StreamingMovies, ["Yes", "No", "No internet service"]),
    encode(Contract, ["Month-to-month", "One year", "Two year"]),
    encode(PaperlessBilling, ["Yes", "No"]),
    encode(PaymentMethod, [
        "Electronic check", 
        "Mailed check", 
        "Bank transfer (automatic)", 
        "Credit card (automatic)"
    ]),
    MonthlyCharges,
    TotalCharges
]])

# Prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ The customer is likely to **Churn**.")
    else:
        st.success("✅ The customer is likely to **Stay**.")

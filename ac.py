import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load the AC price/bill model
model_path = Path(__file__).parent / "AC_Price.pkl"
model = joblib.load(model_path)

st.title("AC Price Predictor")
st.write("Enter the AC units and electric bill details below to predict the estimated outcome.")

ac_units = st.number_input("AC Units", min_value=0.0, max_value=50.0, value=1.0, step=1.0)
electric_bill = st.number_input("Electric Bill (₹)", min_value=0.0, max_value=100000.0, value=500.0, step=50.0)

if st.button("Predict"):
    n_features = getattr(model, "n_features_in_", 1)
    if n_features == 3:
        input_data = pd.DataFrame([[0, ac_units, electric_bill]])
    else:
        input_data = pd.DataFrame({"AC_Units": [ac_units]})

    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Price / Bill: ₹{prediction:.2f}")


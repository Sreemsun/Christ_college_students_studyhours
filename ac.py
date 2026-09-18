import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load the AC electric bill model
model_path = Path(__file__).parent / "AC_Price.pkl"
model = joblib.load(model_path)

st.title("Electric Bill Predictor")
st.write("Enter the number of AC units to predict the estimated electric bill.")

ac_units = st.number_input("AC Units", min_value=0.0, max_value=150.0, value=10.0, step=5.0)
fans_units = st.number_input("Fans Units", min_value=0.0, max_value=200.0, value=10.0, step=5.0)

if st.button("Predict"):
    input_data = pd.DataFrame({"AC_Units": [ac_units],"Fan_Units": [fans_units]})
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Electric Bill: ₹{max(0.0, prediction):.2f}")

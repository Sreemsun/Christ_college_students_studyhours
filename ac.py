import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

model_path = Path(__file__).parent / "AC_Price.pkl"
model = joblib.load(model_path)

st.title("Electric Bill Predictor")
st.write("Enter the AC units to predict the electric bill.")

AC_Units = st.number_input(
    "AC Units",
    min_value=0.0,
    step=5.0
)

if st.button("Predict"):
    input_data = pd.DataFrame({
        "AC_Units": [AC_Units]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Electric Bill: ₹{prediction:.2f}")

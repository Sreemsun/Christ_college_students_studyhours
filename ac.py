import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

model_path = Path(__file__).parent / "AC_Price.pkl"
model = joblib.load(model_path)

st.title("AC Price Predictor")
st.write("Enter the AC units and electric bill to predict the price.")

AC_Units = st.number_input("SAC_Units", min_value=0.0, step=0.5)

if st.button("Predict"):
	input_data = pd.DataFrame({
		"AC Units": [AC_Units]
	})
	prediction = model.predict(input_data)[0]
	probability = model.predict_proba(input_data)[0][int(prediction)]

	st.success(f"Predicted Electric Bill: ₹{prediction:.2f}")

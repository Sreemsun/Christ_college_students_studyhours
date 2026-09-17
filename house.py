import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load the house price model
model_path = Path(__file__).parent / "House_price_model.pkl"
model = joblib.load(model_path)

st.title("House Price Predictor")
st.write("Enter the details of the property below to predict the estimated price.")

area = st.number_input("Area (sq ft)", min_value=100.0, max_value=20000.0, value=1200.0, step=50.0)
floors = st.number_input("Total_Floors", min_value=0, max_value=100, value=5, step=1)
bedroom = st.number_input("Bedrooms", min_value=1, max_value=20, value=3, step=1)


if st.button("Predict"):
	if area > 6000:
		st.error("Error: Area cannot be greater than 6000 sq ft.")
	elif bedroom > 5:
		st.error("Error: Number of bedrooms cannot be greater than 5.")
	elif floors > 10:
		st.error("Error: Number of floors cannot be greater than 10.")
	else:
		input_data = pd.DataFrame({
			"Area": [area],
			"Bedroom": [bedroom],
			"Total_Floors": [floors]
		})

		prediction = model.predict(input_data)[0]
		st.write(f"### Estimated Price: {prediction:.2f} Lakhs")

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load the house price model
model_path = Path(__file__).parent / "House_price_model.pkl"
model = joblib.load(model_path)

st.title("House Price Predictor")
st.write("Enter the details of the property below to predict the estimated price.")

# Input fields for model features: Area, Bedroom, Age
col1, col2, col3 = st.columns(3)

with col1:
    area = st.number_input("Area (sq ft)", min_value=100.0, max_value=10000.0, value=1200.0, step=50.0)

with col2:
    bedroom = st.number_input("Bedrooms", min_value=1, max_value=5, value=3, step=1)

with col3:
    age = st.number_input("Age of House (years)", min_value=0, max_value=70, value=5, step=1)

if st.button("Predict Price", type="primary"):
    input_data = pd.DataFrame({
        "Area": [area],
        "Bedroom": [bedroom],
        "Age": [age]
    })
    
    prediction = model.predict(input_data)[0]
    st.subheader(f"Estimated Price: {prediction:.2f} Lakhs")

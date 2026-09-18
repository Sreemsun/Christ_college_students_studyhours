import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load the AC electric bill model
model_path = Path(__file__).parent / "AC_Price.pkl"
model = joblib.load(model_path)

st.title("Electric Bill Predictor")
st.write("Enter the AC units to predict the electric bill.")

ac_units = st.number_input("AC Units", min_value=0.0, step=1.0)

if st.button("Predict"):
    n_features = getattr(model, "n_features_in_", 1)
    if n_features == 3:
        input_data = pd.DataFrame([[0, ac_units, 0]])
    else:
        input_data = pd.DataFrame({"AC_Units": [ac_units]})

    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Electric Bill: ₹{prediction:.2f}")



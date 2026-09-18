import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures

# Load the AC electric bill model
model_path = Path(__file__).parent / "AC_Price.pkl"
model = joblib.load(model_path)

st.title("Electric Bill Predictor")
st.write("Enter the number of AC and Fan units to predict the estimated electric bill.")

ac_units = st.number_input("AC Units", min_value=0.0, max_value=150.0, value=10.0, step=5.0)
fans_units = st.number_input("Fans Units", min_value=0.0, max_value=200.0, value=10.0, step=5.0)

if st.button("Predict"):
    n_features = getattr(model, "n_features_in_", 2)
    
    try:
        input_data = pd.DataFrame({"AC_Units": [ac_units], "Fan_Units": [fans_units]})
        prediction = model.predict(input_data)[0]
    except Exception:
        if n_features == 6:
            poly = PolynomialFeatures(degree=2)
            input_poly = poly.fit_transform([[ac_units, fans_units]])
            prediction = model.predict(input_poly)[0]
        elif n_features == 1:
            prediction = model.predict([[ac_units]])[0]
        else:
            prediction = model.predict([[ac_units, fans_units]])[0]

    st.success(f"Predicted Electric Bill: ₹{max(0.0, float(prediction)):.2f}")


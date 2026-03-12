import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("wine_quality_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="centered"
)

# Title
st.title("🍷 Wine Quality Prediction App")
st.write("Enter the chemical properties of wine to predict its quality.")

st.markdown("---")

# Two column layout
col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.slider("Fixed Acidity", 4.0, 15.0, 7.0)
    citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.3)
    chlorides = st.slider("Chlorides", 0.01, 0.2, 0.05)
    total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 10, 200, 50)
    pH = st.slider("pH", 2.5, 4.0, 3.2)
    alcohol = st.slider("Alcohol", 8.0, 15.0, 10.0)

with col2:
    volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.5, 0.5)
    residual_sugar = st.slider("Residual Sugar", 0.5, 10.0, 2.0)
    free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1, 70, 15)
    density = st.slider("Density", 0.990, 1.005, 0.996)
    sulphates = st.slider("Sulphates", 0.3, 2.0, 0.6)

st.markdown("---")

# Predict button
if st.button("Predict Wine Quality 🍷"):

    # Arrange input data in correct order
    data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                      residual_sugar, chlorides, free_sulfur_dioxide,
                      total_sulfur_dioxide, density, pH, sulphates, alcohol]])

    prediction = model.predict(data)[0]

    st.subheader("Prediction Result")

    if prediction >= 7:
        st.success(f"🍷 High Quality Wine (Score: {prediction})")
    elif prediction >= 5:
        st.info(f"🍷 Average Quality Wine (Score: {prediction})")
    else:
        st.warning(f"🍷 Low Quality Wine (Score: {prediction})")

st.markdown("---")

st.caption("Machine Learning model built with Scikit-learn and deployed using Streamlit.")
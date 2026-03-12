import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("wine_quality_model.pkl")

# Page settings
st.set_page_config(page_title="Wine Quality Tester", page_icon="🍷")

st.title("🍷 Wine Quality Prediction App")
st.write("Enter the wine chemical properties below to predict the quality.")

st.markdown("---")

# Inputs (no limits)
fixed_acidity = st.number_input("Fixed Acidity", step=0.1)
volatile_acidity = st.number_input("Volatile Acidity", step=0.01)
citric_acid = st.number_input("Citric Acid", step=0.01)
residual_sugar = st.number_input("Residual Sugar", step=0.1)
chlorides = st.number_input("Chlorides", step=0.001)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", step=1)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", step=1)
density = st.number_input("Density", step=0.0001)
pH = st.number_input("pH", step=0.01)
sulphates = st.number_input("Sulphates", step=0.01)
alcohol = st.number_input("Alcohol", step=0.1)

st.markdown("---")

# Prediction button
if st.button("Predict Wine Quality 🍷"):

    data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                      residual_sugar, chlorides, free_sulfur_dioxide,
                      total_sulfur_dioxide, density, pH, sulphates, alcohol]])

    prediction = model.predict(data)

    st.subheader("Prediction Result")

    if prediction[0] >= 7:
        st.success(f"🍷 High Quality Wine (Score: {prediction[0]})")
    elif prediction[0] >= 5:
        st.info(f"🍷 Average Quality Wine (Score: {prediction[0]})")
    else:
        st.warning(f"🍷 Low Quality Wine (Score: {prediction[0]}")

st.markdown("---")
st.caption("Built using Streamlit and Scikit-learn")

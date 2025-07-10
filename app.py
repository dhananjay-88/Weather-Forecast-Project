import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("weather_prediction_model.pkl")

# Page title and icon
st.set_page_config(page_title="Weather Predictor", page_icon="🌦")
st.title("🌦 Weather Prediction App")
st.markdown("Predict whether it will **rain or not** based on temperature, humidity, and wind speed.")

# Input form
with st.form("weather_form"):
    temp = st.number_input("🌡 Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0)
    humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
    wind = st.number_input("🍃 Wind Speed (km/h)", min_value=0.0, max_value=150.0, value=10.0)
    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    input_data = np.array([[temp, humidity, wind]])
    prediction = model.predict(input_data)

    st.markdown("### 🔍 Result:")
    if prediction[0] == 1:
        st.success("🌧️ **Rainy** — Better carry an umbrella!")
    else:
        st.info("☀️ **Not Rainy** — Looks like a dry day ahead.")

# Footer
st.markdown("---")
st.markdown("🛠 Built with Streamlit • Model: `weather_prediction_model.pkl`")

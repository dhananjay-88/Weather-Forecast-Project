import streamlit as st
import datetime
import pandas as pd
import joblib

# Load models once
temp_model = joblib.load("temp_model.pkl")
humidity_model = joblib.load("humidity_model.pkl")
wind_model = joblib.load("wind_model.pkl")
classifier = joblib.load("weather_prediction_model.pkl")

TRAINING_DAYS = 730  # Updated for 2 years dataset length 

# Load dataset for last 5 days weather display
df = pd.read_csv("weather_data_2 years.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

st.title("🌦 Weather Prediction App")

# --- Section 1: Single day prediction ---
st.header("Single Day Weather Prediction")

temp = st.number_input("Temperature (°C)", min_value=-50.0, max_value=60.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
wind = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=150.0, value=10.0)

if st.button("Predict Weather (Manual)"):
    features = [[temp, humidity, wind]]
    prediction = classifier.predict(features)[0]
    result = "🌧️ Rainy : Looks like the sky is throwing a water party" if prediction == 1 else "☀️ Not Rainy : Sun’s out, fun’s out!"
    st.success(f"Prediction: {result}")

st.markdown("---")

# --- Section 2: Multi-day prediction by date ---
st.header("Multi-Day Weather Prediction by Date")

start_date = st.date_input("Start Date", datetime.date.today())
days_to_predict = st.number_input("Number of Days to Predict", min_value=1, max_value=10, value=5)

if st.button("Predict Weather (By Date)"):
    today = datetime.date.today()
    day_diff = (start_date - today).days

    if day_diff < 0:
        st.error("❌ Please enter a future date.")
    else:
        predictions = []
        for i in range(days_to_predict):
            future_day_index = TRAINING_DAYS + day_diff + i
            X_day = pd.DataFrame([[future_day_index]], columns=["DayIndex"])

            # Predict weather features
            temp_pred = temp_model.predict(X_day)[0]
            humidity_pred = humidity_model.predict(X_day)[0]
            wind_pred = wind_model.predict(X_day)[0]

            # Predict rainy or not
            features = pd.DataFrame([[temp_pred, humidity_pred, wind_pred]], columns=["Temperature", "Humidity", "Wind Speed"])
            result = classifier.predict(features)[0]

            pred_label = "🌧️ Rainy : Pack your umbrella!" if result == 1 else "☀️ Not Rainy : Perfect for a picnic!"
            pred_date = start_date + datetime.timedelta(days=i)

            predictions.append({
                "Date": pred_date.strftime("%Y-%m-%d"),
                "Prediction": pred_label
            })

        df_pred = pd.DataFrame(predictions)
        st.table(df_pred)

st.markdown("---")

# --- Section 3: Show last 5 days actual weather ---
st.header("Last 5 Days Actual Weather Data")

last_5_days = df.tail(5)[['Date', 'Temperature', 'Humidity', 'Wind Speed', 'Rainy']].copy()
last_5_days['Rainy'] = last_5_days['Rainy'].apply(lambda x: "🌧️ Yes" if x == 1 else "☀️ No")

st.table(last_5_days.reset_index(drop=True))

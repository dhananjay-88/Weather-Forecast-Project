import sys
import numpy as np
import joblib

# Check if all 3 inputs are provided
if len(sys.argv) != 4:
    print("Usage: python predict_weather.py <Temperature> <Humidity> <Wind Speed>")
    sys.exit(1)

try:
    # Read inputs from command-line arguments
    temp = float(sys.argv[1])
    humidity = float(sys.argv[2])
    wind = float(sys.argv[3])
except ValueError:
    print("❌ Please provide valid numeric values for temperature, humidity, and wind speed.")
    sys.exit(1)

# Load the trained logistic regression model
model = joblib.load('weather_prediction_model.pkl')

# Create a feature array and predict
features = np.array([[temp, humidity, wind]])
prediction = model.predict(features)

# Output the result
if prediction[0] == 1:
    print("🌧️ Rainy")
else:
    print("☀️ Not Rainy")

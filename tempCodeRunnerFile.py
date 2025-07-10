import sys
import numpy as np
import joblib

# Read input from command-line
temp = float(sys.argv[1])
humidity = float(sys.argv[2])
wind = float(sys.argv[3])

# Load trained model
model = joblib.load('weather_prediction_model.pkl')

# Predict
prediction = model.predict([[temp, humidity, wind]])

# Output
if prediction[0] == 1:
    print("🌧️ Rainy")
else:
    print("☀️ Not Rainy")


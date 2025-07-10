import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
import joblib

# Load the 2-year weather dataset
df = pd.read_csv("weather_data_2 years.csv")

# Convert dates to day index
df["DayIndex"] = range(len(df))

# Train regression models to predict Temperature, Humidity, Wind Speed
X = df[["DayIndex"]]

temp_model = LinearRegression().fit(X, df["Temperature"])
humidity_model = LinearRegression().fit(X, df["Humidity"])
wind_model = LinearRegression().fit(X, df["Wind Speed"])

# Train classifier model for Rainy/Not Rainy
features = df[["Temperature", "Humidity", "Wind Speed"]]
target = df["Rainy"]
classifier = LogisticRegression(class_weight="balanced", max_iter=1000).fit(features, target)

# Save models
joblib.dump(temp_model, "temp_model.pkl")
joblib.dump(humidity_model, "humidity_model.pkl")
joblib.dump(wind_model, "wind_model.pkl")
joblib.dump(classifier, "weather_prediction_model.pkl")

print("✅ All models trained and saved successfully.")

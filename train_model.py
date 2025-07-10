import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the updated 2-year balanced dataset
df = pd.read_csv('weather_data_2 years.csv')

# Check and clean any missing values
df = df.dropna()

# Define Features and Target
X = df[['Temperature', 'Humidity', 'Wind Speed']].values
y = df['Rainy'].values

# Split into training and testing (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the logistic regression model with class balancing
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save trained model
joblib.dump(model, 'weather_prediction_model.pkl')
print("✅ Model saved as 'weather_prediction_model.pkl'")


import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import logging
import json

# Load the dataset
data = pd.read_csv("rainfall.csv")

# Input features and output
features = [
    "Temperature",
    "Humidity",
    "Pressure",
    "WindSpeed",
    "CloudCover"
]

X = data[features]

y = data["RainTomorrow"].str.strip()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create the ML model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

logging.info(f"Test accuracy: {accuracy * 100:.2f}%")

# Save the trained model
joblib.dump(model, "rainfall_model.pkl")

# Save metadata for reproducibility
import json
metadata = {
    "features": features,
    "model_params": model.get_params(),
    "classes": model.classes_.tolist()
}
with open("rainfall_model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("Model trained successfully!")
print("Saved as rainfall_model.pkl")
print("Metadata saved as rainfall_model_metadata.json")


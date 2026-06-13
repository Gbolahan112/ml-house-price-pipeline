from fastapi import FastAPI
import pandas as pd
import joblib

# Initialize app
app = FastAPI()

# Load trained model
model = joblib.load("models/rf_model.pkl")


# Home route
@app.get("/")
def home():
    return {"message": "House Price Prediction API is running"}


# Prediction route
@app.post("/predict")
def predict(data: dict):
    try:
        # Convert to DataFrame
        df = pd.DataFrame([data])

        # 🔥 Feature Engineering INSIDE API
        df["Rooms_per_Household"] = df["AveRooms"] / df["AveOccup"]
        df["Bedrooms_per_Household"] = df["AveBedrms"] / df["AveOccup"]
        df["Income_per_Person"] = df["MedInc"] / df["Population"]

        # Ensure correct column order (VERY IMPORTANT)
        expected_cols = [
            "MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude",
            "Rooms_per_Household", "Bedrooms_per_Household",
            "Income_per_Person"
        ]

        df = df[expected_cols]

        # Prediction
        prediction = model.predict(df)

        return {"prediction": float(prediction[0])}

    except Exception as e:
        return {"error": str(e)}
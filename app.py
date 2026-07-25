from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# ==========================================================
# FastAPI App
# ==========================================================

app = FastAPI(title="Online Shoppers Purchasing Intention Prediction API")

# ==========================================================
# Load Model & Scaler
# ==========================================================

model = joblib.load("best_xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

# ==========================================================
# Columns used for StandardScaler
# ==========================================================

num_cols = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType"
]

# ==========================================================
# Input Schema
# ==========================================================

class CustomerData(BaseModel):
    Administrative: int
    Administrative_Duration: float
    Informational: int
    Informational_Duration: float
    ProductRelated: int
    ProductRelated_Duration: float
    BounceRates: float
    ExitRates: float
    PageValues: float
    SpecialDay: float
    Month: int
    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int
    VisitorType: int
    Weekend: int

# ==========================================================
# Home
# ==========================================================

@app.get("/")
def home():
    return {
        "message": "Online Shoppers Purchasing Intention Prediction API is Running Successfully!"
    }

# ==========================================================
# Prediction
# ==========================================================

@app.post("/predict")
def predict(data: CustomerData):

    try:

        input_df = pd.DataFrame([{
            "Administrative": data.Administrative,
            "Administrative_Duration": data.Administrative_Duration,
            "Informational": data.Informational,
            "Informational_Duration": data.Informational_Duration,
            "ProductRelated": data.ProductRelated,
            "ProductRelated_Duration": data.ProductRelated_Duration,
            "BounceRates": data.BounceRates,
            "ExitRates": data.ExitRates,
            "PageValues": data.PageValues,
            "SpecialDay": data.SpecialDay,
            "Month": data.Month,
            "OperatingSystems": data.OperatingSystems,
            "Browser": data.Browser,
            "Region": data.Region,
            "TrafficType": data.TrafficType,
            "VisitorType": data.VisitorType,
            "Weekend": data.Weekend
        }])

        # Scale only numerical columns
        input_df[num_cols] = scaler.transform(input_df[num_cols])

        # Prediction
        prediction = int(model.predict(input_df)[0])

        # Probability
        probability = float(model.predict_proba(input_df)[0][1])

        return {
            "Prediction": prediction,
            "Status": "Purchase" if prediction == 1 else "No Purchase",
            "Purchase_Probability": round(probability, 4)
        }

    except Exception as e:
        return {
            "error": str(e)
        }
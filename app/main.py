from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from app.model import load_production_model

app = FastAPI(title="VoD Streaming Customer Churn Predictor App")

# Load the production random forest binary model on startup
model = load_production_model()

class CustomerFeaturePayload(BaseModel):
    days_since_last_login: float
    subscription_age_months: float
    monthly_watch_hours: float
    customer_support_tickets: float
    watchlist_size: float
    avg_completion_rate: float
    preferred_content_imdb: float
    preferred_content_meta: float
    has_no_watchlist: int        # ← ADD THIS (binary: 0 or 1)
    is_new_subscriber: int       # ← ADD THIS (binary: 0 or 1)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict_churn(payload: CustomerFeaturePayload):
    # Construct structured matrix array input matching original dimensions
    features_array = np.array([[
        payload.days_since_last_login,
        payload.subscription_age_months,
        payload.monthly_watch_hours,
        payload.customer_support_tickets,
        payload.watchlist_size,
        payload.avg_completion_rate,
        payload.preferred_content_imdb,
        payload.preferred_content_meta,
        payload.has_no_watchlist,     # ← ADD THIS
        payload.is_new_subscriber     # ← ADD THIS
    ]])
    
    prediction = int(model.predict(features_array)[0])
    probability = float(model.predict_proba(features_array)[0][1])
    
    # Match the retention strategy criteria from Step 7
    if prediction == 1:
        strategy = "Trigger Hyper-Personalized Movie Re-engagement Email Campaign"
    else:
        strategy = "Maintain Standard Account Cycle"
        
    return {
        "churned": bool(prediction),
        "churn_probability": round(probability, 3),
        "business_intervention_strategy": strategy
    }
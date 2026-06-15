from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from app.model import load_production_model, get_user_by_id, generate_recommendations

app = FastAPI(title="VoD Streaming Customer Churn Predictor App")

# Load the production Random Forest model ONCE at startup (not per request)
model = load_production_model()


class CustomerFeaturePayload(BaseModel):
    days_since_last_login: float
    logins_per_week: float
    monthly_watch_hours: float
    customer_support_tickets: float
    watch_hours_per_month: float
    ticket_burden: float
    avg_completion_rate: float
    preferred_content_imdb: float
    has_no_watchlist: int        # binary 0/1
    is_new_subscriber: int       # binary 0/1


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/features")
def expected_features():
    """Documentation endpoint: the exact field order /predict expects."""
    return {"expected_features": list(CustomerFeaturePayload.model_fields.keys())}


@app.post("/predict")
def predict_churn(payload: CustomerFeaturePayload):
    # Single-row DataFrame whose columns match the model's training feature names
    features_df = pd.DataFrame([{
        "days_since_last_login": payload.days_since_last_login,
        "logins_per_week": payload.logins_per_week,
        "monthly_watch_hours": payload.monthly_watch_hours,
        "customer_support_tickets": payload.customer_support_tickets,
        "watch_hours_per_month": payload.watch_hours_per_month,
        "ticket_burden": payload.ticket_burden,
        "avg_completion_rate": payload.avg_completion_rate,
        "preferred_content_imdb": payload.preferred_content_imdb,
        "has_no_watchlist": payload.has_no_watchlist,
        "is_new_subscriber": payload.is_new_subscriber,
    }])

    prediction = int(model.predict(features_df)[0])
    probability = float(model.predict_proba(features_df)[0][1])

    if probability >= 0.70:
        strategy = "High risk: trigger hyper-personalized re-engagement email campaign"
    elif probability >= 0.40:
        strategy = "Medium risk: route to premium support queue to reduce friction"
    else:
        strategy = "Low risk: maintain standard account cycle"

    return {
        "churned": bool(prediction),
        "churn_probability": round(probability, 3),
        "business_intervention_strategy": strategy,
    }


FEATURE_KEYS = [
    "days_since_last_login", "logins_per_week", "monthly_watch_hours",
    "customer_support_tickets", "watch_hours_per_month", "ticket_burden",
    "avg_completion_rate", "preferred_content_imdb",
    "has_no_watchlist", "is_new_subscriber",
]


@app.get("/recommend/{user_id}")
def recommend_for_user(user_id: int, n: int = 5):
    user_data = get_user_by_id(user_id)
    if user_data is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    row = pd.DataFrame([{k: user_data[k] for k in FEATURE_KEYS}])

    probability = float(model.predict_proba(row)[0][1])

    if probability < 0.5:
        return {
            "user_id": user_id,
            "churn_probability": round(probability, 3),
            "at_risk": False,
            "message": "User is not at risk. No recommendations triggered.",
            "recommendations": [],
        }

    recommendations = generate_recommendations(user_data, n=n)

    return {
        "user_id": user_id,
        "churn_probability": round(probability, 3),
        "at_risk": True,
        "recommendations": recommendations,
    }

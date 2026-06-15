import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

# Temporary absolute system alignment for local VS Code context
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app.scraper import fetch_omdb_raw_data
from app.features import generate_features_and_labels

MODEL_PATH = os.path.join(current_dir, "model.pkl")
CSV_PATH = os.path.join(parent_dir, "data", "raw", "streaming_churn_data.csv")
RAW_DATA_CACHE = []


def get_movie_pool():
    global RAW_DATA_CACHE
    if not RAW_DATA_CACHE:
        RAW_DATA_CACHE = fetch_omdb_raw_data()
    return RAW_DATA_CACHE


def generate_recommendations(user_features: dict, n: int = 5) -> list:
    movie_pool = get_movie_pool()
    preferred_imdb = user_features.get("preferred_content_imdb", 7.0)

    scored = []
    for movie in movie_pool:
        imdb = float(movie.get("imdbRating", 0)) if movie.get("imdbRating") != "N/A" else 0.0
        meta = float(movie.get("Metascore", 0)) if movie.get("Metascore") != "N/A" else 0.0
        match_score = (10 - abs(imdb - preferred_imdb)) * 0.6 + (meta / 100) * 0.4
        scored.append({
            "title": movie.get("Title", "Unknown"),
            "imdb_rating": imdb,
            "metascore": meta,
            "match_score": round(match_score, 3),
        })

    scored.sort(key=lambda x: x["match_score"], reverse=True)
    return scored[:n]


def load_user_data() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        train_and_save_production_model()
    return pd.read_csv(CSV_PATH)


def get_user_by_id(user_id: int) -> dict | None:
    df = load_user_data()
    if user_id < 0 or user_id >= len(df):
        return None
    return df.iloc[user_id].to_dict()


def train_and_save_production_model():
    """ Runs the complete pipeline from scratch and serializes the model file """
    print("Step 1: Scrape raw data components from OMDb API context...")
    raw_data = fetch_omdb_raw_data()

    print("Step 2: Engineering user behavioral feature matrices...")
    data_matrix = generate_features_and_labels(raw_data)

    save_dir = os.path.join(parent_dir, "data", "raw")
    os.makedirs(save_dir, exist_ok=True)

    csv_file_path = os.path.join(save_dir, "streaming_churn_data.csv")
    data_matrix.to_csv(csv_file_path, index=False)
    print(f"Dataset written to: {csv_file_path}")

    X = data_matrix.drop(columns=["churn"])
    y = data_matrix["churn"]

    print("Step 3: Training Random Forest Engine...")
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    print(f"Serialization complete! Model saved to: {MODEL_PATH}")

    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "f1": make_scorer(f1_score),
    }
    cv_scores = cross_validate(model, X, y, cv=5, scoring=scoring)
    print("\n5-Fold Cross-Validation Results:")
    print(f"  Accuracy:  {cv_scores['test_accuracy'].mean():.3f} (+/- {cv_scores['test_accuracy'].std():.3f})")
    print(f"  Precision: {cv_scores['test_precision'].mean():.3f} (+/- {cv_scores['test_precision'].std():.3f})")
    print(f"  Recall:    {cv_scores['test_recall'].mean():.3f} (+/- {cv_scores['test_recall'].std():.3f})")
    print(f"  F1:        {cv_scores['test_f1'].mean():.3f} (+/- {cv_scores['test_f1'].std():.3f})")


def load_production_model():
    """ Dynamic recovery load pattern for FastAPI startup processes """
    if not os.path.exists(MODEL_PATH):
        print("Model pkl missing! Running automated fallback build...")
        train_and_save_production_model()
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    train_and_save_production_model()

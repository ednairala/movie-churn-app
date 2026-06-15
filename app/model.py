import os
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Temporary absolute system alignment for local VS Code context
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app.scraper import fetch_omdb_raw_data
from app.features import generate_features_and_labels

MODEL_PATH = os.path.join(current_dir, "model.pkl")

def train_and_save_production_model():
    """ Runs the complete pipeline from scratch and serializes the model file """
    print("🎬 Step 1: Scrape raw data components from OMDb API context...")
    raw_data = fetch_omdb_raw_data()
    
    print("📊 Step 2: Engineering user behavioral feature matrices...")
    data_matrix = generate_features_and_labels(raw_data)
    
    # Target path: project_root/data/raw/
    save_dir = os.path.join(parent_dir, "data", "raw")
    os.makedirs(save_dir, exist_ok=True)
    
    csv_file_path = os.path.join(save_dir, "streaming_churn_data.csv")
    data_matrix.to_csv(csv_file_path, index=False)
    print(f"📁 Dataset written directly to target: {csv_file_path}")
    
    X = data_matrix.drop(columns=["churn"])
    y = data_matrix["churn"]
    
    print("🤖 Step 3: Training Random Forest Engine...")
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X, y)
    
    joblib.dump(model, MODEL_PATH)
    print(f"🚀 Serialization complete! Production model saved to: {MODEL_PATH}")

def load_production_model():
    """ Dynamic recovery load pattern for FastAPI startup processes """
    if not os.path.exists(MODEL_PATH):
        print("⚠️ Model pkl missing! Running automated fallback build...")
        train_and_save_production_model()
    return joblib.load(MODEL_PATH)

if __name__ == "__main__":
    train_and_save_production_model()

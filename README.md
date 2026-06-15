# Customer Churn Predictor App

A Dockerized FastAPI application that predicts customer churn for a Video-on-Demand streaming service using OMDb API data.

## Project Structure

```
movie-churn-app/
├── app/
│   ├── main.py          # FastAPI app with /predict, /recommend, /health, /features endpoints
│   ├── model.py         # Model training, cross-validation, recommendation logic
│   ├── features.py      # Feature engineering (8+ features across 4 types)
│   └── scraper.py       # OMDb API data fetcher with rate limiting
├── notebooks/
│   └── eda_and_selection.ipynb   # EDA, 4 feature selection methods, PCA, network graph, model comparison
├── data/
│   └── raw/             # Raw CSV data (auto-generated; directory tracked, large files gitignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── REPORT.md            # Full analysis report
```

## Quick Start

```bash
docker-compose up
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/features` | GET | List expected input fields |
| `/predict` | POST | Churn prediction (churned + probability) |
| `/recommend/{user_id}` | GET | Recommendations for at-risk users (prob >= 0.5) |

## Notebook

The Jupyter notebook (`notebooks/eda_and_selection.ipynb`) contains:
- 4 feature selection methods (Filter, RFE, Decision Tree, Random Forest) with comparison table
- PCA elbow plot + SVD 2D scatter colored by churn label
- Network graph with centrality features
- Model comparison: original features vs PCA components vs original + network

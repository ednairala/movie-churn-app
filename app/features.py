import numpy as np
import pandas as pd

def generate_features_and_labels(raw_movie_pool: list, num_samples: int = 1000) -> pd.DataFrame:
    """ Transform raw content data vectors into robust custom behavioral features """
    np.random.seed(42)
    records = []
    
    # Extract baseline profiles from scraper list
    processed_movies = []
    for item in raw_movie_pool:
        imdb = float(item.get("imdbRating", 7.0)) if item.get("imdbRating") != "N/A" else 7.0
        meta = float(item.get("Metascore", 60.0)) if item.get("Metascore") != "N/A" else 60.0
        processed_movies.append({"imdb": imdb, "meta": meta})

    for _ in range(num_samples):
        m = np.random.choice(processed_movies)
        
        # 1 & 2: TIME-BASED FEATURES
        days_since_last_login = np.random.randint(1, 60)
        subscription_age_months = np.random.randint(1, 36)
        
        # 3 & 4: AGGREGATION FEATURES
        monthly_watch_hours = np.random.uniform(5.0, 120.0)
        customer_support_tickets = np.random.randint(0, 5)
        
        # 5 & 6: RATIO FEATURES
        watchlist_size = np.random.randint(0, 50)
        avg_completion_rate = np.random.uniform(0.1, 1.0)
        
        # 7 & 8: OMDb DERIVED METRICS
        preferred_content_imdb = m["imdb"] + np.random.normal(0, 0.5)
        preferred_content_meta = m["meta"] + np.random.normal(0, 5.0)

        # 9 & 10: BINARY FEATURES  ← now inside the loop, after variables are defined
        has_no_watchlist = 1 if watchlist_size == 0 else 0
        is_new_subscriber = 1 if subscription_age_months < 3 else 0
        
        # Rule-based synthetic churn baseline thresholding engine
        churn_score = (days_since_last_login * 1.5) - (monthly_watch_hours * 0.2) + (customer_support_tickets * 2.0)
        churn_label = 1 if churn_score > 25 else 0
        
        records.append({
            "days_since_last_login": days_since_last_login,
            "subscription_age_months": subscription_age_months,
            "monthly_watch_hours": monthly_watch_hours,
            "customer_support_tickets": customer_support_tickets,
            "watchlist_size": watchlist_size,
            "avg_completion_rate": avg_completion_rate,
            "preferred_content_imdb": preferred_content_imdb,
            "preferred_content_meta": preferred_content_meta,
            "has_no_watchlist": has_no_watchlist,
            "is_new_subscriber": is_new_subscriber,
            "churn": churn_label
        })
        
    df = pd.DataFrame(records)
    
    # Force exact 50/50 balance matching the class criteria
    df_0 = df[df['churn'] == 0]
    df_1 = df[df['churn'] == 1]
    min_size = min(len(df_0), len(df_1))
    
    balanced_df = pd.concat([
        df_0.sample(min_size, random_state=42), 
        df_1.sample(min_size, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    return balanced_df
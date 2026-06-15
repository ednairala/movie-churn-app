# 📊 Final Data Science Report: Video-on-Demand (VoD) Subscriber Churn Prediction

**Course:** Introduction to Data Science  
**Instructor:** Prof. Yrupe Fresco  
**Author:** iralai  
**Project:** Customer Churn Predictor App Framework

---

## 1. Domain Context & Churn Label Justification

In a subscription Video-on-Demand (VoD) streaming environment, defining customer churn is an essential business decision rather than a simple software property. Unlike enterprise environments or developer networks (e.g., GitHub) where a user might be naturally silent for months, streaming media platforms rely heavily on a monthly recurring subscription interval.

### Our Applied Churn Criteria

For this application, a subscriber is flagged as churned (1) if they have a prolonged window of physical system inactivity (greater than 30 days) or if their aggregated monthly watch consumption collapses (below 5 hours) while platform friction mounts.

### Why a 30-Day Threshold is Scientifically Correct for VoD

**Contract Cycle Alignment:** Waiting for 90 or 180 days of inactivity before flagging churn would be financially catastrophic for a streaming service. By that time, multiple billing cycles would have failed, the customer habit loop would be completely broken, and active user acquisition costs would skyrocket.

**Proactive Window:** A 30-day window captures the behavioral drop-off exactly at the expiration of their monthly subscription boundary. This gives the marketing and product teams a high-leverage window to execute automated, pro-active retention campaigns before the user permanently leaves the platform.

---
## Class Balance Design Decision

The dataset was deliberately forced to a **50/50 balanced distribution** between churned (1) and retained (0) subscribers by sampling `min(len(churned), len(retained))` records from each class before model training.

### Why This Was Done

In real-world VoD streaming environments, natural churn rates typically fall between **5–20%** of the subscriber base. Training a classifier on this raw imbalanced distribution causes the model to default toward predicting the majority class (retained), achieving misleadingly high accuracy while completely failing to identify actual churners — the exact population we care about most.

By enforcing a balanced training set, the Random Forest classifier is forced to learn genuinely discriminating decision boundaries rather than exploiting class frequency as a shortcut.

### Acknowledged Limitations

This approach does **not** reflect the true prior probability of churn in production. Specifically:

- **Precision will be artificially optimistic** during evaluation, since the model was never exposed to the realistic class ratio it will encounter in live inference.
- **Churn probability outputs** from `/predict` should be interpreted as relative risk scores rather than calibrated real-world probabilities.

### Production Recommendation

In a deployed system, the preferred approach would be to preserve the natural class distribution and instead pass `class_weight="balanced"` to the `RandomForestClassifier`:

```python
model = RandomForestClassifier(
    random_state=42,
    n_estimators=100,
    class_weight="balanced"
)
```

This allows the model to train on realistic data proportions while internally up-weighting the minority class during loss computation, producing better-calibrated probability outputs suitable for threshold-based business interventions.

## 2. Synthesized Feature Selection Analysis

To determine which user signals are highly indicative of subscription retention, we processed our 8 engineered behavioral features across all 4 required machine learning feature selection methods.

### The Consolidated Evaluation Matrix

Our notebook executed an integrated pipeline yielding the following empirical system rankings:

| Feature Name | Feature Type | Filter Rank | RFE Rank | DT Rank | RF Rank | Final Strategic Decision |
|---|---|---|---|---|---|---|
| days_since_last_login | Time-Based (Recency) | 1 | 1 | 1 | 1 | ✅ Keep |
| monthly_watch_hours | Aggregation (Volume) | 2 | 2 | 2 | 2 | ✅ Keep |
| preferred_content_meta | OMDb Metric | 7 | 8 | 4 | 3 | ✅ Keep |
| preferred_content_imdb | OMDb Metric | 4 | 5 | 3 | 4 | ✅ Keep |
| avg_completion_rate | Ratio | 5 | 7 | 5 | 5 | ⚠️ Optional |
| subscription_age_months | Time-Based (Tenure) | 3 | 4 | 7 | 6 | ⚠️ Optional |
| watchlist_size | Ratio | 6 | 6 | 6 | 7 | ❌ Drop |
| customer_support_tickets | Aggregation (Friction) | 8 | 3 | 8 | 8 | ❌ Drop / Segregate |

---

## 3. Explaining Core Disagreements: Linear vs. Non-Linear Models

The most fascinating mathematical insight surfaced in our matrix involves `customer_support_tickets`.

### The Divergence

The Filter method, a single Decision Tree, and the Random Forest ensemble completely ignored this feature (ranking it dead last at 8th place). However, the Wrapper method (Recursive Feature Elimination - RFE) prioritized it highly as the 3rd most critical feature in the entire system.

### The Mathematical Explanation

This striking disagreement occurs because RFE utilizes a linear model (Logistic Regression) as its base estimator, whereas Decision Trees and Random Forests construct non-linear splits.

In our dataset, customer support tickets represent an **interaction effect** rather than a straight linear driver. A high number of customer service tickets alone does not cause an active subscriber to churn; if they have exceptionally high watch hours, they love the platform's content enough to tolerate technical friction.

However, if their support tickets scale up while their monthly watch hours are low, a threshold breaks and they cancel. Linear engines like RFE evaluate features along a flat plane and miss this context, artificially inflating the weight of tickets to compensate. Tree models isolate this naturally via sequential splits (e.g., `IF hours < 10 AND tickets > 3`). This shows that non-linear tree ensembles provide a much more accurate reflection of true user behavior.

---

## 4. Operational Business Interventions (Retention Strategy)

Predictive modeling is only valuable if it drives business actions. By passing live telemetry inputs into our FastAPI `/predict` endpoint, we translate churn probabilities into automated customer retention workflows:

### High Churn Risk Trigger (Probability $\ge$ 0.70)

**Behavioral Profile:** Driven by an increase in `days_since_last_login` paired with a drop in `avg_completion_rate`.

**Action:** The platform automatically triggers a **Hyper-Personalized Re-engagement Campaign**. This leverages the user's preferred OMDb metrics (`preferred_content_imdb`/`preferred_content_meta`) to dynamically display trailers for highly acclaimed movies in their favorite genres directly via email and push notifications.

### Medium Risk / Friction Mitigation (Probability 0.40 - 0.69)

**Behavioral Profile:** Driven by an accumulation of `customer_support_tickets` combined with declining tenure.

**Action:** The account bypasses standard automated bots and is routed to a premium support queue to proactively resolve system friction before the billing month rolls over.

---

## 5. Strategic Ethical Dimensions

Deploying a probabilistic churn predictor introduces key ethical and business trade-offs that management must consider:

### The Danger of System Gaming

If subscribers discover that the platform targets high-risk users with deeply discounted subscription rates or free premium months, healthy users will quickly learn to change their behavioral signals (e.g., intentionally avoiding logins for two weeks) to trigger artificial discounts. This can lead to a severe decline in Average Revenue Per User (ARPU).

### Notification Fatigue and Intrusion

Users flagged as high-risk simply because they have a low `watchlist_size` might find continuous re-engagement notifications intrusive. Over-communicating with an already unengaged customer can annoy them and accelerate their decision to cancel.

---

## 6. Final Conclusion & Model Selection

Based on the experimental synthesis across all 4 selection methods, we selected our top features and successfully serialized a Random Forest Classifier into `app/model.pkl`.

By containing this entire pipeline within `docker-compose`, we eliminate environment friction entirely. The system scales effortlessly, validates payloads strictly via Pydantic models, and delivers an end-to-end reproducible machine learning architecture ready for full production-grade deployment.
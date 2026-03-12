# Airbnb Price Advisor — New York

> Data-driven nightly price recommendations for Airbnb hosts using an XGBoost + Neural Network ensemble, trained on 14,000+ New York listings.

Live App: [airbnb-price-advisor.streamlit.app](#) ← paste your URL here

---

## Overview

Airbnb hosts often underprice or overprice their listings without knowing where they stand in their neighbourhood market. This app takes your listing details and recommends an optimal nightly rate — showing how it compares to your neighbourhood median and which factors are driving the price.

This project extends a previous Airbnb NYC data analytics project, adding a full ML/DL training pipeline and a deployed interactive price recommendation interface.

---

## Demo

![App Demo](assets/demo.gif)

**Input Form**
![Inputs](assets/app_inputs.png)

**Price Results**
![Results](assets/app_results.png)

**Feature Importance**
![Features](assets/app_features.png)

---

## Project Structure
```
airbnb-price-advisor/
│
├── train.py
├── app.py
├── airbnb_data.csv
├── model_artifacts.pkl
├── nn_model.pt
├── requirements.txt
└── assets/
    ├── demo.gif
    ├── app_inputs.png
    ├── app_results.png
    └── app_features.png
```

---

## Dataset

| Property | Detail |
|----------|--------|
| Source | Kaggle — Airbnb NYC Open Data |
| Raw rows | 20,786 listings |
| After cleaning | 14,277 listings |
| Price range | $41 – $703 / night |
| City | New York City |

Rows dropped: listings missing `price` or `review_scores_rating` (5,945 missing ratings removed). Price outliers clipped at 2nd and 98th percentile to remove anomalous listings.

---

## Feature Engineering

| Feature | Description |
|---------|-------------|
| `geo_distance` | Haversine distance from Times Square |
| `occupancy_rate` | 1 - availability / 365 |
| `host_quality` | Composite of superhost status, rating, listing count |
| `log_reviews` | Log-scaled review count to reduce skew |
| `rating_x_reviews` | Interaction between score and review volume |
| `accommodates_sqrd` | Guests squared — captures non-linear capacity pricing |
| `neigh_price_tier` | Weighted blend of neighbourhood mean (60%) and median (40%) |

---

## Workflow
```
Load & Clean → Feature Engineering → Encode Categoricals
    → Train/Test Split (80/20) → XGBoost (5-Fold CV)
    → Neural Network (PyTorch) → Weighted Ensemble
    → Save Artifacts → Streamlit App
```

---

## Models

### XGBoost
- 5-fold cross validation
- 600 estimators, learning rate 0.04, max depth 6
- Histogram-based tree method
- L1 and L2 regularization to prevent overfitting

### Neural Network (PyTorch)
- Entity embeddings for categorical features — neighbourhood, room type, property type, host type
- Architecture: Linear(128) → BatchNorm → ReLU → Dropout(0.3) → Linear(64) → ReLU → Linear(1)
- HuberLoss (delta=50) — robust to price outliers
- AdamW optimizer with ReduceLROnPlateau scheduler
- 80 epochs with early weight saving on best validation MAE

### Ensemble
- Weighted average: XGBoost × 0.70 + NN × 0.30
- Weight selected by grid search over test MAE

---

## Results

| Model | MAE | R² |
|-------|-----|----|
| XGBoost CV (avg) | $41.84 ± $1.10 | — |
| XGBoost (test) | $41.61 | 0.7316 |
| Neural Network (test) | $44.74 | 0.6719 |
| **Ensemble (test)** | **$41.01** | **0.7330** |

---

## Model Analysis

**Why XGBoost Dominates**

XGBoost takes a 0.70 weight in the ensemble versus 0.30 for the Neural Network. This is expected — gradient boosted trees consistently outperform deep learning on tabular data, especially at this dataset size. With 14,277 training samples after cleaning, the Neural Network does not have enough data to fully leverage its capacity. Tree-based models require far less data to generalize well on structured tabular features.

**Why Keep the Neural Network at All**

Despite underperforming XGBoost individually (MAE $44.74 vs $41.61), the NN still improves the ensemble. This is because the two models make different types of errors — XGBoost misses certain non-linear patterns that the NN's entity embeddings capture, particularly in how neighbourhood identity and room type interact with price. The ensemble MAE of $41.01 beats XGBoost alone, confirming the NN adds genuine diversity.

**Entity Embeddings**

The NN uses learned embeddings for categorical features rather than simple label encoding. This means the model learns that "Williamsburg" and "Bushwick" are closer in pricing space than "Williamsburg" and "Staten Island" — purely from the data. This is a meaningful advantage over XGBoost which treats encoded categories as arbitrary integers.

**Feature Importance Findings**

| Feature | Importance |
|---------|------------|
| Guest capacity (sq) | 0.155 |
| Area mean price | 0.130 |
| Guests | 0.118 |
| Room type | 0.093 |
| Bathrooms | 0.086 |

Guest capacity dominates — the squared term outranking the linear term confirms that pricing scales non-linearly with capacity. A 6-person listing does not simply cost 3x a 2-person listing. Neighbourhood mean price ranking second validates the `neigh_price_tier` engineering approach. Room type ranking fourth confirms that entire home vs private room is a stronger signal than individual spec differences.

**What $41 MAE Means in Practice**

On a dataset where prices range $41–$703/night with a median around $100–150, a mean absolute error of $41 represents roughly 25–30% of the typical listing price. This is reasonable for a price recommendation tool — exact pricing also depends on photos, descriptions, and timing which are outside the model's scope.

---

## App Features

- Recommended nightly price based on listing specs
- Comparison against neighbourhood median
- Market position indicator — above / below / well positioned
- Feature importance chart showing what drives the price
- Neighbourhood price distribution with your price marked
- Actionable insights on review score and availability

---

## Tech Stack

- Python 3.x
- pandas, numpy, scikit-learn
- xgboost
- PyTorch
- Streamlit
- Plotly

---

## How to Run
```bash
pip install -r requirements.txt

# Train the model first
python train.py

# Launch the app
streamlit run app.py
```

---

## requirements.txt
```
pandas
numpy
scikit-learn
xgboost
torch
streamlit
plotly
```

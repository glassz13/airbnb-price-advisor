"""
Airbnb Price Recommendation Engine — Training Script
Run: python train.py
Output: model_artifacts.pkl  +  nn_model.pt
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import shap

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

CSV_PATH = "airbnb_data.csv"   # ← change to your filename

# ─────────────────────────────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  Airbnb Price Recommendation Engine — Training")
print("=" * 55)

df = pd.read_csv(CSV_PATH)
print(f"\n✓ Loaded  : {df.shape[0]:,} rows  |  {df.shape[1]} columns")

nulls = df.isnull().sum()
if nulls.sum() > 0:
    print(f"\nMissing values found:")
    print(nulls[nulls > 0].to_string())

df = df.dropna(subset=["price", "review_scores_rating"])
df["reviews_per_month"]  = df["reviews_per_month"].fillna(0)
df["bedrooms"]           = df["bedrooms"].fillna(df["bedrooms"].median())
df["bathrooms"]          = df["bathrooms"].fillna(df["bathrooms"].median())
df["beds"]               = df["beds"].fillna(df["beds"].median())
df["host_is_superhost"]  = (
    df["host_is_superhost"]
    .map({True: 1, False: 0, "TRUE": 1, "FALSE": 0, "t": 1, "f": 0})
    .fillna(0).astype(int)
)

lo, hi = df["price"].quantile([0.02, 0.98])
df = df[(df["price"] >= lo) & (df["price"] <= hi)].reset_index(drop=True)
print(f"✓ Cleaned : {df.shape[0]:,} listings  |  Price ${lo:.0f}–${hi:.0f}/night")


# ─────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────
def haversine(lat, lon, c_lat=40.7580, c_lon=-73.9855):
    R = 6371
    lat, lon     = np.radians(lat), np.radians(lon)
    c_lat, c_lon = np.radians(c_lat), np.radians(c_lon)
    a = (np.sin((c_lat - lat) / 2) ** 2 +
         np.cos(lat) * np.cos(c_lat) * np.sin((c_lon - lon) / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))

df["geo_distance"]      = haversine(df["latitude"], df["longitude"])
df["occupancy_rate"]    = 1 - df["availability_365"] / 365
df["host_quality"]      = (
    df["host_is_superhost"] * 2 +
    df["review_scores_rating"] / 5 +
    np.log1p(df["host_total_listings_count"]) / 3
)
df["log_reviews"]       = np.log1p(df["number_of_reviews"])
df["rating_x_reviews"]  = df["review_scores_rating"] * df["log_reviews"]
df["accommodates_sqrd"] = df["accommodates"] ** 2

neigh_stats = (
    df.groupby("neighbourhood_cleansed")["price"]
    .agg(neigh_median="median", neigh_mean="mean", neigh_count="count")
    .reset_index()
)
neigh_geo = (
    df.groupby("neighbourhood_cleansed")[["latitude", "longitude"]]
    .mean()
    .reset_index()
    .rename(columns={"latitude": "neigh_lat", "longitude": "neigh_lon"})
)
neigh_stats = neigh_stats.merge(neigh_geo, on="neighbourhood_cleansed")
df = df.merge(neigh_stats[["neighbourhood_cleansed", "neigh_median", "neigh_mean"]],
              on="neighbourhood_cleansed", how="left")

df["neigh_price_tier"]  = df["neigh_mean"] * 0.6 + df["neigh_median"] * 0.4

print(f"✓ Engineered features")


# ─────────────────────────────────────────────────────────────────────
# 3. ENCODE CATEGORICALS
# ─────────────────────────────────────────────────────────────────────
CAT_COLS = ["neighbourhood_cleansed", "room_type", "property_type", "host_type"]
label_encoders = {}

for col in CAT_COLS:
    le = LabelEncoder()
    df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

print(f"✓ Encoded  : {len(CAT_COLS)} categorical columns")


# ─────────────────────────────────────────────────────────────────────
# 4. FEATURES & SPLIT
# ─────────────────────────────────────────────────────────────────────
NUM_FEATURES = [
    "host_total_listings_count", "accommodates", "accommodates_sqrd",
    "bedrooms", "beds", "bathrooms", "minimum_nights",
    "availability_365", "number_of_reviews", "log_reviews",
    "reviews_per_month", "review_scores_rating", "rating_x_reviews",
    "host_quality", "geo_distance", "occupancy_rate",
    "neigh_price_tier", "neigh_median", "neigh_mean",
]
CAT_ENC      = [f"{c}_enc" for c in CAT_COLS]
ALL_FEATURES = NUM_FEATURES + CAT_ENC

X = df[ALL_FEATURES].fillna(0)
y = df["price"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = X_train.copy()
X_test_s  = X_test.copy()
X_train_s[NUM_FEATURES] = scaler.fit_transform(X_train[NUM_FEATURES])
X_test_s[NUM_FEATURES]  = scaler.transform(X_test[NUM_FEATURES])

print(f"✓ Split    : {len(X_train):,} train  |  {len(X_test):,} test")


# ─────────────────────────────────────────────────────────────────────
# 5. XGBOOST — 5-FOLD CV
# ─────────────────────────────────────────────────────────────────────
print("\n── XGBoost  5-Fold CV ──────────────────────────────────")

XGB_PARAMS = dict(
    n_estimators=600, learning_rate=0.04, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, tree_method="hist",
)

kf     = KFold(n_splits=5, shuffle=True, random_state=42)
cv_maes = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), 1):
    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(X_train.iloc[tr_idx], y_train[tr_idx],
          eval_set=[(X_train.iloc[val_idx], y_train[val_idx])], verbose=False)
    mae = mean_absolute_error(y_train[val_idx], m.predict(X_train.iloc[val_idx]))
    cv_maes.append(mae)
    print(f"  Fold {fold}  MAE = ${mae:.2f}")

print(f"\n  CV  MAE : ${np.mean(cv_maes):.2f}  ±  ${np.std(cv_maes):.2f}")

xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
xgb_pred  = xgb_model.predict(X_test)
print(f"\n  Test MAE : ${mean_absolute_error(y_test, xgb_pred):.2f}")
print(f"  Test R²  : {r2_score(y_test, xgb_pred):.4f}")


# ─────────────────────────────────────────────────────────────────────
# 6. NEURAL NETWORK
# ─────────────────────────────────────────────────────────────────────
print("\n── Neural Network ──────────────────────────────────────")

class AirbnbPriceNet(nn.Module):
    def __init__(self, n_num, embed_sizes):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(n+1, d) for n, d in embed_sizes])
        total = n_num + sum(d for _, d in embed_sizes)
        self.net = nn.Sequential(
            nn.Linear(total, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),   nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),    nn.ReLU(), nn.Linear(64, 1),
        )
    def forward(self, x_num, x_cat):
        embs = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        return self.net(torch.cat([x_num] + embs, dim=1)).squeeze()

embed_sizes = [
    (len(label_encoders[c].classes_), min(50, len(label_encoders[c].classes_) // 2 + 1))
    for c in CAT_COLS
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device : {DEVICE}")

def to_tensors(X_df, y_arr=None):
    t_num = torch.FloatTensor(X_df[NUM_FEATURES].values).to(DEVICE)
    t_cat = torch.LongTensor(X_df[CAT_ENC].values).to(DEVICE)
    if y_arr is not None:
        return TensorDataset(t_num, t_cat, torch.FloatTensor(y_arr).to(DEVICE))
    return t_num, t_cat

train_loader = DataLoader(to_tensors(X_train_s, y_train), batch_size=256, shuffle=True,  drop_last=True)
test_loader  = DataLoader(to_tensors(X_test_s,  y_test),  batch_size=256, shuffle=False)

nn_model  = AirbnbPriceNet(len(NUM_FEATURES), embed_sizes).to(DEVICE)
optimizer = torch.optim.AdamW(nn_model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
criterion = nn.HuberLoss(delta=50.0)

best_mae, best_state = float("inf"), None
EPOCHS = 100

for epoch in range(1, EPOCHS + 1):
    nn_model.train()
    for x_num, x_cat, yb in train_loader:
        optimizer.zero_grad()
        criterion(nn_model(x_num, x_cat), yb).backward()
        optimizer.step()

    nn_model.eval()
    preds_ep = []
    with torch.no_grad():
        for x_num, x_cat, yb in test_loader:
            preds_ep.extend(nn_model(x_num, x_cat).cpu().numpy())

    val_mae = mean_absolute_error(y_test, preds_ep)
    scheduler.step(val_mae)

    if val_mae < best_mae:
        best_mae  = val_mae
        best_state = {k: v.clone() for k, v in nn_model.state_dict().items()}

    if epoch % 10 == 0:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  MAE = ${val_mae:.2f}")

nn_model.load_state_dict(best_state)
nn_pred = np.array(preds_ep)
print(f"\n  Best NN  MAE : ${best_mae:.2f}")
print(f"  NN       R²  : {r2_score(y_test, nn_pred):.4f}")


# ─────────────────────────────────────────────────────────────────────
# 7. ENSEMBLE
# ─────────────────────────────────────────────────────────────────────
print("\n── Ensemble ────────────────────────────────────────────")

best_w, best_ens_mae = 0.5, float("inf")
for w in np.arange(0.60, 0.90, 0.05):
    ens = w * xgb_pred + (1 - w) * nn_pred
    m   = mean_absolute_error(y_test, ens)
    if m < best_ens_mae:
        best_ens_mae, best_w = m, w

ens_pred = best_w * xgb_pred + (1 - best_w) * nn_pred
ens_r2   = r2_score(y_test, ens_pred)
print(f"  XGBoost weight : {best_w:.2f}  |  NN weight : {1-best_w:.2f}")
print(f"  Ensemble MAE   : ${best_ens_mae:.2f}")
print(f"  Ensemble R²    : {ens_r2:.4f}")


# ─────────────────────────────────────────────────────────────────────
# 8. SHAP + FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────
print("\n── SHAP Values ─────────────────────────────────────────")
explainer        = shap.TreeExplainer(xgb_model)
shap_sample      = X_test.head(300)
shap_values      = explainer.shap_values(shap_sample)
mean_shap        = np.abs(shap_values).mean(axis=0)
feature_importance = dict(zip(ALL_FEATURES, mean_shap))

print("\n── Feature Importance (Mean |SHAP|) ────────────────────")
sorted_fi = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
for feat, val in sorted_fi:
    bar = "█" * int(val / max(feature_importance.values()) * 30)
    print(f"  {feat:<35} ${val:6.2f}  {bar}")


# ─────────────────────────────────────────────────────────────────────
# 9. SAVE
# ─────────────────────────────────────────────────────────────────────
df_clean = df[["neighbourhood_cleansed", "room_type", "property_type",
               "host_type", "price", "neigh_median", "accommodates",
               "bedrooms", "review_scores_rating"]].copy()

artifacts = {
    "xgb_model"          : xgb_model,
    "xgb_weight"         : best_w,
    "label_encoders"     : label_encoders,
    "scaler"             : scaler,
    "neigh_stats"        : neigh_stats,
    "df_clean"           : df_clean,
    "num_features"       : NUM_FEATURES,
    "cat_enc_features"   : CAT_ENC,
    "all_features"       : ALL_FEATURES,
    "cat_cols"           : CAT_COLS,
    "embed_sizes"        : embed_sizes,
    "shap_explainer"     : explainer,
    "feature_importance" : feature_importance,
    "cv_mae"             : np.mean(cv_maes),
    "test_mae"           : best_ens_mae,
    "test_r2"            : ens_r2,
    "n_train"            : len(X_train),
}

with open("model_artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)

torch.save(best_state, "nn_model.pt")

print(f"""
╔══════════════════════════════════════╗
║         Final Model Results          ║
╠══════════════════════════════════════╣
║  Ensemble MAE  :  ${best_ens_mae:<7.2f}/night        ║
║  Ensemble R²   :  {ens_r2:.4f}               ║
║  CV MAE (XGB)  :  ${np.mean(cv_maes):<7.2f}/night        ║
╚══════════════════════════════════════╝

Next step: streamlit run app.py
""")
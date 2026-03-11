"""
Airbnb Price Advisor — Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

st.set_page_config(page_title="Airbnb Price Advisor", layout="centered")

st.markdown("""
<style>
.block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────────────────────────────────
class AirbnbPriceNet(nn.Module):
    def __init__(self, n_num, embed_sizes):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(n + 1, d) for n, d in embed_sizes
        ])
        total = n_num + sum(d for _, d in embed_sizes)
        self.net = nn.Sequential(
            nn.Linear(total, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),    nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
    def forward(self, x_num, x_cat):
        embs = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        return self.net(torch.cat([x_num] + embs, dim=1)).squeeze()


# ─────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    with open("model_artifacts.pkl", "rb") as f:
        art = pickle.load(f)
    nn_m = AirbnbPriceNet(len(art["num_features"]), art["embed_sizes"])
    nn_m.load_state_dict(torch.load("nn_model.pt", map_location="cpu"))
    nn_m.eval()
    return art, nn_m

try:
    art, nn_m = load_model()
except FileNotFoundError:
    st.error("Run `python train.py` first to generate model files.")
    st.stop()

NEIGHBOURHOODS = sorted(art["label_encoders"]["neighbourhood_cleansed"].classes_.tolist())
ROOM_TYPES     = art["label_encoders"]["room_type"].classes_.tolist()
PROP_TYPES     = art["label_encoders"]["property_type"].classes_.tolist()


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
def haversine(lat, lon, c_lat=40.7580, c_lon=-73.9855):
    R = 6371
    lat, lon     = np.radians(lat), np.radians(lon)
    c_lat, c_lon = np.radians(c_lat), np.radians(c_lon)
    a = (np.sin((c_lat - lat) / 2) ** 2 +
         np.cos(lat) * np.cos(c_lat) * np.sin((c_lon - lon) / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))

def safe_enc(le, val):
    return list(le.classes_).index(val) if val in le.classes_ else 0

def build_row(inp, art):
    ns  = art["neigh_stats"]
    row = ns[ns["neighbourhood_cleansed"] == inp["neighbourhood"]]
    nm  = row["neigh_median"].values[0] if len(row) else 150.0
    nmn = row["neigh_mean"].values[0]   if len(row) else 150.0
    lat = row["neigh_lat"].values[0]    if len(row) else 40.71
    lon = row["neigh_lon"].values[0]    if len(row) else -74.00
    lr  = np.log1p(inp["n_reviews"])

    feat = {
        "host_total_listings_count" : 1,
        "accommodates"              : inp["accommodates"],
        "accommodates_sqrd"         : inp["accommodates"] ** 2,
        "bedrooms"                  : inp["bedrooms"],
        "beds"                      : max(inp["bedrooms"], 1),
        "bathrooms"                 : inp["bathrooms"],
        "minimum_nights"            : inp["min_nights"],
        "availability_365"          : inp["availability"],
        "number_of_reviews"         : inp["n_reviews"],
        "log_reviews"               : lr,
        "reviews_per_month"         : round(inp["n_reviews"] / 12, 2),
        "review_scores_rating"      : inp["rating"],
        "rating_x_reviews"          : inp["rating"] * lr,
        "host_quality"              : inp["rating"] / 5 + 1 / 3,
        "geo_distance"              : haversine(lat, lon),
        "occupancy_rate"            : 1 - inp["availability"] / 365,
        "neigh_price_tier"          : nmn * 0.6 + nm * 0.4,
        "neigh_median"              : nm,
        "neigh_mean"                : nmn,
        "neighbourhood_cleansed_enc": safe_enc(art["label_encoders"]["neighbourhood_cleansed"], inp["neighbourhood"]),
        "room_type_enc"             : safe_enc(art["label_encoders"]["room_type"],               inp["room_type"]),
        "property_type_enc"         : safe_enc(art["label_encoders"]["property_type"],           inp["property_type"]),
        "host_type_enc"             : safe_enc(art["label_encoders"]["host_type"],               "Individual"),
    }
    return pd.DataFrame([feat])[art["all_features"]], nm

def predict(X_raw, art, nn_m):
    X_s = X_raw.copy()
    X_s[art["num_features"]] = art["scaler"].transform(X_raw[art["num_features"]])
    xgb_p = art["xgb_model"].predict(X_raw)[0]
    t_num = torch.FloatTensor(X_s[art["num_features"]].values)
    t_cat = torch.LongTensor(X_s[art["cat_enc_features"]].values)
    with torch.no_grad():
        nn_p = nn_m(t_num, t_cat).item()
    w = art["xgb_weight"]
    return w * xgb_p + (1 - w) * nn_p, xgb_p, nn_p


# ─────────────────────────────────────────────────────────────────────
# UI — HEADER
# ─────────────────────────────────────────────────────────────────────
st.title("Airbnb Price Advisor — NYC")
st.caption(
    "Enter your listing details to get a data-driven price recommendation. "
    "The model analyzes neighbourhood pricing, property specs, and host metrics "
    "across 14,000+ NYC listings to suggest the optimal nightly rate — "
    "helping you stay competitive without leaving money on the table."
)
st.divider()


# ─────────────────────────────────────────────────────────────────────
# UI — INPUTS
# ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
neighbourhood = col1.selectbox("Neighbourhood", NEIGHBOURHOODS,
                                index=NEIGHBOURHOODS.index("Williamsburg")
                                if "Williamsburg" in NEIGHBOURHOODS else 0)
room_type     = col2.selectbox("Room Type", ROOM_TYPES)
property_type = col1.selectbox("Property Type", PROP_TYPES)
rating        = col2.number_input("Review Score (1–5)", 1.0, 5.0, 4.5, step=0.1)

c1, c2, c3 = st.columns(3)
accommodates = c1.number_input("Guests",    1, 16, 2)
bedrooms     = c2.number_input("Bedrooms",  0, 10, 1)
bathrooms    = c3.number_input("Bathrooms", 0.5, 10.0, 1.0, step=0.5)

c4, c5, c6 = st.columns(3)
min_nights   = c4.number_input("Min Nights",         1, 365, 30)
availability = c5.number_input("Days Available / yr", 0, 365, 200)
n_reviews    = c6.number_input("Number of Reviews",   0, 2000, 50)


predict_btn = st.button("Get Price Recommendation", use_container_width=True, type="primary")


# ─────────────────────────────────────────────────────────────────────
# UI — RESULTS
# ─────────────────────────────────────────────────────────────────────
if predict_btn:
    inp = dict(
        neighbourhood=neighbourhood, room_type=room_type,
        property_type=property_type, accommodates=accommodates,
        bedrooms=bedrooms, bathrooms=bathrooms, rating=rating,
        n_reviews=n_reviews, min_nights=min_nights, availability=availability,
    )

    with st.spinner("Analysing your listing..."):
        X_raw, neigh_median = build_row(inp, art)
        price, xgb_p, nn_p  = predict(X_raw, art, nn_m)

    price_int  = int(round(price))
    median_int = int(round(neigh_median))
    diff       = price_int - median_int
    diff_pct   = diff / median_int * 100
    diff_str   = ("+" if diff_pct >= 0 else "") + f"{diff_pct:.1f}%"

    # ── Metrics ───────────────────────────────────────────────────────
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Recommended Price",    str(price_int) + " / night")
    m2.metric("Neighbourhood Median", str(median_int) + " / night")
    m3.metric("vs Median",            diff_str)

    # ── Market position ───────────────────────────────────────────────
    if diff_pct > 15:
        st.warning(
            "Your recommended price is "
            + str(round(diff_pct)) + "% above the area median of "
            + str(median_int) + " per night. "
            "Strong amenities and photos are essential to justify this."
        )
    elif diff_pct < -15:
        st.info(
            "Your recommended price is below the area median of "
            + str(median_int) + " per night. "
            "There is room to increase your rate gradually."
        )
    else:
        st.success(
            "Your recommended price is within 15% of the area median of "
            + str(median_int) + " per night — a strong competitive position."
        )

    st.divider()

    # ── Feature importance ────────────────────────────────────────────
    st.subheader("What influences this price?")
    st.caption("Relative importance of each factor in the XGBoost model.")

    fi     = art["feature_importance"]
    LABELS = {
        "neigh_price_tier"          : "Neighbourhood Tier",
        "neigh_mean"                : "Area Mean Price",
        "neigh_median"              : "Area Median Price",
        "geo_distance"              : "Distance to Centre",
        "accommodates"              : "Guests",
        "accommodates_sqrd"         : "Guest Demand (sq)",
        "bedrooms"                  : "Bedrooms",
        "bathrooms"                 : "Bathrooms",
        "minimum_nights"            : "Min Nights",
        "review_scores_rating"      : "Review Score",
        "rating_x_reviews"          : "Rating x Reviews",
        "log_reviews"               : "Review Count",
        "host_quality"              : "Host Quality",
        "room_type_enc"             : "Room Type",
        "property_type_enc"         : "Property Type",
        "availability_365"          : "Availability",
        "occupancy_rate"            : "Occupancy Rate",
        "number_of_reviews"         : "No. of Reviews",
        "reviews_per_month"         : "Reviews / Month",
        "neighbourhood_cleansed_enc": "Neighbourhood",
    }

    top_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:12]
    names  = [LABELS.get(f, f) for f, _ in top_fi][::-1]
    values = [v for _, v in top_fi][::-1]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color="#FF5A5F",
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis_title="Importance Score",
        yaxis=dict(showgrid=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Neighbourhood distribution ────────────────────────────────────
    st.subheader("Price distribution — " + neighbourhood)
    st.caption(
        "How your recommended price compares to all listings in " + neighbourhood + "."
    )

    neigh_prices = art["df_clean"][
        art["df_clean"]["neighbourhood_cleansed"] == neighbourhood
    ]["price"]

    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=neigh_prices, nbinsx=30,
        marker_color="#636EFA", name="Listings",
    ))
    fig2.add_vline(
        x=price_int, line_color="#FF5A5F", line_width=2,
        annotation_text="Your price " + str(price_int),
        annotation_font_color="#FF5A5F",
    )
    fig2.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Price per night",
        yaxis_title="Number of listings",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Insights ──────────────────────────────────────────────────────
    st.subheader("Insights")

    i1, i2, i3 = st.columns(3)

    with i1:
        if diff_pct > 15:
            st.info(
                "**Market Position**\n\n"
                "You are " + str(round(diff_pct)) + "% above the area median. "
                "Strong amenities and photos are essential to justify this."
            )
        elif diff_pct < -15:
            st.info(
                "**Market Position**\n\n"
                "You are below the area median. "
                "There is room to increase your rate gradually."
            )
        else:
            st.info(
                "**Market Position**\n\n"
                "You are well positioned near the area median."
            )

    with i2:
        if rating < 4.7:
            st.info(
                "**Review Score**\n\n"
                "Your score of " + str(rating) + " is good. "
                "Reaching 4.8+ typically supports a higher price."
            )
        else:
            st.info(
                "**Review Score**\n\n"
                "A score of " + str(rating) + " is excellent "
                "and supports a premium price."
            )

    with i3:
        if availability < 100:
            st.info(
                "**Availability**\n\n"
                "Only " + str(availability) + " days open. "
                "Consider opening more dates to increase earnings."
            )
        elif availability > 300:
            st.info(
                "**Availability**\n\n"
                + str(availability) + " days open is high. "
                "Strategic blackout dates can support a higher price."
            )
        else:
            st.info(
                "**Availability**\n\n"
                + str(availability) + " days open is a healthy balance."
            )

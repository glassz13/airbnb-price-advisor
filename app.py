"""
Airbnb Price Advisor — Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, torch, torch.nn as nn
import plotly.graph_objects as go
import warnings; warnings.filterwarnings("ignore")

st.set_page_config(page_title="Airbnb Price Advisor", page_icon="🏠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');
* { font-family: 'DM Sans', sans-serif; }
.stApp { background: #000000; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2rem 4rem; max-width: 780px; }

label { font-size: 0.72rem !important; font-weight: 600 !important;
        text-transform: uppercase; letter-spacing: 0.05em; color: #999 !important; }

.stSelectbox > div > div, .stNumberInput input {
    background: #fff !important; border: 1.5px solid #ebe8e3 !important;
    border-radius: 10px !important; color: #1a1a1a !important;
    font-size: 0.95rem !important; font-weight: 500 !important; }
.stSelectbox > div > div > div { color: #1a1a1a !important; }
input[type="number"] { color: #1a1a1a !important; font-weight: 500 !important; }

div[data-testid="metric-container"] {
    background: #fff; border-radius: 14px; padding: 20px 24px;
    border: 1px solid #ebe8e3; }
div[data-testid="metric-container"] label {
    font-size: 0.68rem !important; color: #aaa !important; }
div[data-testid="metric-container"] > div {
    font-size: 1.4rem !important; font-weight: 700 !important; color: #1a1a1a !important; }

.stButton > button {
    background: #222 !important; color: #fff !important; border: none !important;
    border-radius: 12px !important; font-weight: 600 !important;
    font-size: 0.95rem !important; padding: 14px 0 !important;
    letter-spacing: 0.02em; width: 100%; transition: background 0.2s; }
.stButton > button:hover { background: #FF5A5F !important; }

.price-box {
    background: #222; color: #fff; border-radius: 18px;
    padding: 36px 40px; text-align: center; margin: 8px 0; }
.price-box .label { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #666; margin-bottom: 8px; }
.price-box .price { font-family: 'Syne', sans-serif; font-size: 4.2rem;
    font-weight: 800; color: #FF5A5F; line-height: 1; }
.price-box .sub { font-size: 0.85rem; color: #666; margin-top: 12px; line-height: 1.6; }
.price-box .tag { display: inline-block; margin-top: 10px; padding: 4px 14px;
    border-radius: 100px; font-size: 0.75rem; font-weight: 600; }
.tag-above  { background: #3a2200; color: #ffaa40; }
.tag-below  { background: #0a2e1a; color: #4ade80; }
.tag-good   { background: #1a2a1a; color: #86efac; }

.insight { background: #fff; border-radius: 14px; padding: 18px 20px;
    border: 1px solid #ebe8e3; font-size: 0.85rem;
    line-height: 1.7; color: #555; height: 100%; }
.insight strong { color: #1a1a1a; }
.section-label { font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #bbb; margin-bottom: 10px; }
hr { border: none; border-top: 1px solid #ebe8e3; margin: 28px 0; }
</style>
""", unsafe_allow_html=True)


# ── Model ─────────────────────────────────────────────────────────────
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


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    with open("model_artifacts.pkl", "rb") as f:
        art = pickle.load(f)
    nn_m = AirbnbPriceNet(len(art["num_features"]), art["embed_sizes"])
    nn_m.load_state_dict(torch.load("nn_model.pt", map_location="cpu"))
    nn_m.eval()
    return art, nn_m


# ── Helpers ───────────────────────────────────────────────────────────
def haversine(lat, lon):
    R, c_lat, c_lon = 6371, np.radians(40.7580), np.radians(-73.9855)
    lat, lon = np.radians(lat), np.radians(lon)
    a = np.sin((c_lat-lat)/2)**2 + np.cos(lat)*np.cos(c_lat)*np.sin((c_lon-lon)/2)**2
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
        "host_total_listings_count": 1,
        "accommodates":              inp["accommodates"],
        "accommodates_sqrd":         inp["accommodates"] ** 2,
        "bedrooms":                  inp["bedrooms"],
        "beds":                      max(inp["bedrooms"], 1),
        "bathrooms":                 inp["bathrooms"],
        "minimum_nights":            inp["min_nights"],
        "availability_365":          inp["availability"],
        "number_of_reviews":         inp["n_reviews"],
        "log_reviews":               lr,
        "reviews_per_month":         round(inp["n_reviews"] / 12, 2),
        "review_scores_rating":      inp["rating"],
        "rating_x_reviews":          inp["rating"] * lr,
        "host_quality":              inp["rating"] / 5 + 1/3,
        "geo_distance":              haversine(lat, lon),
        "occupancy_rate":            1 - inp["availability"] / 365,
        "neigh_price_tier":          nmn * 0.6 + nm * 0.4,
        "neigh_median":              nm,
        "neigh_mean":                nmn,
        "neighbourhood_cleansed_enc": safe_enc(art["label_encoders"]["neighbourhood_cleansed"], inp["neighbourhood"]),
        "room_type_enc":              safe_enc(art["label_encoders"]["room_type"],               inp["room_type"]),
        "property_type_enc":          safe_enc(art["label_encoders"]["property_type"],           inp["property_type"]),
        "host_type_enc":              safe_enc(art["label_encoders"]["host_type"],               "Individual"),
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
    return w * xgb_p + (1-w) * nn_p, xgb_p, nn_p


# ── Load ──────────────────────────────────────────────────────────────
try:
    art, nn_m = load_model()
except FileNotFoundError:
    st.error("Run `python train.py` first to generate model files.")
    st.stop()

NEIGHBOURHOODS = sorted(art["label_encoders"]["neighbourhood_cleansed"].classes_.tolist())
ROOM_TYPES     = art["label_encoders"]["room_type"].classes_.tolist()
PROP_TYPES     = art["label_encoders"]["property_type"].classes_.tolist()


# ── Header ────────────────────────────────────────────────────────────
st.markdown("<h1 style='font-family:Syne,sans-serif;font-size:2.1rem;margin-bottom:4px;'>Airbnb Price Advisor</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#aaa;font-size:0.88rem;margin-bottom:28px;'>Data-driven pricing for smarter hosting. Enter your listing details below.</p>", unsafe_allow_html=True)


# ── Inputs ────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
neighbourhood = c1.selectbox("Neighbourhood", NEIGHBOURHOODS,
                              index=NEIGHBOURHOODS.index("Williamsburg") if "Williamsburg" in NEIGHBOURHOODS else 0)
room_type     = c2.selectbox("Room Type", ROOM_TYPES)
property_type = c1.selectbox("Property Type", PROP_TYPES)
rating        = c2.number_input("Review Score", 1.0, 5.0, 4.5, step=0.1)

c3, c4, c5 = st.columns(3)
accommodates = c3.number_input("Guests",    1, 16, 2)
bedrooms     = c4.number_input("Bedrooms",  0, 10, 1)
bathrooms    = c5.number_input("Bathrooms", 0.5, 10.0, 1.0, step=0.5)

c6, c7, c8 = st.columns(3)
min_nights   = c6.number_input("Min Nights",         1, 365, 30)
availability = c7.number_input("Days Available / yr", 0, 365, 200)
n_reviews    = c8.number_input("No. of Reviews",      0, 2000, 50)

st.markdown("<hr>", unsafe_allow_html=True)
predict_btn = st.button("Get Price Recommendation →")


# ── Results ───────────────────────────────────────────────────────────
if predict_btn:
    inp = dict(
        neighbourhood=neighbourhood, room_type=room_type,
        property_type=property_type, accommodates=accommodates,
        bedrooms=bedrooms, bathrooms=bathrooms, rating=rating,
        n_reviews=n_reviews, min_nights=min_nights, availability=availability,
    )

    with st.spinner("Analysing your listing…"):
        X_raw, neigh_median = build_row(inp, art)
        price, xgb_p, nn_p  = predict(X_raw, art, nn_m)
        shap_vals = art["shap_explainer"].shap_values(X_raw)[0]

    diff_pct = (price - neigh_median) / neigh_median * 100

    if diff_pct > 15:
        tag_html = "<span class='tag tag-above'>Above Market</span>"
        insight_market = f"You're <strong>{diff_pct:.0f}% above</strong> the neighbourhood median. Make sure your photos and amenities justify the premium."
    elif diff_pct < -15:
        tag_html = "<span class='tag tag-below'>Below Market — room to grow</span>"
        insight_market = f"You could earn up to <strong>${abs(price - neigh_median):.0f} more/night</strong> by raising closer to the neighbourhood median of ${neigh_median:.0f}."
    else:
        tag_html = "<span class='tag tag-good'>Well Positioned</span>"
        insight_market = f"Your price sits within 15% of the neighbourhood median — a <strong>strong competitive position</strong>."

    # ── Price hero ────────────────────────────────────────────────────
    st.markdown(f"""
    <div class='price-box'>
        <div class='label'>Recommended Price</div>
        <div class='price'>${price:.0f}<span style='font-size:1.3rem;color:#444'>/night</span></div>
        <div class='sub'>Neighbourhood median ${neigh_median:.0f} &nbsp;·&nbsp; {neighbourhood}</div>
        {tag_html}
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics ───────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("Price Range",  f"${price*0.92:.0f} – ${price*1.08:.0f}")
    m2.metric("Area Median",  f"${neigh_median:.0f}")
    m3.metric("vs Median",    f"{'+' if diff_pct>=0 else ''}{diff_pct:.1f}%")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── SHAP ──────────────────────────────────────────────────────────
    LABELS = {
        "neigh_price_tier":"Neighbourhood Tier", "neigh_mean":"Area Mean Price",
        "neigh_median":"Area Median Price",       "geo_distance":"Distance to Centre",
        "accommodates":"Guests",                  "accommodates_sqrd":"Guest Demand",
        "bedrooms":"Bedrooms",                    "bathrooms":"Bathrooms",
        "minimum_nights":"Min Nights",            "review_scores_rating":"Review Score",
        "rating_x_reviews":"Rating × Reviews",   "log_reviews":"Review Count",
        "host_quality":"Host Quality",            "room_type_enc":"Room Type",
        "property_type_enc":"Property Type",      "availability_365":"Availability",
        "occupancy_rate":"Occupancy Rate",        "number_of_reviews":"No. of Reviews",
        "reviews_per_month":"Reviews / Month",    "beds":"Beds",
        "host_total_listings_count":"Host Listings",
    }
    pairs = sorted(
        zip([LABELS.get(f, f) for f in art["all_features"]], shap_vals),
        key=lambda x: abs(x[1]), reverse=True
    )[:12]
    names = [n for n,_ in pairs][::-1]
    vals  = [v for _,v in pairs][::-1]

    st.markdown("<div class='section-label'>Why this price?</div>", unsafe_allow_html=True)
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=["#FF5A5F" if v > 0 else "#7B9CDA" for v in vals],
        text=[f"${abs(v):.1f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="#000000", plot_bgcolor="#080602",
        font=dict(family="DM Sans", size=11, color="#555"),
        margin=dict(l=160, r=60, t=10, b=20), height=370,
        xaxis=dict(title="Price impact ($)", showgrid=True,
                   gridcolor="#ebe8e3", zeroline=True, zerolinecolor="#ccc"),
        yaxis=dict(showgrid=False),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<p style='font-size:0.75rem;color:#bbb;margin-top:-12px;'>"
                "🔴 Pushes price up &nbsp; 🔵 Pushes price down &nbsp;·&nbsp; "
                "Relative to dataset average</p>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Distribution ──────────────────────────────────────────────────
    neigh_prices = art["df_clean"][
        art["df_clean"]["neighbourhood_cleansed"] == neighbourhood]["price"]

    st.markdown("<div class='section-label'>Price distribution in your neighbourhood</div>",
                unsafe_allow_html=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=neigh_prices, nbinsx=30,
        marker_color="#E0DBD4", marker_line_color="#d4cfc8",
        marker_line_width=0.5, name="Listings",
    ))
    fig2.add_vline(x=price, line_color="#FF5A5F", line_width=2.5,
                   annotation_text=f" Your price ${price:.0f}",
                   annotation_font_color="#FF5A5F", annotation_font_size=12)
    fig2.update_layout(
        paper_bgcolor="#F7F5F2", plot_bgcolor="#F7F5F2",
        font=dict(family="DM Sans", size=11, color="#555"),
        margin=dict(l=20, r=20, t=10, b=20), height=260,
        xaxis=dict(title="$/night", showgrid=False),
        yaxis=dict(title="Listings", showgrid=True, gridcolor="#ebe8e3"),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Insights ──────────────────────────────────────────────────────
    st.markdown("<div class='section-label'>Insights</div>", unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3)

    with i1:
        st.markdown(f"<div class='insight'>📍 <strong>Market Position</strong><br>{insight_market}</div>",
                    unsafe_allow_html=True)
    with i2:
        if rating < 4.7:
            tip = f"Improving from <strong>{rating}</strong> to 4.8+ directly increases your recommended price. Focus on cleanliness and communication."
        else:
            tip = f"A score of <strong>{rating}</strong> is excellent and supports a premium price. Keep it up."
        st.markdown(f"<div class='insight'>⭐ <strong>Review Score</strong><br>{tip}</div>",
                    unsafe_allow_html=True)
    with i3:
        if availability < 100:
            tip = "Low availability can signal high demand — but also limits your earnings window. Consider opening more dates."
        elif availability > 300:
            tip = "High availability may signal lower demand in the model. Consider strategic blackout dates to create scarcity."
        else:
            tip = f"<strong>{availability} days</strong> available is a balanced availability profile."
        st.markdown(f"<div class='insight'>📅 <strong>Availability</strong><br>{tip}</div>",
                    unsafe_allow_html=True)
"""
Mobile App Review Analyzer
---------------------------
An AI PM tool that turns a firehose of raw app-store reviews into a weekly,
skimmable signal: sentiment trend by version, auto-tagged complaint themes,
and the specific reviews worth reading.

Drop this file into the `pages/` folder of the existing bhuvi-ai-lab
Streamlit multipage app. No external API key required — sentiment and
theme-tagging run locally so the demo works instantly and free of cost.

Author: Bhuvaneswari Kuduva Premkumar
"""

import re
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Mobile App Review Analyzer", page_icon="📱", layout="wide")

# ---------------------------------------------------------------------------
# 1. Lightweight, dependency-free sentiment + theme engine
#    (V1 deliberately avoids a paid LLM API call per review: it needs to
#    process thousands of reviews at zero marginal cost and stay fully
#    explainable to non-technical stakeholders. See "AI Architecture"
#    write-up on the site for the fine-tuned/LLM upgrade path.)
# ---------------------------------------------------------------------------

POS_WORDS = {
    "love", "great", "amazing", "excellent", "awesome", "fantastic", "easy",
    "smooth", "fast", "helpful", "best", "perfect", "good", "nice", "works",
    "reliable", "intuitive", "convenient", "recommend", "wonderful",
}
NEG_WORDS = {
    "crash", "crashes", "crashed", "bug", "buggy", "slow", "broken", "freeze",
    "freezes", "hate", "worst", "terrible", "awful", "annoying", "useless",
    "confusing", "glitch", "glitchy", "error", "fail", "fails", "failed",
    "lag", "laggy", "unresponsive", "disappointed", "scam", "rude",
}

THEME_KEYWORDS = {
    "Crashes / Stability": {"crash", "crashes", "crashed", "freeze", "freezes", "force close"},
    "Performance / Speed": {"slow", "lag", "laggy", "loading", "unresponsive", "battery"},
    "UI / UX": {"confusing", "layout", "design", "interface", "navigate", "ui", "ux", "font"},
    "Pricing / Billing": {"price", "expensive", "subscription", "charged", "refund", "billing"},
    "Customer Support": {"support", "response", "customer service", "help desk", "contact"},
    "Login / Account": {"login", "log in", "password", "account", "signup", "sign up", "locked out"},
    "Feature Request": {"wish", "please add", "would be nice", "missing", "feature request"},
}


def score_sentiment(text: str) -> str:
    words = re.findall(r"[a-z']+", text.lower())
    pos = sum(w in POS_WORDS for w in words)
    neg = sum(w in NEG_WORDS for w in words)
    if pos == neg:
        return "Neutral"
    return "Positive" if pos > neg else "Negative"


def tag_themes(text: str):
    low = text.lower()
    hits = [theme for theme, kws in THEME_KEYWORDS.items() if any(kw in low for kw in kws)]
    return hits or ["Uncategorized"]


# ---------------------------------------------------------------------------
# 2. Synthetic sample dataset so the demo works with zero setup
# ---------------------------------------------------------------------------

SAMPLE_REVIEWS = [
    ("The app keeps crashing every time I try to upload a photo. So frustrating!", 1, "3.2"),
    ("Love how fast the new checkout is, huge improvement!", 5, "3.4"),
    ("Support never responds, been waiting 5 days for a refund.", 2, "3.3"),
    ("Clean design, super intuitive, exactly what I needed.", 5, "3.4"),
    ("App is so slow after the last update, battery drains fast too.", 2, "3.3"),
    ("Login keeps failing, had to reset my password 3 times.", 1, "3.3"),
    ("Great app overall, wish it had a dark mode though.", 4, "3.2"),
    ("Charged me twice for the subscription, terrible billing experience.", 1, "3.4"),
    ("Works perfectly, best banking app I've used.", 5, "3.4"),
    ("Confusing navigation, I can't find my past orders anymore.", 2, "3.3"),
    ("Freezes constantly on Android, unusable since the update.", 1, "3.4"),
    ("Really smooth experience, customer service was quick to help.", 5, "3.2"),
    ("Please add a way to export my data, missing feature.", 3, "3.2"),
    ("Awesome update, everything loads so much faster now!", 5, "3.4"),
    ("Locked out of my account and support hasn't replied.", 1, "3.3"),
]


def build_sample_df(n_days=30):
    rows = []
    start = datetime.today() - timedelta(days=n_days)
    for i in range(220):
        text, rating, version = random.choice(SAMPLE_REVIEWS)
        date = start + timedelta(days=random.randint(0, n_days), hours=random.randint(0, 23))
        rows.append({"review_text": text, "rating": rating, "app_version": version, "date": date})
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 3. UI
# ---------------------------------------------------------------------------

st.title("📱 Mobile App Review Analyzer")
st.caption(
    "Paste or upload raw app-store reviews → get sentiment trend by version, "
    "auto-tagged complaint themes, and the reviews worth reading — in minutes, not hours."
)

with st.sidebar:
    st.header("Data Source")
    source = st.radio("Choose input", ["Use sample dataset", "Upload CSV"], index=0)
    st.caption("CSV needs columns: `review_text`, `rating`, `app_version`, `date`")
    uploaded = None
    if source == "Upload CSV":
        uploaded = st.file_uploader("Upload reviews CSV", type=["csv"])

if source == "Upload CSV" and uploaded is not None:
    df = pd.read_csv(uploaded, parse_dates=["date"])
else:
    df = build_sample_df()
    if source == "Upload CSV":
        st.info("No file uploaded yet — showing the sample dataset below.")

df["sentiment"] = df["review_text"].apply(score_sentiment)
df["themes"] = df["review_text"].apply(tag_themes)

# --- Top-line metrics -------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
total = len(df)
pos_pct = (df["sentiment"] == "Positive").mean() * 100
neg_pct = (df["sentiment"] == "Negative").mean() * 100
avg_rating = df["rating"].mean()

col1.metric("Reviews analyzed", f"{total:,}")
col2.metric("Positive sentiment", f"{pos_pct:.0f}%")
col3.metric("Negative sentiment", f"{neg_pct:.0f}%", delta=f"{neg_pct-30:.0f}pp vs. target", delta_color="inverse")
col4.metric("Avg. star rating", f"{avg_rating:.1f} ★")

st.divider()

# --- Sentiment over time by version ----------------------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Sentiment trend by app version")
    trend = (
        df.groupby([pd.Grouper(key="date", freq="D"), "sentiment"])
        .size()
        .reset_index(name="count")
    )
    fig = px.area(
        trend, x="date", y="count", color="sentiment",
        color_discrete_map={"Positive": "#00c2a8", "Neutral": "#e2ddd6", "Negative": "#ef4444"},
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=340)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Sentiment split")
    pie = px.pie(
        df, names="sentiment", hole=0.55,
        color="sentiment",
        color_discrete_map={"Positive": "#00c2a8", "Neutral": "#e2ddd6", "Negative": "#ef4444"},
    )
    pie.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=340, showlegend=True)
    st.plotly_chart(pie, use_container_width=True)

st.divider()

# --- Themes ------------------------------------------------------------------
st.subheader("What are negative reviews actually about?")
neg_df = df[df["sentiment"] == "Negative"].explode("themes")
if len(neg_df):
    theme_counts = neg_df["themes"].value_counts().reset_index()
    theme_counts.columns = ["theme", "count"]
    bar = px.bar(theme_counts, x="count", y="theme", orientation="h", color="count",
                 color_continuous_scale=["#1a4fff", "#ef4444"])
    bar.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(bar, use_container_width=True)
else:
    st.info("No negative reviews in this dataset.")

# --- Version comparison table -------------------------------------------------
st.subheader("Sentiment by version — did the last release help or hurt?")
version_summary = (
    df.groupby("app_version")
    .agg(reviews=("review_text", "count"),
         avg_rating=("rating", "mean"),
         pct_negative=("sentiment", lambda s: (s == "Negative").mean() * 100))
    .round(1)
    .sort_index()
)
st.dataframe(version_summary, use_container_width=True)

# --- Reviews worth reading -----------------------------------------------------
st.subheader("Top negative reviews to read this week")
st.dataframe(
    df[df["sentiment"] == "Negative"][["date", "app_version", "rating", "review_text", "themes"]]
    .sort_values("date", ascending=False)
    .head(10),
    use_container_width=True,
)

# --- Export --------------------------------------------------------------------
st.download_button(
    "⬇ Download full analysis as CSV",
    df.drop(columns=["themes"]).assign(themes=df["themes"].apply(lambda t: ", ".join(t))).to_csv(index=False),
    file_name="review_analysis_summary.csv",
    mime="text/csv",
)

st.divider()
st.caption(
    "Built by Bhuvaneswari Kuduva Premkumar · V1 uses a lexicon-based sentiment + keyword "
    "taxonomy engine for speed, cost, and explainability. See the AI Architecture section "
    "on aiwithbhuvi.blog for the transformer-model upgrade path."
)

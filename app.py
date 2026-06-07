"""
ETF Predictor Dashboard — Streamlit App

Run from project root:
    streamlit run app.py
"""

import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

from config import AVAILABLE_ETFS, PREDICTION_DAYS, REFRESH_INTERVAL_SECONDS
from src.data_loader import download_etf_data, update_etf_data, load_etf_data
from src.features import add_features, prepare_training_data
from src.model import ProphetModel, XGBoostModel
from src.charts import create_price_chart, create_metrics_card


# --- Page Config ---
st.set_page_config(
    page_title="ETF Predictor",
    page_icon="📈",
    layout="wide",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2a2a4a;
    }
    .price-up { color: #26a69a; }
    .price-down { color: #ef5350; }
    .stApp { background-color: #0e1117; }
</style>
""", unsafe_allow_html=True)

# --- Auto-refresh during market hours (every 5 min) ---
# TSX market hours: 9:30 AM - 4:00 PM ET, Mon-Fri
import pytz
now_et = datetime.now(pytz.timezone("America/Toronto"))
is_market_hours = (
    now_et.weekday() < 5
    and now_et.hour >= 9 and now_et.hour < 16
    and not (now_et.hour == 9 and now_et.minute < 30)
)

if is_market_hours:
    st_autorefresh(
        interval=REFRESH_INTERVAL_SECONDS * 1000,
        key="live_refresh",
    )


def get_live_price(ticker: str) -> dict:
    """Fetch the latest delayed price from Yahoo Finance."""
    import yfinance as yf
    try:
        etf = yf.Ticker(ticker)
        info = etf.fast_info
        return {
            "price": info.last_price,
            "prev_close": info.previous_close,
            "open": info.open,
            "day_high": info.day_high,
            "day_low": info.day_low,
        }
    except Exception:
        return None


# --- Session State Init ---
if "selected_etfs" not in st.session_state:
    st.session_state.selected_etfs = []
if "models" not in st.session_state:
    st.session_state.models = {}
if "predictions" not in st.session_state:
    st.session_state.predictions = {}


# --- Sidebar ---
st.sidebar.title("📈 ETF Predictor")
st.sidebar.markdown("Canadian ETF Price Prediction Dashboard")
st.sidebar.divider()

# ETF Selection
st.sidebar.subheader("Select ETFs to Track")
selected = st.sidebar.multiselect(
    "Choose ETFs",
    options=list(AVAILABLE_ETFS.keys()),
    default=st.session_state.selected_etfs or ["XIU.TO"],
    format_func=lambda x: f"{x} — {AVAILABLE_ETFS[x]}",
)
st.session_state.selected_etfs = selected

# Model selection
st.sidebar.divider()
model_type = st.sidebar.radio(
    "Prediction Model",
    ["Prophet", "XGBoost"],
    help="Prophet: trend/seasonality based. XGBoost: technical indicators based.",
)

# Chart type
chart_type = st.sidebar.radio(
    "Chart Style",
    ["line", "candlestick"],
)

# Prediction horizon
pred_days = st.sidebar.slider(
    "Prediction Days",
    min_value=7,
    max_value=90,
    value=PREDICTION_DAYS,
    step=7,
)

# Time range for historical view
time_range = st.sidebar.selectbox(
    "Historical Range",
    ["1M", "3M", "6M", "1Y", "2Y", "All"],
    index=0,
)

# Action buttons
st.sidebar.divider()
refresh_data = st.sidebar.button("🔄 Refresh Data", use_container_width=True)
retrain = st.sidebar.button("🧠 Train/Retrain Models", use_container_width=True)


def filter_by_range(df: pd.DataFrame, range_str: str) -> pd.DataFrame:
    """Filter DataFrame by time range string."""
    if range_str == "All":
        return df

    range_map = {
        "1M": 30, "3M": 90, "6M": 180, "1Y": 365, "2Y": 730,
    }
    days = range_map.get(range_str, 365)
    cutoff = df.index.max() - pd.Timedelta(days=days)
    return df[df.index >= cutoff]


def load_or_download(ticker: str) -> pd.DataFrame:
    """Try to load data, download if not available."""
    try:
        return load_etf_data(ticker)
    except FileNotFoundError:
        return download_etf_data(ticker)


def train_model(ticker: str, df: pd.DataFrame, model_type: str):
    """Train the selected model for a ticker."""
    if model_type == "Prophet":
        model = ProphetModel()
        model.train(df, ticker)
        predictions = model.predict(days=pred_days)
        model.save()
    else:
        df_feat = add_features(df)
        X, y = prepare_training_data(df_feat)
        model = XGBoostModel()
        model.train(X, y, ticker)

        # For XGBoost, we need to create future predictions differently
        # Use the last row's features to predict next day iteratively
        last_features = X.iloc[[-1]]
        predictions = pd.DataFrame(
            {"Predicted_Close": [model.predict(last_features).values[0]]},
            index=[df.index.max() + pd.Timedelta(days=1)],
        )
        model.save()

    return model, predictions


# --- Main Content ---
if not selected:
    st.title("📈 ETF Predictor")
    st.info("👈 Select one or more ETFs from the sidebar to get started.")
    st.stop()

# --- Refresh data if requested ---
if refresh_data:
    with st.spinner("Updating market data..."):
        for ticker in selected:
            update_etf_data(ticker)
    st.success("Data updated!")
    st.rerun()

# --- Train models if requested ---
if retrain:
    progress = st.progress(0)
    for i, ticker in enumerate(selected):
        with st.spinner(f"Training {model_type} model for {ticker}..."):
            df = load_or_download(ticker)
            model, preds = train_model(ticker, df, model_type)
            st.session_state.models[ticker] = model
            st.session_state.predictions[ticker] = preds
        progress.progress((i + 1) / len(selected))
    st.success("All models trained!")
    st.rerun()

# --- Auto-train on first load if no predictions exist ---
for ticker in selected:
    if ticker not in st.session_state.predictions:
        with st.spinner(f"First load — training {model_type} for {ticker}..."):
            df = load_or_download(ticker)
            model, preds = train_model(ticker, df, model_type)
            st.session_state.models[ticker] = model
            st.session_state.predictions[ticker] = preds

# --- Display each ETF ---
for ticker in selected:
    st.markdown(f"## {ticker} — {AVAILABLE_ETFS[ticker]}")

    # Load data
    try:
        df = load_or_download(ticker)
    except Exception as e:
        st.error(f"Error loading {ticker}: {e}")
        continue

    # Filter by time range
    df_display = filter_by_range(df, time_range)

    # Get predictions if available
    preds = st.session_state.predictions.get(ticker)

    # --- Metrics row ---
    col1, col2, col3, col4 = st.columns(4)

    # Use live price during market hours, historical close otherwise
    live = get_live_price(ticker) if is_market_hours else None

    if live and live["price"]:
        current_price = live["price"]
        prev_close = live["prev_close"]
        day_high = live["day_high"]
        day_low = live["day_low"]
        col1.metric("Live Price 🟢", f"${current_price:.2f}")
    else:
        current_price = df["Close"].iloc[-1]
        prev_close = df["Close"].iloc[-2]
        day_high = df["High"].iloc[-1]
        day_low = df["Low"].iloc[-1]
        col1.metric("Last Close", f"${current_price:.2f}")

    day_change = current_price - prev_close
    day_change_pct = (day_change / prev_close) * 100

    col2.metric(
        "Day Change",
        f"${day_change:.2f}",
        f"{day_change_pct:+.2f}%",
    )
    col3.metric("Day High", f"${day_high:.2f}")
    col4.metric("Day Low", f"${day_low:.2f}")

    # Prediction metrics
    if preds is not None:
        future_preds = preds[preds.index > df.index.max()]
        if not future_preds.empty:
            next_pred = future_preds.iloc[0]
            pred_price = next_pred["Predicted_Close"]
            pred_change = pred_price - current_price
            pred_pct = (pred_change / current_price) * 100

            pcol1, pcol2, pcol3 = st.columns(3)
            pcol1.metric(
                "Predicted Next Close",
                f"${pred_price:.2f}",
                f"{pred_pct:+.2f}%",
            )
            if "Lower_Bound" in future_preds.columns:
                pcol2.metric("Lower Bound", f"${next_pred['Lower_Bound']:.2f}")
                pcol3.metric("Upper Bound", f"${next_pred['Upper_Bound']:.2f}")

    # --- Chart ---
    fig = create_price_chart(
        historical_df=df_display,
        predictions_df=preds,
        ticker=ticker,
        chart_type=chart_type,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Quick stats ---
    with st.expander(f"📊 {ticker} Details"):
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        stat_col1.write(f"**Data Points:** {len(df)}")
        stat_col1.write(f"**Date Range:** {df.index.min().date()} to {df.index.max().date()}")
        stat_col2.write(f"**52-Week High:** ${df['Close'].tail(252).max():.2f}")
        stat_col2.write(f"**52-Week Low:** ${df['Close'].tail(252).min():.2f}")
        stat_col3.write(f"**Avg Volume (21d):** {df['Volume'].tail(21).mean():,.0f}")

        model_obj = st.session_state.models.get(ticker)
        if model_obj and model_obj.last_trained:
            stat_col3.write(f"**Model Last Trained:** {model_obj.last_trained.strftime('%Y-%m-%d %H:%M')}")

    st.divider()

# --- Footer ---
st.sidebar.divider()
market_status = "🟢 Market Open — auto-refreshing" if is_market_hours else "🔴 Market Closed"
st.sidebar.caption(
    f"{market_status}\n\n"
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    "⚠️ Prices delayed ~15 min. Predictions are for learning purposes only."
)

"""
Feature Engineering — compute technical indicators for ML model input.

Adds moving averages, RSI, MACD, volatility, and other features
that help the model understand price trends and momentum.
"""

import pandas as pd
import numpy as np


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicator features to a price DataFrame.

    Args:
        df: DataFrame with at least a 'Close' and 'Volume' column

    Returns:
        DataFrame with additional feature columns
    """
    df = df.copy()

    # --- Moving Averages ---
    df["MA_7"] = df["Close"].rolling(window=7).mean()
    df["MA_21"] = df["Close"].rolling(window=21).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()

    # --- Price relative to moving averages (as ratios) ---
    df["Close_to_MA7"] = df["Close"] / df["MA_7"]
    df["Close_to_MA21"] = df["Close"] / df["MA_21"]
    df["Close_to_MA50"] = df["Close"] / df["MA_50"]

    # --- Daily returns ---
    df["Daily_Return"] = df["Close"].pct_change()

    # --- Rolling volatility (21-day) ---
    df["Volatility_21"] = df["Daily_Return"].rolling(window=21).std()

    # --- RSI (Relative Strength Index, 14-day) ---
    df["RSI_14"] = _compute_rsi(df["Close"], window=14)

    # --- MACD ---
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

    # --- Volume features ---
    df["Volume_MA_21"] = df["Volume"].rolling(window=21).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA_21"]

    # --- Lag features (previous days' returns) ---
    for lag in [1, 2, 3, 5]:
        df[f"Return_Lag_{lag}"] = df["Daily_Return"].shift(lag)

    # --- Target: next day's close (what we want to predict) ---
    df["Target_Close"] = df["Close"].shift(-1)
    df["Target_Return"] = df["Daily_Return"].shift(-1)
    df["Target_Direction"] = (df["Target_Return"] > 0).astype(int)

    return df


def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute RSI (Relative Strength Index)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def prepare_training_data(df: pd.DataFrame) -> tuple:
    """
    Prepare feature matrix X and target y for model training.
    Drops rows with NaN values (from rolling calculations).

    Args:
        df: DataFrame with features already added (via add_features)

    Returns:
        (X, y) tuple — features and target
    """
    feature_columns = [
        "Close_to_MA7", "Close_to_MA21", "Close_to_MA50",
        "Daily_Return", "Volatility_21",
        "RSI_14", "MACD", "MACD_Signal", "MACD_Histogram",
        "Volume_Ratio",
        "Return_Lag_1", "Return_Lag_2", "Return_Lag_3", "Return_Lag_5",
    ]

    df_clean = df.dropna(subset=feature_columns + ["Target_Close"])

    X = df_clean[feature_columns]
    y = df_clean["Target_Close"]

    return X, y


# --- Quick test ---
if __name__ == "__main__":
    from data_loader import load_etf_data

    ticker = "XIU.TO"
    df = load_etf_data(ticker)
    df = add_features(df)

    print(f"Features for {ticker}")
    print(f"Shape after features: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample (last 3 rows):")
    print(df.tail(3).to_string())

    X, y = prepare_training_data(df)
    print(f"\nTraining data: X={X.shape}, y={y.shape}")
"""
Momentum Scanner — scan a universe of tickers and find the hottest movers.

Ranks stocks by cumulative return over recent days,
plus additional signals like consecutive green days, volume surge, and RSI.

Usage:
    from src.scanner import scan_momentum

    results = scan_momentum(days=10, top_n=20)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.watchlist import get_ticker_universe


def _fetch_ticker_data(ticker: str, period: str = "1mo") -> tuple:
    """Fetch recent data for a single ticker. Returns (ticker, df) or (ticker, None)."""
    try:
        etf = yf.Ticker(ticker)
        df = etf.history(period=period)
        if df.empty or len(df) < 5:
            return (ticker, None)
        df.index = df.index.tz_localize(None)
        return (ticker, df)
    except Exception:
        return (ticker, None)


def _compute_momentum_score(df: pd.DataFrame, lookback_days: int = 10) -> dict:
    """
    Compute momentum metrics for a single ticker's data.

    Returns dict with all momentum signals.
    """
    if df is None or len(df) < lookback_days:
        return None

    recent = df.tail(lookback_days)
    close = df["Close"]
    recent_close = recent["Close"]

    # --- Core signal: cumulative return over lookback period ---
    cum_return = (recent_close.iloc[-1] / recent_close.iloc[0] - 1) * 100

    # --- Consecutive green days ---
    daily_returns = close.pct_change().dropna()
    consecutive_green = 0
    for ret in reversed(daily_returns.values):
        if ret > 0:
            consecutive_green += 1
        else:
            break

    # --- Average daily return (last N days) ---
    avg_daily_return = daily_returns.tail(lookback_days).mean() * 100

    # --- Volume surge (recent avg vs 21-day avg) ---
    vol_recent = df["Volume"].tail(5).mean()
    vol_21d = df["Volume"].tail(21).mean()
    volume_surge = (vol_recent / vol_21d) if vol_21d > 0 else 1.0

    # --- RSI (14-day) ---
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = (100 - (100 / (1 + rs))).iloc[-1]

    # --- Price vs moving averages ---
    ma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else close.mean()
    price_vs_ma50 = (close.iloc[-1] / ma_50 - 1) * 100

    # --- Current price and daily change ---
    current_price = close.iloc[-1]
    prev_close = close.iloc[-2] if len(close) >= 2 else current_price
    daily_change = (current_price / prev_close - 1) * 100

    return {
        "current_price": round(current_price, 2),
        "daily_change_pct": round(daily_change, 2),
        "cum_return_pct": round(cum_return, 2),
        "avg_daily_return_pct": round(avg_daily_return, 2),
        "consecutive_green_days": consecutive_green,
        "volume_surge": round(volume_surge, 2),
        "rsi_14": round(rsi, 2) if not np.isnan(rsi) else 50.0,
        "price_vs_ma50_pct": round(price_vs_ma50, 2),
    }


def scan_momentum(
    market: str = "all",
    lookback_days: int = 10,
    top_n: int = 20,
    min_return: float = 5.0,
    max_workers: int = 10,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Scan the ticker universe for momentum stocks.

    Args:
        market: "tsx", "us", "etfs", or "all"
        lookback_days: Number of days for cumulative return calculation
        top_n: Number of top movers to return
        min_return: Minimum cumulative return % to include
        max_workers: Number of parallel download threads
        progress_callback: Optional callable(current, total) for progress updates

    Returns:
        DataFrame with momentum scores, sorted by cumulative return
    """
    tickers = get_ticker_universe(market)
    total = len(tickers)
    results = []

    # Fetch data in parallel for speed
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_ticker_data, ticker, "3mo"): ticker
            for ticker in tickers
        }

        for i, future in enumerate(as_completed(futures)):
            ticker = futures[future]
            try:
                ticker_name, df = future.result()
                if df is not None:
                    score = _compute_momentum_score(df, lookback_days)
                    if score is not None:
                        score["ticker"] = ticker_name
                        results.append(score)
            except Exception:
                pass

            if progress_callback:
                progress_callback(i + 1, total)

    if not results:
        return pd.DataFrame()

    # Build DataFrame and sort by cumulative return
    df_results = pd.DataFrame(results)
    df_results = df_results.set_index("ticker")

    # Filter by minimum return
    df_results = df_results[df_results["cum_return_pct"] >= min_return]

    # Sort by cumulative return (highest first)
    df_results = df_results.sort_values("cum_return_pct", ascending=False)

    # Take top N
    df_results = df_results.head(top_n)

    return df_results


def get_momentum_details(ticker: str, period: str = "6mo") -> dict:
    """
    Get detailed data for a specific momentum stock.

    Returns dict with full price history and momentum metrics.
    """
    try:
        etf = yf.Ticker(ticker)
        df = etf.history(period=period)
        if df.empty:
            return None

        df.index = df.index.tz_localize(None)
        df = df[["Open", "High", "Low", "Close", "Volume"]]

        score = _compute_momentum_score(df, lookback_days=10)

        return {
            "data": df,
            "metrics": score,
            "info": {
                "name": getattr(etf, "info", {}).get("shortName", ticker),
                "sector": getattr(etf, "info", {}).get("sector", "N/A"),
            },
        }
    except Exception:
        return None


# --- Quick test ---
if __name__ == "__main__":
    print("Scanning for momentum stocks...")
    print("This may take 1-2 minutes...\n")

    def show_progress(current, total):
        if current % 20 == 0 or current == total:
            print(f"  Scanned {current}/{total} tickers...")

    results = scan_momentum(
        market="all",
        lookback_days=10,
        top_n=15,
        min_return=3.0,
        progress_callback=show_progress,
    )

    if results.empty:
        print("No stocks met the criteria.")
    else:
        print(f"\n🔥 Top {len(results)} Momentum Stocks (last 10 days):\n")
        print(results.to_string())
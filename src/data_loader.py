"""
Data Loader — fetch and update historical ETF data using yfinance.

Usage:
    from src.data_loader import download_etf_data, update_etf_data, load_etf_data

    # First time: download full history
    download_etf_data("XIU.TO")

    # Daily update: append latest data
    update_etf_data("XIU.TO")

    # Load from disk
    df = load_etf_data("XIU.TO")
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from config import AVAILABLE_ETFS, DATA_DIR, HISTORY_YEARS


def _get_filepath(ticker: str) -> str:
    """Get the parquet file path for a given ticker."""
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f"{ticker.replace('.', '_')}.parquet")


def download_etf_data(ticker: str, years: int = HISTORY_YEARS) -> pd.DataFrame:
    """
    Download full historical data for an ETF and save as parquet.

    Args:
        ticker: ETF ticker symbol (e.g., "XIU.TO")
        years: Number of years of history to download

    Returns:
        DataFrame with OHLCV data
    """
    if ticker not in AVAILABLE_ETFS:
        raise ValueError(
            f"Unknown ticker: {ticker}. "
            f"Available: {list(AVAILABLE_ETFS.keys())}"
        )

    print(f"Downloading {years} years of data for {ticker}...")

    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)

    etf = yf.Ticker(ticker)
    df = etf.history(start=start_date, end=end_date)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check the ticker symbol.")

    # Clean up the dataframe
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "Date"
    df.index = df.index.tz_localize(None)  # Remove timezone info for consistency

    # Save to parquet
    filepath = _get_filepath(ticker)
    df.to_parquet(filepath)
    print(f"Saved {len(df)} rows to {filepath}")

    return df


def update_etf_data(ticker: str) -> pd.DataFrame:
    """
    Update existing data with the latest available prices.
    If no existing data, downloads full history instead.

    Args:
        ticker: ETF ticker symbol

    Returns:
        Updated DataFrame
    """
    filepath = _get_filepath(ticker)

    if not os.path.exists(filepath):
        print(f"No existing data for {ticker}. Downloading full history...")
        return download_etf_data(ticker)

    # Load existing data
    existing_df = pd.read_parquet(filepath)
    last_date = existing_df.index.max()

    # Fetch new data from the day after our last record
    start_date = last_date + timedelta(days=1)
    end_date = datetime.today()

    if start_date.date() >= end_date.date():
        print(f"{ticker} is already up to date (last: {last_date.date()})")
        return existing_df

    print(f"Updating {ticker} from {start_date.date()} to {end_date.date()}...")

    etf = yf.Ticker(ticker)
    new_df = etf.history(start=start_date, end=end_date)

    if new_df.empty:
        print(f"No new data available for {ticker}")
        return existing_df

    new_df = new_df[["Open", "High", "Low", "Close", "Volume"]]
    new_df.index.name = "Date"
    new_df.index = new_df.index.tz_localize(None)

    # Combine and deduplicate
    combined_df = pd.concat([existing_df, new_df])
    combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
    combined_df.sort_index(inplace=True)

    # Save
    combined_df.to_parquet(filepath)
    print(f"Updated {ticker}: now {len(combined_df)} rows (added {len(new_df)} new)")

    return combined_df


def load_etf_data(ticker: str) -> pd.DataFrame:
    """
    Load saved ETF data from disk.

    Args:
        ticker: ETF ticker symbol

    Returns:
        DataFrame with OHLCV data

    Raises:
        FileNotFoundError if data hasn't been downloaded yet
    """
    filepath = _get_filepath(ticker)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"No data file for {ticker}. Run download_etf_data('{ticker}') first."
        )

    return pd.read_parquet(filepath)


def download_all_etfs() -> dict:
    """Download full history for all available ETFs."""
    results = {}
    for ticker in AVAILABLE_ETFS:
        try:
            results[ticker] = download_etf_data(ticker)
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
    return results


def update_all_etfs() -> dict:
    """Update data for all available ETFs."""
    results = {}
    for ticker in AVAILABLE_ETFS:
        try:
            results[ticker] = update_etf_data(ticker)
        except Exception as e:
            print(f"Error updating {ticker}: {e}")
    return results


# --- Quick test ---
if __name__ == "__main__":
    # Test with one ETF
    ticker = "XIU.TO"
    print(f"\n{'='*50}")
    print(f"Testing data loader with {ticker}")
    print(f"{'='*50}\n")

    # Download
    df = download_etf_data(ticker)
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"\nLast 5 rows:")
    print(df.tail())

    # Test update (should say "already up to date")
    print(f"\n--- Testing update ---")
    df = update_etf_data(ticker)

    # Test load
    print(f"\n--- Testing load ---")
    df = load_etf_data(ticker)
    print(f"Loaded {len(df)} rows")
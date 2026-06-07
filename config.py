"""
Configuration for ETF Predictor
"""

# Canadian ETFs available for tracking
# Format: "TICKER.TO" for TSX-listed ETFs
AVAILABLE_ETFS = {
    "XIU.TO": "iShares S&P/TSX 60 Index ETF",
    "VFV.TO": "Vanguard S&P 500 Index ETF (CAD)",
    "ZSP.TO": "BMO S&P 500 Index ETF",
    "XIC.TO": "iShares Core S&P/TSX Capped Composite Index ETF",
    "VCN.TO": "Vanguard FTSE Canada All Cap Index ETF",
    "ZAG.TO": "BMO Aggregate Bond Index ETF",
    "XBB.TO": "iShares Core Canadian Universe Bond Index ETF",
    "VGRO.TO": "Vanguard Growth ETF Portfolio",
    "VBAL.TO": "Vanguard Balanced ETF Portfolio",
    "XEI.TO": "iShares S&P/TSX Composite High Dividend Index ETF",
    "ZEB.TO": "BMO Equal Weight Banks Index ETF",
    "VDY.TO": "Vanguard FTSE Canadian High Dividend Yield Index ETF",
    "HXQ.TO": "Global X NASDAQ-100 Index Corporate Class ETF",
    "XQQ.TO": "iShares NASDAQ 100 Index ETF (CAD-Hedged)",
    "ZQQ.TO": "BMO NASDAQ 100 Equity Index ETF",
}

# Data settings
HISTORY_YEARS = 5                # How many years of historical data to pull
DATA_DIR = "data/raw"            # Where to store raw price data
MODELS_DIR = "models/saved"      # Where to store trained models

# Model settings
PREDICTION_DAYS = 30             # How many days ahead to predict
RETRAIN_ON_UPDATE = True         # Retrain model when new data arrives

# Streamlit settings
REFRESH_INTERVAL_SECONDS = 300   # Poll for new prices every 5 minutes
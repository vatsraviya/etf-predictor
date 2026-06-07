# 📈 ETF Predictor

Canadian ETF price prediction dashboard with live delayed quotes and ML-powered forecasting.

## Features
- Track Canadian ETFs (TSX) with delayed price data
- ML predictions using Facebook Prophet and XGBoost
- Interactive Plotly charts with prediction overlays
- Auto-refresh during market hours (~15 min delayed quotes)
- Confidence bands for future predictions

## Tech Stack
- **Frontend/Backend:** Streamlit
- **ML Models:** Prophet, XGBoost
- **Data Source:** Yahoo Finance (via yfinance)
- **Charts:** Plotly

## Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/etf-predictor.git
cd etf-predictor

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

## Disclaimer
⚠️ Prices are delayed ~15 minutes. Predictions are for educational purposes only — not financial advice.
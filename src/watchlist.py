"""
Watchlist — curated universe of high-volume US + Canadian tickers to scan.

Kept manageable (~200) so yfinance doesn't rate-limit us.
"""

# --- TSX Canadian ETFs & Stocks ---
TSX_TICKERS = [
    # ETFs
    "XIU.TO", "VFV.TO", "ZSP.TO", "XIC.TO", "VCN.TO", "ZAG.TO",
    "XBB.TO", "VGRO.TO", "VBAL.TO", "XEI.TO", "ZEB.TO", "VDY.TO",
    "HXQ.TO", "XQQ.TO", "ZQQ.TO", "ZWB.TO", "XEQT.TO", "XGRO.TO",
    # Banks
    "RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO", "NA.TO",
    # Energy
    "ENB.TO", "TRP.TO", "SU.TO", "CNQ.TO", "CVE.TO", "IMO.TO",
    # Mining
    "ABX.TO", "NTR.TO", "FM.TO", "TECK.TO",
    # Tech
    "SHOP.TO", "CSU.TO", "OTEX.TO", "BB.TO", "LSPD.TO",
    # Telecom / Utilities
    "BCE.TO", "T.TO", "RCI-B.TO", "FTS.TO", "EMA.TO",
    # Other
    "CP.TO", "CNR.TO", "MFC.TO", "SLF.TO", "GWO.TO",
    "WCN.TO", "DOL.TO", "ATD.TO", "QSR.TO", "GIB-A.TO",
]

# --- US Large Cap (S&P 500 top components) ---
US_LARGECAP = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "UNH", "JNJ", "JPM", "V", "PG", "XOM", "MA", "HD", "CVX", "MRK",
    "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "TMO",
    "CSCO", "ACN", "ABT", "DHR", "NEE", "LIN", "PM", "TXN", "UNP",
    "RTX", "HON", "LOW", "ORCL", "CRM", "AMD", "INTC", "QCOM", "AMAT",
    "CAT", "BA", "GS", "AXP", "BLK", "ISRG", "MDLZ", "ADI", "REGN",
    "GILD", "VRTX", "SYK", "BKNG", "PLD", "CI", "CB", "SO", "DUK",
]

# --- US Growth / Momentum / Popular ---
US_GROWTH = [
    "PLTR", "SOFI", "RIVN", "LCID", "NIO", "MARA", "RIOT", "COIN",
    "SNOW", "DDOG", "NET", "CRWD", "ZS", "PANW", "OKTA", "MDB",
    "SMCI", "ARM", "MSTR", "RDDT", "HOOD", "RBLX", "U", "GRAB",
    "SE", "SHOP", "SQ", "PYPL", "ROKU", "SNAP", "PINS", "SPOT",
    "ABNB", "DASH", "UBER", "LYFT", "TTD", "DKNG", "PENN",
    "ENPH", "FSLR", "RUN", "PLUG", "CHPT",
    "GME", "AMC", "BBBY", "CLOV", "WISH", "SPCE",
    "AI", "SOUN", "IONQ", "RGTI", "QUBT",
]

# --- US Sector ETFs ---
US_ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "ARKK", "ARKW",
    "XLF", "XLE", "XLK", "XLV", "XLI", "XLY", "XLP", "XLU",
    "SOXL", "TQQQ", "SQQQ", "UVXY",
    "GLD", "SLV", "USO", "TLT", "HYG",
]

# --- Combined universe ---
ALL_TICKERS = TSX_TICKERS + US_LARGECAP + US_GROWTH + US_ETFS


def get_ticker_universe(market: str = "all") -> list:
    """
    Get list of tickers for a specific market.

    Args:
        market: "tsx", "us", "etfs", or "all"
    """
    market_map = {
        "tsx": TSX_TICKERS,
        "us": US_LARGECAP + US_GROWTH,
        "etfs": US_ETFS,
        "all": ALL_TICKERS,
    }
    return market_map.get(market, ALL_TICKERS)
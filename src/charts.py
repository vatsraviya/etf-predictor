"""
Charts — build interactive Plotly charts for the Streamlit dashboard.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_price_chart(
    historical_df: pd.DataFrame,
    predictions_df: pd.DataFrame = None,
    ticker: str = "",
    chart_type: str = "line",
    show_volume: bool = True,
) -> go.Figure:
    """
    Create an interactive price chart with prediction overlay.

    Shows one continuous 'Prediction' line alongside actual prices,
    extending into the future after the last historical date.
    """
    rows = 2 if show_volume else 1
    row_heights = [0.75, 0.25] if show_volume else [1.0]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
    )

    # --- Actual price ---
    if chart_type == "candlestick":
        fig.add_trace(
            go.Candlestick(
                x=historical_df.index,
                open=historical_df["Open"],
                high=historical_df["High"],
                low=historical_df["Low"],
                close=historical_df["Close"],
                name="Actual Price",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
            ),
            row=1, col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=historical_df.index,
                y=historical_df["Close"],
                mode="lines",
                name="Actual Price",
                line=dict(color="#2196F3", width=2),
            ),
            row=1, col=1,
        )

    # --- Prediction line (single continuous line: past fit + future forecast) ---
    if predictions_df is not None and "Predicted_Close" in predictions_df.columns:
        # Filter predictions to match the historical view range
        hist_start = historical_df.index.min()
        preds_filtered = predictions_df[predictions_df.index >= hist_start]

        if not preds_filtered.empty:
            last_hist_date = historical_df.index.max()

            # One continuous prediction line across past and future
            fig.add_trace(
                go.Scatter(
                    x=preds_filtered.index,
                    y=preds_filtered["Predicted_Close"],
                    mode="lines",
                    name="Prediction",
                    line=dict(color="#FF9800", width=2, dash="dash"),
                ),
                row=1, col=1,
            )

            # Confidence band (future portion only)
            future_preds = preds_filtered[preds_filtered.index > last_hist_date]
            if not future_preds.empty and "Upper_Bound" in future_preds.columns:
                fig.add_trace(
                    go.Scatter(
                        x=future_preds.index,
                        y=future_preds["Upper_Bound"],
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=1, col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=future_preds.index,
                        y=future_preds["Lower_Bound"],
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor="rgba(255, 152, 0, 0.15)",
                        name="Confidence Band",
                    ),
                    row=1, col=1,
                )

    # --- Volume bars ---
    if show_volume:
        colors = [
            "#26a69a" if c >= o else "#ef5350"
            for o, c in zip(historical_df["Open"], historical_df["Close"])
        ]
        fig.add_trace(
            go.Bar(
                x=historical_df.index,
                y=historical_df["Volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.5,
            ),
            row=2, col=1,
        )

    # --- Layout ---
    fig.update_layout(
        title=f"{ticker} — Price & Prediction" if ticker else "ETF Price & Prediction",
        template="plotly_dark",
        height=600,
        margin=dict(l=50, r=30, t=50, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )

    fig.update_yaxes(title_text="Price (CAD)", row=1, col=1)
    if show_volume:
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def create_metrics_card(
    current_price: float,
    predicted_price: float,
    lower_bound: float = None,
    upper_bound: float = None,
) -> dict:
    """Compute display metrics for the dashboard."""
    change = predicted_price - current_price
    change_pct = (change / current_price) * 100
    direction = "up" if change > 0 else "down"

    return {
        "current_price": current_price,
        "predicted_price": predicted_price,
        "change": change,
        "change_pct": change_pct,
        "direction": direction,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    }
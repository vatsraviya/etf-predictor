"""
Model — train and predict ETF prices.

Two approaches:
1. Prophet: time-series forecasting (good baseline, handles trends/seasonality)
2. XGBoost: gradient boosting with technical indicators as features

Usage:
    from src.model import ProphetModel, XGBoostModel

    model = ProphetModel()
    model.train(df)
    predictions = model.predict(days=30)
"""

import os
import pickle
from datetime import datetime

import pandas as pd
import numpy as np

from config import MODELS_DIR, PREDICTION_DAYS


class ProphetModel:
    """
    Facebook Prophet model for ETF price prediction.
    Good for capturing trends and seasonality with minimal tuning.
    """

    def __init__(self):
        self.model = None
        self.ticker = None
        self.last_trained = None

    def train(self, df: pd.DataFrame, ticker: str = "unknown") -> None:
        """
        Train Prophet on historical close prices.

        Args:
            df: DataFrame with DatetimeIndex and 'Close' column
            ticker: Ticker symbol (used for saving)
        """
        from prophet import Prophet

        self.ticker = ticker

        # Prophet requires columns named 'ds' (date) and 'y' (value)
        prophet_df = pd.DataFrame({
            "ds": df.index,
            "y": df["Close"].values
        })

        self.model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,  # Controls trend flexibility
        )

        # Suppress verbose output
        self.model.fit(prophet_df, iter=300)
        self.last_trained = datetime.now()

        print(f"Prophet model trained on {len(prophet_df)} data points for {ticker}")

    def predict(self, days: int = PREDICTION_DAYS) -> pd.DataFrame:
        """
        Predict future prices.

        Args:
            days: Number of days to predict ahead

        Returns:
            DataFrame with columns: Date, Predicted_Close, Lower_Bound, Upper_Bound
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Create future dates (business days only isn't built-in,
        # so we generate extra and filter later if needed)
        future = self.model.make_future_dataframe(periods=days)
        forecast = self.model.predict(future)

        # Extract only the future predictions
        predictions = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        predictions.columns = ["Date", "Predicted_Close", "Lower_Bound", "Upper_Bound"]
        predictions.set_index("Date", inplace=True)

        return predictions

    def save(self) -> str:
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save.")

        os.makedirs(MODELS_DIR, exist_ok=True)
        filepath = os.path.join(
            MODELS_DIR,
            f"prophet_{self.ticker.replace('.', '_')}.pkl"
        )

        with open(filepath, "wb") as f:
            pickle.dump({
                "model": self.model,
                "ticker": self.ticker,
                "last_trained": self.last_trained,
            }, f)

        print(f"Model saved to {filepath}")
        return filepath

    def load(self, ticker: str) -> None:
        """Load a trained model from disk."""
        filepath = os.path.join(
            MODELS_DIR,
            f"prophet_{ticker.replace('.', '_')}.pkl"
        )

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No saved model for {ticker}")

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.ticker = data["ticker"]
        self.last_trained = data["last_trained"]

        print(f"Loaded model for {ticker} (trained: {self.last_trained})")


class XGBoostModel:
    """
    XGBoost model using technical indicators as features.
    Better at capturing complex patterns but needs feature engineering.
    """

    def __init__(self):
        self.model = None
        self.ticker = None
        self.last_trained = None
        self.feature_columns = None

    def train(self, X: pd.DataFrame, y: pd.Series, ticker: str = "unknown") -> dict:
        """
        Train XGBoost on engineered features.

        Args:
            X: Feature matrix (from features.prepare_training_data)
            y: Target values (next day's close price)
            ticker: Ticker symbol

        Returns:
            Dict with training metrics
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import xgboost as xgb

        self.ticker = ticker
        self.feature_columns = list(X.columns)

        # Split: use last 20% as test (time-series, so no shuffle)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        self.last_trained = datetime.now()

        # Evaluate
        y_pred = self.model.predict(X_test)
        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mape": np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        print(f"XGBoost trained for {ticker}")
        print(f"  MAE:  ${metrics['mae']:.4f}")
        print(f"  RMSE: ${metrics['rmse']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")

        return metrics

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict close prices for given features.

        Args:
            X: Feature matrix (same columns as training)

        Returns:
            Series of predicted close prices
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        predictions = self.model.predict(X)
        return pd.Series(predictions, index=X.index, name="Predicted_Close")

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance ranking."""
        if self.model is None:
            raise ValueError("Model not trained.")

        importance = pd.DataFrame({
            "Feature": self.feature_columns,
            "Importance": self.model.feature_importances_
        }).sort_values("Importance", ascending=False)

        return importance

    def save(self) -> str:
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save.")

        os.makedirs(MODELS_DIR, exist_ok=True)
        filepath = os.path.join(
            MODELS_DIR,
            f"xgboost_{self.ticker.replace('.', '_')}.pkl"
        )

        with open(filepath, "wb") as f:
            pickle.dump({
                "model": self.model,
                "ticker": self.ticker,
                "last_trained": self.last_trained,
                "feature_columns": self.feature_columns,
            }, f)

        print(f"Model saved to {filepath}")
        return filepath

    def load(self, ticker: str) -> None:
        """Load a trained model from disk."""
        filepath = os.path.join(
            MODELS_DIR,
            f"xgboost_{ticker.replace('.', '_')}.pkl"
        )

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No saved model for {ticker}")

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.ticker = data["ticker"]
        self.last_trained = data["last_trained"]
        self.feature_columns = data["feature_columns"]

        print(f"Loaded XGBoost model for {ticker} (trained: {self.last_trained})")


# --- Quick test ---
if __name__ == "__main__":
    from data_loader import load_etf_data
    from features import add_features, prepare_training_data

    ticker = "XIU.TO"
    print(f"\n{'='*50}")
    print(f"Testing models with {ticker}")
    print(f"{'='*50}")

    # Load data
    df = load_etf_data(ticker)
    df_feat = add_features(df)

    # --- Test Prophet ---
    print(f"\n--- Prophet Model ---")
    prophet_model = ProphetModel()
    prophet_model.train(df, ticker)

    predictions = prophet_model.predict(days=30)
    print(f"\nPredictions (next 5 business days):")
    # Show only future predictions (after last historical date)
    future_preds = predictions[predictions.index > df.index.max()]
    print(future_preds.head().to_string())

    prophet_model.save()

    # --- Test XGBoost ---
    print(f"\n--- XGBoost Model ---")
    X, y = prepare_training_data(df_feat)

    xgb_model = XGBoostModel()
    metrics = xgb_model.train(X, y, ticker)

    print(f"\nFeature Importance (top 5):")
    print(xgb_model.get_feature_importance().head().to_string(index=False))

    xgb_model.save()

    print(f"\nDone! Both models trained and saved.")
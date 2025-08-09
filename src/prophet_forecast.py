# src/prophet_forecast.py (updated)
import pandas as pd
import logging
from prophet import Prophet
from model_config import PROPHET_PARAMS, FORECAST_YEARS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prophet_forecast(df):
    """
    Generates Prophet forecast for the next FORECAST_YEARS periods.

    Args:
        df (pd.DataFrame): DataFrame with 'Year' and 'Inflation Rate' columns.

    Returns:
        pd.DataFrame: DataFrame with 'ds' (date) and 'yhat' (forecast) columns.
    """
    required_cols = {'Year', 'Inflation Rate'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        logger.error(f"Missing columns: {missing_cols}")
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")
    if df['Inflation Rate'].isnull().any():
        logger.warning("Missing values detected in 'Inflation Rate'. Dropping rows.")
    before = len(df)
    df = df.dropna(subset=['Year', 'Inflation Rate']).reset_index(drop=True)
    after = len(df)
    if after < before:
        logger.info(f"Dropped {before - after} rows due to missing or invalid data.")
    if len(df) < 5:
        logger.error("Not enough data points for Prophet forecasting (minimum 5 required).")
        raise ValueError("Not enough data points for Prophet forecasting (minimum 5 required).")

    prophet_df = pd.DataFrame({
        'ds': pd.to_datetime(df['Year'], format='%Y'),
        'y': df['Inflation Rate']
    })
    model = Prophet(**PROPHET_PARAMS)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=FORECAST_YEARS, freq='Y')
    forecast = model.predict(future)
    logger.info("Prophet forecast generated successfully.")
    tail = forecast.tail(FORECAST_YEARS)
    return tail[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
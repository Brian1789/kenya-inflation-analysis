import pandas as pd
import logging
from statsmodels.tsa.arima.model import ARIMA
from model_config import ARIMA_ORDER, FORECAST_YEARS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def arima_forecast(df):
    """
    Generates ARIMA forecast for the next `forecast_steps` periods.

    Args:
        df (pd.DataFrame): DataFrame with 'Year' and 'Inflation Rate' columns.
        forecast_steps (int): Number of future periods to forecast.

    Returns:
        pd.DataFrame: DataFrame with 'Year' and 'Forecast' columns.
    """
    # Input validation
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
        logger.error("Not enough data points for ARIMA forecasting (minimum 5 required).")
        raise ValueError("Not enough data points for ARIMA forecasting (minimum 5 required).")

    # Fit ARIMA model
    model = ARIMA(df['Inflation Rate'], order=ARIMA_ORDER)
    model_fit = model.fit()

    # Forecast future periods
    forecast = model_fit.forecast(steps=FORECAST_YEARS)
    # Fix: get integer year for addition
    last_year = df['Year'].dt.year.iloc[-1] if hasattr(df['Year'], 'dt') else int(df['Year'].iloc[-1])
    forecast_years = [last_year + i for i in range(1, FORECAST_YEARS + 1)]

    forecast_df = pd.DataFrame({
        'Year': forecast_years,
        'Forecast': forecast.values
    })
    logger.info("ARIMA forecast generated successfully.")
    return forecast_df
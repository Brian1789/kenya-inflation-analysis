# src/prophet_forecast.py (updated)
import logging
import pandas as pd
from model_config import PROPHET_PARAMS, FORECAST_YEARS

try:
    from prophet import Prophet
except Exception:
    # Prophet may be named fbprophet in some environments; try fallback
    try:
        from fbprophet import Prophet  # type: ignore
    except Exception:
        Prophet = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prophet_forecast(df: pd.DataFrame, params=None, periods=None):
    """
    Returns a DataFrame with columns: ds, yhat, yhat_lower, yhat_upper
    """
    params = params or PROPHET_PARAMS
    periods = periods or FORECAST_YEARS

    if Prophet is None:
        logger.warning("Prophet not available in environment.")
        # Return empty predictable structure
        last_year = df['Year'].dt.year.iloc[-1] if hasattr(df['Year'], 'dt') else int(df['Year'].iloc[-1])
        years = [last_year + i for i in range(1, periods+1)]
        return pd.DataFrame({'ds': pd.to_datetime(years, format='%Y'), 'yhat': [float('nan')]*periods, 'yhat_lower': [float('nan')]*periods, 'yhat_upper': [float('nan')]*periods})

    prophet_df = pd.DataFrame({'ds': pd.to_datetime(df['Year'].dt.year.astype(str)), 'y': df['Inflation Rate'].astype(float)})
    model = Prophet(**params)
    try:
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=periods, freq='Y')
        forecast = model.predict(future)
        tail = forecast.tail(periods).reset_index(drop=True)
        logger.info("Prophet forecast generated.")
        return tail[['ds','yhat','yhat_lower','yhat_upper']]
    except Exception as e:
        logger.exception("Prophet forecasting failed: %s", e)
        last_year = df['Year'].dt.year.iloc[-1] if hasattr(df['Year'], 'dt') else int(df['Year'].iloc[-1])
        years = [last_year + i for i in range(1, periods+1)]
        return pd.DataFrame({'ds': pd.to_datetime(years, format='%Y'), 'yhat': [float('nan')]*periods, 'yhat_lower': [float('nan')]*periods, 'yhat_upper': [float('nan')]*periods})
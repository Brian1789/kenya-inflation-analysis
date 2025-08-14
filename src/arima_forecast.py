import logging
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from model_config import ARIMA_ORDER, FORECAST_YEARS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def arima_forecast(df: pd.DataFrame, order=None, steps=None):
    """
    Produces ARIMA forecast DataFrame with columns:
    Year, Forecast, Forecast_lower, Forecast_upper
    """
    order = order or ARIMA_ORDER
    steps = steps or FORECAST_YEARS

    series = df['Inflation Rate'].astype(float).reset_index(drop=True)
    if series.empty:
        return pd.DataFrame(columns=['Year','Forecast','Forecast_lower','Forecast_upper'])

    try:
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        forecast_obj = model_fit.get_forecast(steps=steps)
        forecast = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int()

        last_year = df['Year'].dt.year.iloc[-1] if hasattr(df['Year'], 'dt') else int(df['Year'].iloc[-1])
        forecast_years = [last_year + i for i in range(1, steps+1)]
        forecast_df = pd.DataFrame({
            'Year': forecast_years,
            'Forecast': forecast.values,
            'Forecast_lower': conf_int.iloc[:,0].values,
            'Forecast_upper': conf_int.iloc[:,1].values
        })
        logger.info("ARIMA forecast generated.")
        return forecast_df
    except Exception as e:
        logger.exception("ARIMA forecasting failed: %s", e)
        # Return empty structured DF on failure
        last_year = df['Year'].dt.year.iloc[-1] if hasattr(df['Year'], 'dt') else int(df['Year'].iloc[-1])
        forecast_years = [last_year + i for i in range(1, (steps or 1)+1)]
        return pd.DataFrame({
            'Year': forecast_years,
            'Forecast': [float('nan')] * len(forecast_years),
            'Forecast_lower': [float('nan')] * len(forecast_years),
            'Forecast_upper': [float('nan')] * len(forecast_years)
        })
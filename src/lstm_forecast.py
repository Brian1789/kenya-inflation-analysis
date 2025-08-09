import pandas as pd
import numpy as np
import logging
from model_config import FORECAST_YEARS, LSTM_LOOK_BACK, LSTM_EPOCHS, LSTM_UNITS
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def lstm_forecast(df, look_back=LSTM_LOOK_BACK):
    """
    Generates LSTM forecast for the next `forecast_years` periods.
    Args:
        df (pd.DataFrame): DataFrame with 'Year' and 'Inflation Rate' columns.
        forecast_years (int): Number of future periods to forecast.
        look_back (int): Number of previous time steps to use as input.
    Returns:
        pd.DataFrame: DataFrame with 'Year' and 'Forecast' columns.
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
    if len(df) < look_back + 1:
        logger.error(f"Not enough data points for LSTM forecasting (minimum {look_back + 1} required).")
        raise ValueError(f"Not enough data points for LSTM forecasting (minimum {look_back + 1} required).")

    # Prepare data
    values = df['Inflation Rate'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # Create sequences
    X, y = [], []
    for i in range(len(scaled) - look_back):
        X.append(scaled[i:i+look_back, 0])
        y.append(scaled[i+look_back, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(Input(shape=(look_back, 1)))
    model.add(LSTM(LSTM_UNITS, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=LSTM_EPOCHS, verbose=0)

    # Forecast future values
    last_seq = scaled[-look_back:].reshape((1, look_back, 1))
    forecasts = []
    for _ in range(FORECAST_YEARS):
        next_pred = model.predict(last_seq, verbose=0)[0][0]
        forecasts.append(next_pred)
        last_seq = np.append(last_seq[:, 1:, :], [[[next_pred]]], axis=1)

    # Inverse scale
    forecasts = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
    # Fix: get integer year for addition
    last_year = df['Year'].dt.year.iloc[-1] if hasattr(df['Year'], 'dt') else int(df['Year'].iloc[-1])
    forecast_years_list = [last_year + i for i in range(1, FORECAST_YEARS + 1)]

    # Fix: include nan columns for consistency
    forecast_df = pd.DataFrame({
        'Year': forecast_years_list,
        'Forecast': forecasts,
        'Forecast_lower': [float('nan')] * len(forecasts),
        'Forecast_upper': [float('nan')] * len(forecasts)
    })
    logger.info("LSTM forecast generated successfully.")
    return forecast_df
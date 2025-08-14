import logging
import numpy as np
import pandas as pd
from model_config import FORECAST_YEARS, LSTM_LOOK_BACK, LSTM_EPOCHS, LSTM_UNITS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def lstm_forecast(df: pd.DataFrame, look_back=None, epochs=None, units=None):
    """
    Lightweight LSTM-style forecasting. If TF is not present, fallback to naive persistence forecast.
    Returns DataFrame with Year, Forecast, Forecast_lower (NaN), Forecast_upper (NaN)
    """
    look_back = look_back or LSTM_LOOK_BACK
    epochs = epochs or LSTM_EPOCHS
    units = units or LSTM_UNITS

    series = df['Inflation Rate'].astype(float).reset_index(drop=True)
    if series.empty:
        return pd.DataFrame(columns=['Year','Forecast','Forecast_lower','Forecast_upper'])

    # Try to use TensorFlow if available; otherwise fall back to persistence forecast
    try:
        import tensorflow as tf  # type: ignore
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data = scaler.fit_transform(series.values.reshape(-1,1))

        # build sequences
        X, y = [], []
        for i in range(look_back, len(data)):
            X.append(data[i-look_back:i, 0])
            y.append(data[i, 0])
        if not X:
            # Not enough data for sequences; fallback to last-value forecast
            raise ValueError("Not enough data for LSTM sequences")

        X = np.array(X).reshape(-1, look_back, 1)
        y = np.array(y)

        model = Sequential()
        model.add(LSTM(units, input_shape=(look_back,1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=min(epochs,50), verbose=0)

        # Forecast iteratively
        last_seq = data[-look_back:,0].tolist()
        preds = []
        for _ in range(FORECAST_YEARS):
            seq_in = np.array(last_seq[-look_back:]).reshape(1, look_back, 1)
            p = model.predict(seq_in, verbose=0)[0,0]
            preds.append(p)
            last_seq.append(p)
        preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
        last_year = df['Year'].dt.year.iloc[-1] if hasattr(df['Year'], 'dt') else int(df['Year'].iloc[-1])
        years = [last_year + i for i in range(1, len(preds)+1)]
        return pd.DataFrame({'Year': years, 'Forecast': preds, 'Forecast_lower':[float('nan')]*len(preds), 'Forecast_upper':[float('nan')]*len(preds)})
    except Exception as e:
        logger.info("TensorFlow LSTM not available or failed (%s). Falling back to persistence forecast.", e)
        last_val = series.iloc[-1]
        last_year = df['Year'].dt.year.iloc[-1] if hasattr(df['Year'], 'dt') else int(df['Year'].iloc[-1])
        years = [last_year + i for i in range(1, FORECAST_YEARS+1)]
        preds = [last_val]*FORECAST_YEARS
        return pd.DataFrame({'Year': years, 'Forecast': preds, 'Forecast_lower':[float('nan')]*len(preds), 'Forecast_upper':[float('nan')]*len(preds)})
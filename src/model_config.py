# ARIMA Configuration
ARIMA_ORDER = (1, 1, 2)  # (p, d, q) - Tune these for best results

# Prophet Configuration
PROPHET_PARAMS = {
    'yearly_seasonality': True,           # Set to True/False as needed
    'changepoint_prior_scale': 0.05,      # Lower values make the trend less flexible
    'seasonality_mode': 'multiplicative', # 'additive' or 'multiplicative'
    # Add more Prophet parameters here for fine-tuning
}

# LSTM Configuration
LSTM_LOOK_BACK = 44        # Number of previous time steps to use as input
LSTM_EPOCHS = 200         # Number of training epochs
LSTM_UNITS = 100           # Number of LSTM units

# General Settings
FORECAST_YEARS = 5        # Number of future periods to forecast
# Kenya Inflation Analysis Dashboard

## Overview

This project provides an interactive dashboard for analyzing and forecasting Kenya's inflation rates using ARIMA, Prophet, and LSTM time series models. The dashboard is built with [Streamlit](https://streamlit.io/) and allows users to upload their own inflation data, view exploratory data analysis (EDA), and compare forecasts from different models.

## Features

- **CSV Upload:** Easily upload your own inflation data.
- **Data Preprocessing:** Automatic cleaning and validation of input data.
- **Exploratory Data Analysis:** Visualize historical trends, summary statistics, and autocorrelation plots.
- **Forecasting:** Generate future inflation forecasts using ARIMA, Prophet, and LSTM models.
- **Forecast Comparison:** Interactive plots to compare model predictions.
- **Configurable Models:** Easily adjust model parameters via `src/model_config.py`.

## Project Structure

```
kenya-inflation-analysis/
├── src/
│   ├── streamlit_app.py        # Main Streamlit dashboard
│   ├── preprocess.py           # Data cleaning and validation
│   ├── arima_forecast.py       # ARIMA forecasting logic
│   ├── prophet_forecast.py     # Prophet forecasting logic
│   ├── lstm_forecast.py        # LSTM forecasting logic
│   ├── eda.py                  # EDA plots and statistics
│   ├── visualize.py            # Visualization utilities
│   ├── model_config.py         # Model parameters and settings
├── README.md                   # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/)

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/kenya-inflation-analysis.git
    cd kenya-inflation-analysis
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Dashboard

```bash
streamlit run src/streamlit_app.py
```

Open your browser and navigate to the provided local URL to interact with the dashboard.

## Usage

1. **Upload Data:** Click "Upload CSV" and select your inflation data file. The file should contain at least `Year` and `Inflation Rate` columns.
2. **View EDA:** Explore historical trends and summary statistics.
3. **Forecast:** View ARIMA, Prophet, and LSTM forecasts for future years.
4. **Compare Models:** Analyze the forecast comparison plot to evaluate model performance.

## Configuration

Model parameters can be adjusted in `src/model_config.py`:

```python
ARIMA_ORDER = (0, 1, 2)
PROPHET_PARAMS = {
    'yearly_seasonality': False,
    'changepoint_prior_scale': 0.1
}
FORECAST_YEARS = 5
```

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements and bug fixes.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Statsmodels](https://www.statsmodels.org/)
- [Prophet](https://facebook.github.io/prophet/)
- [TensorFlow/Keras](https://www.tensorflow.org/)
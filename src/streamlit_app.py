import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from preprocess import preprocess_data
from arima_forecast import arima_forecast
from prophet_forecast import prophet_forecast
from lstm_forecast import lstm_forecast
from eda import display_eda_plots
from visualize import plot_forecast_comparison
from model_config import ARIMA_ORDER, FORECAST_YEARS, LSTM_LOOK_BACK, LSTM_EPOCHS, LSTM_UNITS, PROPHET_PARAMS


st.set_page_config(page_title="Kenya Inflation Dashboard", layout="wide")

def main():
    st.title("Kenya Inflation Forecast Dashboard")

    # Sidebar controls
    st.sidebar.header("Controls")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    st.sidebar.subheader("Model Parameters")
    arima_p = st.sidebar.slider("ARIMA p", 0, 5, ARIMA_ORDER[0])
    arima_d = st.sidebar.slider("ARIMA d", 0, 2, ARIMA_ORDER[1])
    arima_q = st.sidebar.slider("ARIMA q", 0, 5, ARIMA_ORDER[2])
    forecast_years = st.sidebar.slider("Forecast Years", 1, 10, FORECAST_YEARS)
    lstm_epochs = st.sidebar.slider("LSTM Epochs", 10, 300, LSTM_EPOCHS)
    lstm_units = st.sidebar.slider("LSTM Units", 10, 200, LSTM_UNITS)
    prophet_changepoint = st.sidebar.slider("Prophet Changepoint Prior Scale", 0.01, 0.5, PROPHET_PARAMS.get('changepoint_prior_scale', 0.05), 0.01)
    prophet_seasonality = st.sidebar.selectbox("Prophet Seasonality Mode", ['additive', 'multiplicative'], index=0 if PROPHET_PARAMS.get('seasonality_mode', 'additive') == 'additive' else 1)

    if uploaded_file:
        try:
            raw_df = pd.read_csv(uploaded_file)
            cleaned_df = preprocess_data(raw_df)

            # Dynamically set max look_back based on data length
            max_look_back = max(1, len(cleaned_df) - 1)
            lstm_look_back = st.sidebar.slider(
                "LSTM Look Back",
                min_value=1,
                max_value=max_look_back,
                value=min(LSTM_LOOK_BACK, max_look_back),
                key="lstm_look_back"
            )
            lstm_epochs = st.sidebar.slider("LSTM Epochs", 10, 300, LSTM_EPOCHS, key="lstm_epochs")
            lstm_units = st.sidebar.slider("LSTM Units", 10, 200, LSTM_UNITS, key="lstm_units")

            # Update model configs based on sidebar
            arima_order = (arima_p, arima_d, arima_q)
            prophet_params = {
                'yearly_seasonality': True,
                'changepoint_prior_scale': prophet_changepoint,
                'seasonality_mode': prophet_seasonality
            }

            # Tabs for EDA, Forecasts, Comparison
            tab1, tab2, tab3 = st.tabs(["EDA", "Forecasts", "Comparison"])

            with tab1:
                display_eda_plots(cleaned_df)

            with st.spinner("Generating forecasts..."):
                arima_results = arima_forecast(cleaned_df)
                prophet_results = prophet_forecast(cleaned_df)
                if len(cleaned_df) < lstm_look_back + 1:
                    lstm_results = pd.DataFrame({'Year': [], 'Forecast': []})
                    st.warning(f"Not enough data points for LSTM forecasting (minimum {lstm_look_back + 1} required). Please reduce 'LSTM Look Back' or upload more data.")
                else:
                    lstm_results = lstm_forecast(cleaned_df, look_back=lstm_look_back)

            with tab2:
                st.subheader("Forecast Data")
                st.write("### ARIMA Forecast")
                st.dataframe(arima_results)
                st.download_button("Download ARIMA Results", arima_results.to_csv(index=False), "arima_forecast.csv")
                st.write("### Prophet Forecast")
                st.dataframe(prophet_results)
                st.download_button("Download Prophet Results", prophet_results.to_csv(index=False), "prophet_forecast.csv")
                st.write("### LSTM Forecast")
                st.dataframe(lstm_results)
                st.download_button("Download LSTM Results", lstm_results.to_csv(index=False), "lstm_forecast.csv")

            with tab3:
                st.write("### Forecast Comparison (ARIMA vs Prophet)")
                fig = plot_forecast_comparison(arima_results, prophet_results)
                st.pyplot(fig)

                st.write("### LSTM Forecast")
                fig_lstm, ax_lstm = plt.subplots(figsize=(10, 5))
                ax_lstm.plot(lstm_results['Year'], lstm_results['Forecast'], marker='o', color='tab:green', label='LSTM Forecast')
                ax_lstm.set_xlabel("Year")
                ax_lstm.set_ylabel("Inflation Rate (%)")
                ax_lstm.set_title("LSTM Forecast")
                ax_lstm.grid(True, linestyle='--', alpha=0.7)
                ax_lstm.legend()
                st.pyplot(fig_lstm)

            st.success("Forecasts generated successfully!")

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()

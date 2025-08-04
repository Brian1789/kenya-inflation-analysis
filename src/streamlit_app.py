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


st.set_page_config(page_title="Kenya Inflation Dashboard")

def main():
    st.title("Kenya Inflation Forecast")
    
    # Upload CSV
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        try:
            raw_df = pd.read_csv(uploaded_file)
            cleaned_df = preprocess_data(raw_df)
            display_eda_plots(cleaned_df)  # Show EDA

            # Generate forecasts
            arima_results = arima_forecast(cleaned_df)
            prophet_results = prophet_forecast(cleaned_df)
            lstm_results = lstm_forecast(cleaned_df)

            # Forecast comparison plot (ARIMA vs Prophet)
            st.write("### Forecast Comparison (ARIMA vs Prophet)")
            fig = plot_forecast_comparison(arima_results, prophet_results)
            st.pyplot(fig)

            # LSTM Forecast plot
            st.write("### LSTM Forecast")
            fig_lstm, ax_lstm = plt.subplots(figsize=(10, 5))
            ax_lstm.plot(lstm_results['Year'], lstm_results['Forecast'], marker='o', color='tab:green', label='LSTM Forecast')
            ax_lstm.set_xlabel("Year")
            ax_lstm.set_ylabel("Inflation Rate (%)")
            ax_lstm.set_title("LSTM Forecast")
            ax_lstm.grid(True, linestyle='--', alpha=0.7)
            ax_lstm.legend()
            st.pyplot(fig_lstm)

            # Show forecast data
            st.subheader("Forecast Data")
            st.write("### ARIMA Forecast")
            st.dataframe(arima_results)
            st.write("### Prophet Forecast")
            st.dataframe(prophet_results)
            st.write("### LSTM Forecast")
            st.dataframe(lstm_results)
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()

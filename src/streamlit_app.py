import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from preprocess import preprocess_data
from arima_forecast import arima_forecast
from prophet_forecast import prophet_forecast
from lstm_forecast import lstm_forecast
from model_config import ARIMA_ORDER, FORECAST_YEARS, LSTM_LOOK_BACK, LSTM_EPOCHS, LSTM_UNITS, PROPHET_PARAMS
from statsmodels.tsa.stattools import acf, pacf
import io
from visualize import compute_metrics

st.set_page_config(page_title="Kenya Inflation Dashboard", layout="wide")

@st.cache_data(ttl=600)
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_data(ttl=600)
def preprocess_cached(df):
    return preprocess_data(df)

def plot_forecast_plotly(historical_df, arima_df, prophet_df, lstm_df, models_to_show):
    fig = go.Figure()
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_df['Year'].dt.year if hasattr(historical_df['Year'], 'dt') else historical_df['Year'],
        y=historical_df['Inflation Rate'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='black', dash='dot')
    ))
    # Forecasts
    if 'ARIMA' in models_to_show and not arima_df.empty:
        fig.add_trace(go.Scatter(
            x=arima_df['Year'],
            y=arima_df['Forecast'],
            mode='lines+markers',
            name='ARIMA',
            hovertemplate="Model: %{name}<br>Year: %{x}<br>Forecast: %{y:.2f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=arima_df['Year'],
            y=arima_df['Forecast_upper'],
            fill=None, mode='lines', line=dict(color='rgba(0,0,0,0)'),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=arima_df['Year'],
            y=arima_df['Forecast_lower'],
            fill='tonexty', mode='lines', line=dict(color='rgba(0,0,0,0)'),
            fillcolor='rgba(31,119,180,0.2)', showlegend=True, name='ARIMA CI'
        ))
    if 'Prophet' in models_to_show and not prophet_df.empty:
        fig.add_trace(go.Scatter(
            x=prophet_df['ds'].dt.year if 'ds' in prophet_df else prophet_df['Year'],
            y=prophet_df['yhat'] if 'yhat' in prophet_df else prophet_df['Forecast'],
            mode='lines+markers',
            name='Prophet',
            hovertemplate="Model: %{name}<br>Year: %{x}<br>Forecast: %{y:.2f}<extra></extra>"
        ))
    if 'LSTM' in models_to_show and not lstm_df.empty:
        fig.add_trace(go.Scatter(
            x=lstm_df['Year'],
            y=lstm_df['Forecast'],
            mode='lines+markers',
            name='LSTM',
            hovertemplate="Model: %{name}<br>Year: %{x}<br>Forecast: %{y:.2f}<extra></extra>"
        ))
    fig.update_layout(title='Historical vs Forecast Comparison', xaxis_title='Year', yaxis_title='Inflation Rate (%)')
    return fig

def plot_acf_plotly(series, lags=20, title="ACF"):
    acf_vals = acf(series, nlags=lags)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals, marker_color='#1f77b4'))
    fig.update_layout(title=title, xaxis_title="Lag", yaxis_title="ACF", template="plotly_white")
    return fig

def plot_pacf_plotly(series, lags=20, title="PACF"):
    pacf_vals = pacf(series, nlags=lags)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, marker_color='#ff7f0e'))
    fig.update_layout(title=title, xaxis_title="Lag", yaxis_title="PACF", template="plotly_white")
    return fig

def plot_histogram_plotly(series, title="Histogram of Inflation Rate"):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=series, marker_color='#636EFA', nbinsx=20))
    fig.update_layout(title=title, xaxis_title="Inflation Rate (%)", yaxis_title="Count", template="plotly_white")
    return fig

def plot_timeseries_plotly(df, title="Inflation Rate Over Time"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Year'].dt.year if hasattr(df['Year'], 'dt') else df['Year'],
        y=df['Inflation Rate'],
        mode='lines+markers',
        name='Inflation Rate',
        line=dict(color='#00CC96')
    ))
    fig.update_layout(title=title, xaxis_title="Year", yaxis_title="Inflation Rate (%)", template="plotly_white")
    return fig

def main():
    st.title("Kenya Inflation Forecast Dashboard")

    # Sidebar controls
    st.sidebar.header("Controls")
    st.sidebar.info("Upload your CSV and adjust model parameters. Hover over chart points for details.")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    st.sidebar.subheader("Model Parameters")
    arima_p = st.sidebar.slider("ARIMA p", 0, 5, ARIMA_ORDER[0], key="arima_p")
    arima_d = st.sidebar.slider("ARIMA d", 0, 2, ARIMA_ORDER[1], key="arima_d")
    arima_q = st.sidebar.slider("ARIMA q", 0, 5, ARIMA_ORDER[2], key="arima_q")
    forecast_years = st.sidebar.slider("Forecast Years", 1, 10, FORECAST_YEARS, key="forecast_years")
    prophet_changepoint = st.sidebar.slider("Prophet Changepoint Prior Scale", 0.01, 0.5, PROPHET_PARAMS.get('changepoint_prior_scale', 0.05), 0.01, key="prophet_changepoint")
    prophet_seasonality = st.sidebar.selectbox("Prophet Seasonality Mode", ['additive', 'multiplicative'], index=0 if PROPHET_PARAMS.get('seasonality_mode', 'additive') == 'additive' else 1, key="prophet_seasonality")

    with st.sidebar.expander("LSTM Advanced Settings"):
        lstm_epochs = st.number_input("LSTM Epochs", min_value=10, max_value=300, value=LSTM_EPOCHS, step=10, key="lstm_epochs")
        lstm_units = st.number_input("LSTM Units", min_value=10, max_value=200, value=LSTM_UNITS, step=10, key="lstm_units")

    model_options = ['ARIMA', 'Prophet', 'LSTM']
    models_to_show = st.sidebar.multiselect(
        "Select models to compare",
        options=model_options,
        default=model_options
    )

    dataset_choice = st.sidebar.selectbox("Select Dataset", ["Monthly", "Quarterly"])

    if uploaded_file:
        try:
            raw_df = load_data(uploaded_file)
            cleaned_df = preprocess_cached(raw_df)
            historical_df = cleaned_df.copy()

            # Dynamically set max look_back based on data length
            max_look_back = max(1, len(cleaned_df) - 1)
            lstm_look_back = st.sidebar.slider(
                "LSTM Look Back",
                min_value=1,
                max_value=max_look_back,
                value=min(LSTM_LOOK_BACK, max_look_back),
                key="lstm_look_back"
            )

            # Update model configs based on sidebar
            arima_order = (arima_p, arima_d, arima_q)
            prophet_params = {
                'yearly_seasonality': True,
                'changepoint_prior_scale': prophet_changepoint,
                'seasonality_mode': prophet_seasonality
            }

            tab1, tab2, tab3 = st.tabs(["📊 EDA", "📈 Forecasts", "🔀 Comparison"])

            with st.spinner("Generating forecasts..."):
                arima_results = arima_forecast(cleaned_df)
                prophet_results = prophet_forecast(cleaned_df)
                if len(cleaned_df) < lstm_look_back + 1:
                    lstm_results = pd.DataFrame({'Year': [], 'Forecast': []})
                    st.warning(f"Not enough data points for LSTM forecasting (minimum {lstm_look_back + 1} required). Please reduce 'LSTM Look Back' or upload more data.")
                else:
                    lstm_results = lstm_forecast(cleaned_df, look_back=lstm_look_back)

            # Performance Metrics DataFrame
            N = min(len(cleaned_df), FORECAST_YEARS)
            actual = cleaned_df['Inflation Rate'].iloc[-N:]

            arima_mape, arima_rmse, arima_mae = compute_metrics(actual, arima_results['Forecast'][:N])
            prophet_mape, prophet_rmse, prophet_mae = compute_metrics(actual, prophet_results['yhat'][:N])
            lstm_mape, lstm_rmse, lstm_mae = compute_metrics(actual, lstm_results['Forecast'][:N])

            # Find best model by MAPE (lowest error)
            best_model_accuracy = 100 - min(arima_mape, prophet_mape, lstm_mape)

            perf_df = pd.DataFrame({
                "Model": ["ARIMA", "Prophet", "LSTM"],
                "MAPE": [arima_mape, prophet_mape, lstm_mape],
                "RMSE": [arima_rmse, prophet_rmse, lstm_rmse],
                "MAE": [arima_mae, prophet_mae, lstm_mae]
            })

            with tab1:
                st.subheader("Summary Statistics (Inflation Rate)")
                stats = cleaned_df['Inflation Rate'].describe().to_frame().T
                stats.index = ['Inflation Rate']
                st.dataframe(stats)
                st.subheader("Inflation Rate Over Time")
                st.plotly_chart(plot_timeseries_plotly(cleaned_df), use_container_width=True)
                st.subheader("Histogram of Inflation Rate")
                st.plotly_chart(plot_histogram_plotly(cleaned_df['Inflation Rate']), use_container_width=True)
                st.subheader("Autocorrelation (ACF)")
                st.plotly_chart(plot_acf_plotly(cleaned_df['Inflation Rate'], lags=20), use_container_width=True)
                st.subheader("Partial Autocorrelation (PACF)")
                st.plotly_chart(plot_pacf_plotly(cleaned_df['Inflation Rate'], lags=20), use_container_width=True)

            with tab2:
                col1, col2, col3 = st.columns(3)
                col1.metric("Latest Inflation", f"{cleaned_df['Inflation Rate'].iloc[-1]:.2f}%")
                col2.metric("Forecast Range", f"{forecast_years} years")
                col3.metric("Best Model Accuracy", f"{best_model_accuracy:.2f}%")

                with col1:
                    st.write("### ARIMA Forecast")
                    st.dataframe(arima_results)
                    st.download_button("Download ARIMA Results", arima_results.to_csv(index=False), "arima_forecast.csv")
                with col2:
                    st.write("### Prophet Forecast")
                    st.dataframe(prophet_results)
                    st.download_button("Download Prophet Results", prophet_results.to_csv(index=False), "prophet_forecast.csv")
                with col3:
                    st.write("### LSTM Forecast")
                    st.dataframe(lstm_results)
                    st.download_button("Download LSTM Results", lstm_results.to_csv(index=False), "lstm_forecast.csv")

                st.dataframe(perf_df)

            with tab3:
                st.write("### Historical vs Forecast Comparison (Interactive)")
                fig = plot_forecast_plotly(historical_df, arima_results, prophet_results, lstm_results, models_to_show)
                # Add vertical line and annotation for currency devaluation
                peak_value = historical_df['Inflation Rate'].max()
                fig.add_vline(x=1993, line_dash="dash", line_color="red")
                fig.add_annotation(x=1993, y=peak_value, text="Currency devaluation", showarrow=True)
                st.plotly_chart(fig, use_container_width=True)

            st.success("Forecasts generated successfully!")

        except Exception as e:
            st.error(f"Error: {e}")

    st.sidebar.markdown("### ⚙️ Model Controls")
    template_csv = "Year,Inflation Rate\n1990,15.2\n1991,18.5\n"
    st.sidebar.download_button("Download Template CSV", data=template_csv, file_name="template.csv")

    """st.markdown(
        """
    """    <div style='display: flex; align-items: center; justify-content: center; margin-bottom: 2em;'>
            <img src='https://knbs.or.ke/wp-content/uploads/2021/11/KNBS-Logo.png' height='60' style='margin-right:20px;'>
            <img src='https://festival.globaldatafest.org/logo.png' height='60' style='margin-right:20px;'>
            <img src='https://www.scb.se/ImageVaultFiles/id_20882/cf_1445/statistics-sweden-logo.png' height='60' style='margin-right:20px;'>
            <span style='font-size:2em; color:#1f77b4; font-weight:bold; margin-left:20px;'>
                Kenya Inflation Analysis Dashboard
            </span>
        </div>
        """, """unsafe_allow_html=True)"""

if __name__ == "__main__":
    main()

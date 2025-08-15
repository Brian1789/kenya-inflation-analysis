import os
import io
import logging
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import inspect
from preprocess import preprocess_data
from eda import display_eda_plots
from arima_forecast import arima_forecast
from prophet_forecast import prophet_forecast
from lstm_forecast import lstm_forecast
from visualize import compute_metrics
from model_config import ARIMA_ORDER, FORECAST_YEARS, LSTM_LOOK_BACK, LSTM_EPOCHS, LSTM_UNITS, PROPHET_PARAMS

st.set_page_config(page_title="Kenya Inflation Forecast Dashboard", layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_data(ttl=600)
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_data(ttl=600)
def preprocess_cached(df):
    return preprocess_data(df)

# Color palette used across the app
COLOR_PALETTE = {
    "primary": "#00d1ff",
    "accent": "#ff7f50",
    "success": "#7df97d",
    "muted": "#9aaab1",
    "bg": "#0b1220",
    "panel": "#071322",
    "text": "#cbe7ff"
}

def plot_forecast_plotly(historical_df, arima_df, prophet_df, lstm_df, models_to_show):
    fig = go.Figure()
    # Historical
    fig.add_trace(go.Scattergl(
        x=historical_df['Year'].dt.year if hasattr(historical_df['Year'], 'dt') else historical_df['Year'],
        y=historical_df['Inflation Rate'],
        mode='lines+markers',
        name='Historical',
        line=dict(color=COLOR_PALETTE['muted'], dash='dot'),
        marker=dict(size=6, color=COLOR_PALETTE['muted'])
    ))
    # ARIMA with CI
    if 'ARIMA' in models_to_show and arima_df is not None and not arima_df.empty:
        fig.add_trace(go.Scatter(x=arima_df['Year'], y=arima_df['Forecast'], mode='lines+markers', name='ARIMA',
                                 line=dict(color=COLOR_PALETTE['primary'], width=2)))
        if 'Forecast_lower' in arima_df and 'Forecast_upper' in arima_df:
            fig.add_trace(go.Scatter(x=arima_df['Year'], y=arima_df['Forecast_upper'], mode='lines',
                                     line=dict(color='rgba(0,0,0,0)'), showlegend=False))
            fig.add_trace(go.Scatter(x=arima_df['Year'], y=arima_df['Forecast_lower'], mode='lines',
                                     line=dict(color='rgba(0,0,0,0)'), fill='tonexty',
                                     fillcolor='rgba(0,209,255,0.12)', name='ARIMA CI'))
    # Prophet with CI
    if 'Prophet' in models_to_show and prophet_df is not None and not prophet_df.empty:
        x = prophet_df['ds'].dt.year if 'ds' in prophet_df and hasattr(prophet_df['ds'], 'dt') else prophet_df.get('Year', prophet_df.get('ds'))
        y = prophet_df['yhat'] if 'yhat' in prophet_df else prophet_df.get('Forecast')
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='Prophet',
                                 line=dict(color=COLOR_PALETTE['accent'], width=2)))
        if 'yhat_lower' in prophet_df and 'yhat_upper' in prophet_df:
            fig.add_trace(go.Scatter(x=x, y=prophet_df['yhat_upper'], mode='lines', line=dict(color='rgba(0,0,0,0)'), showlegend=False))
            fig.add_trace(go.Scatter(x=x, y=prophet_df['yhat_lower'], mode='lines', line=dict(color='rgba(0,0,0,0)'), fill='tonexty', fillcolor='rgba(255,127,80,0.10)', name='Prophet CI'))
    # LSTM (no CI)
    if 'LSTM' in models_to_show and lstm_df is not None and not lstm_df.empty:
        fig.add_trace(go.Scatter(x=lstm_df['Year'], y=lstm_df['Forecast'], mode='lines+markers', name='LSTM',
                                 line=dict(color=COLOR_PALETTE['success'], width=2)))

    # Layout: techy dark
    fig.update_layout(
        title='Historical vs Forecast Comparison',
        xaxis_title='Year',
        yaxis_title='Inflation Rate (%)',
        template="plotly_dark",
        paper_bgcolor=COLOR_PALETTE['bg'],
        plot_bgcolor=COLOR_PALETTE['panel'],
        font=dict(color=COLOR_PALETTE['text'], family="Source Sans Pro"),
        legend=dict(bgcolor='rgba(0,0,0,0.2)', bordercolor='rgba(255,255,255,0.05)')
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.03)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.03)')
    return fig

def header():
    # gradient header + small CSS for techy look (text-only, no KPI cards)
    st.markdown(
        f"""
        <style>
        .tech-header {{
            background: linear-gradient(90deg, {COLOR_PALETTE['panel']}, {COLOR_PALETTE['bg']});
            padding: 22px;
            border-radius: 8px;
            color: {COLOR_PALETTE['text']};
            margin-bottom: 18px;
            text-align:center;
        }}
        .tech-sub {{
            color: rgba(203,231,255,0.84);
            margin-top:6px;
            font-size:0.95rem;
        }}
        </style>
        <div class="tech-header">
          <h1 style="margin:0; font-weight:800; font-size:34px; color:{COLOR_PALETTE['primary']};">Kenya Inflation Forecast Dashboard</h1>
          <div class="tech-sub">KNBS • Global Data Festival • Statistics Sweden</div>
        </div>
        """, unsafe_allow_html=True
    )

def main():
    header()

    # Sidebar controls
    st.sidebar.markdown("### ⚙️ Model Controls")
    st.sidebar.info("Upload your CSV and adjust model parameters. Hover over chart points for details.")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    # Template CSV download
    template_csv = "Year,Inflation Rate\n1990,15.2\n1991,18.5\n"
    st.sidebar.download_button("Download Template CSV", data=template_csv, file_name="template.csv")

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

    if uploaded_file is None:
        st.info("Upload a CSV to see EDA and forecasts.")
        return

    try:
        raw_df = load_data(uploaded_file)
        cleaned_df = preprocess_cached(raw_df)
        historical_df = cleaned_df.copy()

        # ensure enough rows
        if cleaned_df.empty:
            st.error("Uploaded CSV contains no usable rows after preprocessing.")
            return

        # LSTM look back slider dynamic
        max_look_back = max(1, len(cleaned_df) - 1)
        lstm_look_back = st.sidebar.slider("LSTM Look Back", min_value=1, max_value=max_look_back, value=min(LSTM_LOOK_BACK, max_look_back), key="lstm_look_back")

        # Tabs
        tab1, tab2, tab3 = st.tabs(["📊 EDA", "📈 Forecasts", "🔀 Comparison"])

        # --- generate forecasts
        with st.spinner("Generating forecasts..."):
            arima_results = arima_forecast(cleaned_df, order=(arima_p, arima_d, arima_q), steps=forecast_years)
            prophet_results = prophet_forecast(cleaned_df, params={'changepoint_prior_scale': prophet_changepoint, 'seasonality_mode': prophet_seasonality}, periods=forecast_years)
            lstm_results = lstm_forecast(cleaned_df, look_back=lstm_look_back, epochs=lstm_epochs, units=lstm_units)

        # --- compute performance metrics (align last N actuals with first N forecasts where possible)
        N = min(len(cleaned_df), forecast_years)
        actual = cleaned_df['Inflation Rate'].iloc[-N:].reset_index(drop=True)

        def first_n(series_like, n):
            if series_like is None:
                return pd.Series([float('nan')]*n)
            s = pd.Series(series_like).reset_index(drop=True)
            return s.iloc[:n].reset_index(drop=True)

        arima_pred = first_n(arima_results['Forecast'] if 'Forecast' in arima_results.columns else arima_results.get('yhat', []), N)
        prophet_pred = first_n(prophet_results['yhat'] if 'yhat' in prophet_results.columns else prophet_results.get('Forecast', []), N)
        lstm_pred = first_n(lstm_results['Forecast'] if 'Forecast' in lstm_results.columns else lstm_results.get('yhat', []), N)

        arima_mape, arima_rmse, arima_mae = compute_metrics(actual, arima_pred)
        prophet_mape, prophet_rmse, prophet_mae = compute_metrics(actual, prophet_pred)
        lstm_mape, lstm_rmse, lstm_mae = compute_metrics(actual, lstm_pred)

        mape_candidates = [v for v in [arima_mape, prophet_mape, lstm_mape] if pd.notna(v)]
        best_model_accuracy = (100 - min(mape_candidates)) if mape_candidates else float('nan')

        perf_df = pd.DataFrame({
            "Model": ["ARIMA", "Prophet", "LSTM"],
            "MAPE": [arima_mape, prophet_mape, lstm_mape],
            "RMSE": [arima_rmse, prophet_rmse, lstm_rmse],
            "MAE": [arima_mae, prophet_mae, lstm_mae]
        })

        # --- EDA tab
        with tab1:
            st.subheader("Summary Statistics (Inflation Rate)")
            stats = cleaned_df['Inflation Rate'].describe().to_frame().T
            stats.index = ['Inflation Rate']
            st.dataframe(stats)
            st.subheader("Inflation Rate Over Time")
            figs = display_eda_plots(cleaned_df)
            st.plotly_chart(figs['timeseries'], use_container_width=True)
            st.subheader("Histogram of Inflation Rate")
            st.plotly_chart(figs['histogram'], use_container_width=True)
            st.subheader("Autocorrelation (ACF)")
            st.plotly_chart(figs['acf'], use_container_width=True)
            st.subheader("Partial Autocorrelation (PACF)")
            st.plotly_chart(figs['pacf'], use_container_width=True)

        # --- Forecasts tab
        with tab2:
            with st.expander("Key KPIs"):
                st.metric("Latest Inflation", f"{cleaned_df['Inflation Rate'].iloc[-1]:.2f}%")
                st.metric("Forecast Range", f"{forecast_years} years")
                st.metric("Best Model Accuracy", f"{best_model_accuracy:.2f}%" if pd.notna(best_model_accuracy) else "—")

            with st.expander("Model performance metrics"):
                st.dataframe(perf_df)

            cola, colb, colc = st.columns(3)
            with cola:
                st.write("### ARIMA Forecast")
                st.dataframe(arima_results)
                st.download_button("Download ARIMA Results", arima_results.to_csv(index=False), "arima_forecast.csv")
            with colb:
                st.write("### Prophet Forecast")
                st.dataframe(prophet_results)
                st.download_button("Download Prophet Results", prophet_results.to_csv(index=False), "prophet_forecast.csv")
            with colc:
                st.write("### LSTM Forecast")
                st.dataframe(lstm_results)
                st.download_button("Download LSTM Results", lstm_results.to_csv(index=False), "lstm_forecast.csv")

        # --- Comparison tab
        with tab3:
            st.write("### Historical vs Forecast Comparison (Interactive)")
            fig = plot_forecast_plotly(historical_df, arima_results, prophet_results, lstm_results, models_to_show)
            try:
                peak_value = historical_df['Inflation Rate'].max()
                fig.add_vline(x=1993, line_dash="dash", line_color="red")
                fig.add_annotation(x=1993, y=peak_value, text="Currency devaluation (1993)", showarrow=True)
            except Exception:
                pass
            st.plotly_chart(fig, use_container_width=True)

        st.success("Forecasts generated successfully!")

    except Exception as e:
        logger.exception("App error: %s", e)
        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()

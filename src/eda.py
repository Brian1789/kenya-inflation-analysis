import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
import os
import streamlit as st
import plotly.graph_objects as go

def validate_df(df):
    required_cols = {'Year', 'Inflation Rate'}
    if not required_cols.issubset(df.columns):
        raise ValueError("DataFrame must contain 'Year' and 'Inflation Rate' columns.")
    if df['Inflation Rate'].isnull().any():
        raise ValueError("Missing values detected in 'Inflation Rate'. Please clean your data.")

def generate_eda_plots(df, output_dir):
    """
    Generate EDA plots for inflation data and save to disk.
    """
    validate_df(df)
    os.makedirs(output_dir, exist_ok=True)
    # Convert 'Year' to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['Year']):
        df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    # Time series plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['Year'], df['Inflation Rate'], marker='o', color='#1f77b4')
    plt.title("Historical Inflation Rates", fontsize=14)
    plt.xlabel("Year")
    plt.ylabel("Inflation Rate (%)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "historical_trend.png"))
    plt.close()
    # ACF/PACF plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(df['Inflation Rate'], lags=10, ax=ax1, title='Autocorrelation (ACF)')
    plot_pacf(df['Inflation Rate'], lags=10, ax=ax2, method='ywm', title='Partial Autocorrelation (PACF)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "acf_pacf.png"))
    plt.close()

def plot_timeseries_plotly(df, title="Inflation Rate Over Time"):
    fig = go.Figure()
    x = df['Year'].dt.year if hasattr(df['Year'], 'dt') else df['Year']
    fig.add_trace(go.Scattergl(x=x, y=df['Inflation Rate'], mode='lines+markers', name='Inflation Rate',
                               line=dict(color='#00CC96')))
    fig.update_layout(title=title, xaxis_title="Year", yaxis_title="Inflation Rate (%)", template="plotly_white")
    return fig

def plot_histogram_plotly(series, title="Histogram of Inflation Rate"):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=series, marker_color='#636EFA', nbinsx=20))
    fig.update_layout(title=title, xaxis_title="Inflation Rate (%)", yaxis_title="Count", template="plotly_white")
    return fig

def plot_acf_plotly(series, lags=20, title="ACF"):
    acf_vals = acf(series.dropna(), nlags=lags)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals, marker_color='#1f77b4', name='ACF'))
    fig.update_layout(title=title, xaxis_title="Lag", yaxis_title="ACF", template="plotly_white")
    return fig

def plot_pacf_plotly(series, lags=20, title="PACF"):
    pacf_vals = pacf(series.dropna(), nlags=lags)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, marker_color='#ff7f0e', name='PACF'))
    fig.update_layout(title=title, xaxis_title="Lag", yaxis_title="PACF", template="plotly_white")
    return fig

def display_eda_plots(df):
    """
    Returns a dict of figures for the EDA tab (caller can st.plotly_chart them).
    """
    figs = {
        "timeseries": plot_timeseries_plotly(df),
        "histogram": plot_histogram_plotly(df['Inflation Rate']),
        "acf": plot_acf_plotly(df['Inflation Rate']),
        "pacf": plot_pacf_plotly(df['Inflation Rate'])
    }
    return figs

if __name__ == "__main__":
    # For standalone script usage
    data_path = os.path.join("..", "results", "cleaned_data.csv")
    output_dir = os.path.join("..", "results", "eda_plots")
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    if not pd.api.types.is_datetime64_any_dtype(df['Year']):
        df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    generate_eda_plots(df, output_dir)
    print(f"EDA plots saved to {output_dir}")
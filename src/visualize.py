import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_forecast_comparison(arima_df, prophet_df):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(arima_df['Year'], arima_df['Forecast'], label='ARIMA Forecast', marker='o')
    ax.plot(prophet_df['ds'].dt.year, prophet_df['yhat'], label='Prophet Forecast', marker='o')
    ax.set_xlabel('Year')
    ax.set_ylabel('Inflation Rate (%)')
    ax.set_title('ARIMA vs Prophet Forecast Comparison')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    return fig

def generate_report(arima_df, prophet_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig = plot_forecast_comparison(arima_df, prophet_df)
    fig_path = os.path.join(output_dir, "forecast_comparison.png")
    fig.savefig(fig_path)
    plt.close(fig)
    # Generate markdown report
    with open(os.path.join(output_dir, "final_report.md"), 'w') as f:
        f.write("# Kenya Inflation Forecast Report\n")
        f.write("![Forecast Comparison](forecast_comparison.png)\n")

def compute_metrics(actual, forecast):
    """
    Compute MAPE, RMSE, MAE between two iterables/Series.
    Returns (mape, rmse, mae). Handles NaNs by dropping aligned pairs.
    """
    a = pd.Series(actual).astype(float).reset_index(drop=True)
    f = pd.Series(forecast).astype(float).reset_index(drop=True)
    # align lengths
    n = min(len(a), len(f))
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    a = a.iloc[:n]
    f = f.iloc[:n]
    mask = a.notna() & f.notna() & (a != 0)
    if not mask.any():
        # if denominator would be 0 for MAPE, compute MAE/RMSE only
        mae = (a - f).abs().mean()
        rmse = np.sqrt(((a - f)**2).mean())
        return float('nan'), float(rmse), float(mae)
    a_masked = a[mask]
    f_masked = f[mask]
    mape = ( (a_masked - f_masked).abs() / a_masked.abs() ).mean() * 100
    rmse = np.sqrt(((a - f)**2).mean())
    mae = (a - f).abs().mean()
    return float(mape), float(rmse), float(mae)

if __name__ == "__main__":
    arima_path = os.path.join("..", "results", "forecasts", "arima_forecast.csv")
    prophet_path = os.path.join("..", "results", "forecasts", "prophet_forecast.csv")
    output_dir = os.path.join("..", "results")
    arima_df = pd.read_csv(arima_path)
    prophet_df = pd.read_csv(prophet_path)
    generate_report(arima_df, prophet_df, output_dir)
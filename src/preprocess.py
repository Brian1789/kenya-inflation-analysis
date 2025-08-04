import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data(df):
    required_cols = {'Year', 'Inflation Rate'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        logger.error(f"Missing columns: {missing_cols}")
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")
    df['Year'] = pd.to_datetime(df['Year'], format='%Y', errors='coerce')
    df['Inflation Rate'] = pd.to_numeric(df['Inflation Rate'], errors='coerce')
    before = len(df)
    df = df.dropna(subset=['Year', 'Inflation Rate']).reset_index(drop=True)
    after = len(df)
    if after < before:
        logger.info(f"Dropped {before - after} rows due to invalid or missing data.")
    return df
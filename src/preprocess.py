import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic preprocessing for the uploaded inflation CSV.
    - Ensures 'Year' and 'Inflation Rate' columns exist
    - Converts Year to datetime (year start)
    - Coerces Inflation Rate to numeric and drops rows without year or rate
    - Sorts by year
    """
    df = df.copy()
    expected = {'Year', 'Inflation Rate'}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {expected}")

    # Year -> datetime (use first day of year)
    try:
        # Accept numbers (e.g., 1990) or strings like '1990' or full dates
        df['Year'] = pd.to_datetime(df['Year'].astype(str).str.strip(), errors='coerce', format='%Y')
        # Fallback: try full datetime parse
        mask_na = df['Year'].isna()
        if mask_na.any():
            df.loc[mask_na, 'Year'] = pd.to_datetime(df.loc[mask_na, 'Year'].astype(str), errors='coerce')
    except Exception:
        df['Year'] = pd.to_datetime(df['Year'], errors='coerce')

    # Inflation Rate -> numeric
    df['Inflation Rate'] = pd.to_numeric(df['Inflation Rate'], errors='coerce')

    # Drop rows missing required values
    df = df.dropna(subset=['Year', 'Inflation Rate'])

    # Sort by Year
    df = df.sort_values('Year').reset_index(drop=True)

    return df
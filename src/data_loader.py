import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    return pd.read_csv(path)

def basic_info(df: pd.DataFrame):
    """Print basic dataset info."""
    print(df.shape)
    print("Missing values:\n", df.isna().sum())
    print("Unique engines:", df["unit_number"].nunique())
import pandas as pd
import numpy as np

def calculate_rul(df: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
# Calculate Remaining Useful Life (RUL).
    max_cycles = df.groupby("unit_number")["time_in_cycles"].max().reset_index()
    max_cycles.columns = ["unit_number", "max_cycles"]

    df = df.merge(max_cycles, on="unit_number", how="left")
    df["RUL"] = df["max_cycles"] - df["time_in_cycles"]
    df["RUL"] = df["RUL"].clip(upper=cap)
    
    df = df.drop(columns=["max_cycles"])
    return df

def drop_correlated_features(df: pd.DataFrame, threshold: float = 0.98) -> pd.DataFrame:
# Drop highly correlated sensor features.
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=to_drop), to_drop
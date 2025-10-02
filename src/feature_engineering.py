import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def rolling_features(df: pd.DataFrame, sensor_cols: list, op_cols: list, window: int = 5) -> pd.DataFrame:
# Generate rolling mean, std, slope, variance for features.
    df_copy = df.copy()
    df_copy[op_cols + sensor_cols] = df_copy.groupby("unit_number")[op_cols + sensor_cols].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # Standard Deviation
    df_copy[[f"{c}_std" for c in op_cols + sensor_cols]] = df_copy.groupby("unit_number")[op_cols + sensor_cols].rolling(window, min_periods=1).std().reset_index(level=0, drop=True)
    
    # Slope
    for col in op_cols + sensor_cols:
        df_copy[f"{col}_slope"] = df_copy.groupby("unit_number")[col].diff(periods=window).rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # Variance
    for col in op_cols + sensor_cols:
        df_copy[f"{col}_var"] = df_copy.groupby("unit_number")[col].rolling(window * 4, min_periods=1).var().reset_index(level=0, drop=True)
    
    return df_copy

def apply_pca(df: pd.DataFrame, cols: list, n_components: int = 10) -> pd.DataFrame:
# Apply PCA and return transformed features.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[cols])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
    return pd.concat([df.reset_index(drop=True), pca_df], axis=1), pca
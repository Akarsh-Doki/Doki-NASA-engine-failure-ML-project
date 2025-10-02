import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(8,6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.title("Predicted vs Actual RUL")
    plt.grid(True)
    plt.show()

def save_results(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Sample data (replace with your actual predictions and true values)
# These are placeholder values - you should replace them with your model's actual predictions
y_true = [100, 105, 98, 110, 95]  # Actual values
y_pred_rf = [102, 104, 97, 108, 94]  # RF predictions
y_pred_lstm = [101, 103, 96, 109, 93]  # LSTM predictions

def calculate_metrics(y_true, y_pred, model_name):
    """Calculate and print metrics for a model"""
    print(f"\n=== {model_name} Metrics ===")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_true, y_pred):.4f}")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")

# Calculate metrics for both models
calculate_metrics(y_true, y_pred_rf, "Random Forest")
calculate_metrics(y_true, y_pred_lstm, "LSTM")

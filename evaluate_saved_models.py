import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta

def load_data():
    """Load the GOOGL stock data"""
    try:
        data_path = os.path.join('media', 'stock_data', 'GOOGL_data.csv')
        print(f"Loading data from: {os.path.abspath(data_path)}")
        
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Basic feature engineering
        df['Return'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['MA7'] = df['Close'].rolling(7).mean()
        df['MA21'] = df['Close'].rolling(21).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        
        # Target: Next day's close price
        df['Target'] = df['Close'].shift(-1)
        
        # Drop rows with missing values
        df = df.dropna()
        
        print(f"Loaded {len(df)} rows of data")
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def evaluate_random_forest(model_path, test_data):
    """Evaluate the Random Forest model"""
    print("\n=== Random Forest Model Evaluation ===")
    try:
        # Load the model
        model = joblib.load(model_path)
        print(f"Loaded Random Forest model from {model_path}")
        
        # Prepare test data
        X_test = test_data[['Close', 'Volume', 'MA7', 'MA21', 'MA50', 'Return', 'Volume_Change']]
        y_test = test_data['Target']
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Directional accuracy
        actual_direction = np.sign(y_test.diff().dropna())
        pred_direction = np.sign(np.diff(y_pred))
        direction_match = (actual_direction == pred_direction).mean() * 100
        
        print("\nPerformance Metrics:")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R-squared (R²): {r2:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"Directional Accuracy: {direction_match:.2f}%")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            print("\nFeature Importances:")
            features = X_test.columns
            importances = model.feature_importances_
            for feature, importance in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
                print(f"{feature}: {importance:.4f}")
        
        # Plot predictions vs actual
        plot_predictions(test_data['Date'], y_test, y_pred, 'Random Forest')
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'direction_accuracy': direction_match
        }
        
    except Exception as e:
        print(f"Error evaluating Random Forest model: {str(e)}")
        return None

def evaluate_lstm(model_path, test_data):
    """Evaluate the LSTM model"""
    print("\n=== LSTM Model Evaluation ===")
    try:
        # Load the model
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded LSTM model from {model_path}")
        
        # Prepare test data (simplified - in practice, you'd need to preprocess the same way as during training)
        # This is a simplified evaluation - you'll need to adjust based on your LSTM's input requirements
        X_test = test_data[['Close']].values
        y_test = test_data['Target'].values
        
        # Reshape for LSTM (samples, time_steps, features)
        # Note: This is simplified - adjust based on your model's expected input shape
        X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        # Make predictions
        y_pred = model.predict(X_test_reshaped).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Directional accuracy
        actual_direction = np.sign(np.diff(y_test))
        pred_direction = np.sign(np.diff(y_pred))
        direction_match = (actual_direction == pred_direction).mean() * 100
        
        print("\nPerformance Metrics:")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"R-squared (R²): {r2:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"Directional Accuracy: {direction_match:.2f}%")
        
        # Plot predictions vs actual
        plot_predictions(test_data['Date'], y_test, y_pred, 'LSTM')
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'direction_accuracy': direction_match
        }
        
    except Exception as e:
        print(f"Error evaluating LSTM model: {str(e)}")
        return None

def plot_predictions(dates, y_true, y_pred, model_name):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(14, 7))
    plt.plot(dates, y_true, label='Actual', color='blue', alpha=0.7)
    plt.plot(dates, y_pred, label='Predicted', color='red', alpha=0.7)
    plt.title(f'{model_name} - Actual vs Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('reports', exist_ok=True)
    plot_path = os.path.join('reports', f'{model_name.lower().replace(" ", "_")}_predictions.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved prediction plot to: {os.path.abspath(plot_path)}")
    
    # Create interactive plot with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y_true, mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=dates, y=y_pred, mode='lines', name='Predicted', line=dict(color='red')))
    
    fig.update_layout(
        title=f'{model_name} - Actual vs Predicted Prices',
        xaxis_title='Date',
        yaxis_title='Price',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified'
    )
    
    # Save interactive plot
    html_path = os.path.join('reports', f'{model_name.lower().replace(" ", "_")}_predictions.html')
    fig.write_html(html_path)
    print(f"Saved interactive plot to: {os.path.abspath(html_path)}")

if __name__ == "__main__":
    # Load and prepare data
    df = load_data()
    if df is None:
        print("Failed to load data. Exiting.")
        exit(1)
    
    # Split data into train and test sets (80-20 split)
    train_size = int(len(df) * 0.8)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    
    print(f"\nData split:")
    print(f"- Training data: {len(train_data)} samples ({train_data['Date'].min().date()} to {train_data['Date'].max().date()})")
    print(f"- Test data: {len(test_data)} samples ({test_data['Date'].min().date()} to {test_data['Date'].max().date()})")
    
    # Evaluate models
    rf_metrics = evaluate_random_forest('models/rf_model.joblib', test_data)
    lstm_metrics = evaluate_lstm('models/simple_lstm_model.h5', test_data)
    
    # Compare models if both evaluations were successful
    if rf_metrics and lstm_metrics:
        print("\n=== Model Comparison ===")
        print(f"{'Metric':<25} {'Random Forest':<15} {'LSTM':<15}")
        print("-" * 50)
        for metric in ['mae', 'rmse', 'r2', 'mape', 'direction_accuracy']:
            print(f"{metric.upper():<25} {rf_metrics[metric]:<15.4f} {lstm_metrics[metric]:<15.4f}")
    
    print("\nEvaluation complete. Check the 'reports' directory for visualizations.")

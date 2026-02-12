import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data():
    """Load the stock data"""
    try:
        df = pd.read_csv('media/stock_data/GOOGL_data.csv')
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
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def get_rf_metrics(model_path, test_data):
    """Get metrics for Random Forest model"""
    try:
        model = joblib.load(model_path)
        X_test = test_data[['Close', 'Volume', 'MA7', 'MA21', 'MA50', 'Return', 'Volume_Change']]
        y_test = test_data['Target']
        
        y_pred = model.predict(X_test)
        
        return {
            'R2': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred)
        }
    except Exception as e:
        print(f"Error evaluating RF model: {str(e)}")
        return None

def get_lstm_metrics(model_path, test_data):
    """Get metrics for LSTM model"""
    try:
        model = tf.keras.models.load_model(model_path)
        X_test = test_data[['Close']].values.reshape(-1, 1, 1)
        y_test = test_data['Target'].values
        
        y_pred = model.predict(X_test).flatten()
        
        return {
            'R2': r2_score(y_test, y_pred),
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred)
        }
    except Exception as e:
        print(f"Error evaluating LSTM model: {str(e)}")
        return None

# Main execution
if __name__ == "__main__":
    print("Loading data and evaluating models...")
    
    # Load and prepare data
    df = load_data()
    if df is None:
        exit(1)
    
    # Split data (80% train, 20% test)
    train_size = int(len(df) * 0.8)
    test_data = df.iloc[train_size:]
    
    # Get metrics for both models
    rf_metrics = get_rf_metrics('models/enhanced_rf_model.joblib', test_data)
    lstm_metrics = get_lstm_metrics('models/lstm_model.keras', test_data)
    
    # Print results
    print("\n=== Model Performance Metrics ===")
    print(f"{'Metric':<10} {'Random Forest':<15} {'LSTM':<15}")
    print("-" * 40)
    
    if rf_metrics and lstm_metrics:
        for metric in ['R2', 'MAE', 'MSE']:
            print(f"{metric}: {rf_metrics[metric]:.6f} (RF)  |  {lstm_metrics[metric]:.6f} (LSTM)")
    elif rf_metrics:
        print("\nOnly Random Forest metrics available:")
        for metric, value in rf_metrics.items():
            print(f"{metric}: {value:.6f}")
    elif lstm_metrics:
        print("\nOnly LSTM metrics available:")
        for metric, value in lstm_metrics.items():
            print(f"{metric}: {value:.6f}")
    else:
        print("Failed to get metrics for either model")

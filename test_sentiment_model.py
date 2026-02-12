"""
Test script for sentiment-enhanced Random Forest model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from rf_model import EnhancedRandomForestModel
from enhanced_rf_model import SentimentEnhancedRFModel

def load_test_data(file_path):
    """Load test data from CSV file"""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def compare_models(data_path, test_size=0.2):
    """Compare regular RF model with sentiment-enhanced model"""
    # Load data
    df = load_test_data(data_path)
    
    # Split data
    train_size = int(len(df) * (1 - test_size))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    print(f"Training data: {len(train_df)} rows")
    print(f"Testing data: {len(test_df)} rows")
    
    # Train regular RF model
    print("\nTraining regular RF model...")
    rf_model = EnhancedRandomForestModel()
    rf_model.train(train_df)
    
    # Train sentiment-enhanced RF model
    print("\nTraining sentiment-enhanced RF model...")
    sentiment_rf_model = SentimentEnhancedRFModel()
    sentiment_rf_model.train(train_df)
    
    # Make predictions
    print("\nMaking predictions...")
    rf_predictions = []
    sentiment_rf_predictions = []
    actuals = []
    
    # Use a rolling window of historical data for predictions
    window_size = 200  # Increase window size to ensure sufficient historical data
    
    # Start predictions only after we have enough historical data
    for i in range(len(test_df)-window_size-1):
        # Create a DataFrame with historical data
        pred_df = test_df.iloc[i:i+window_size]
        actual = test_df['Close'].iloc[i+window_size]
        
        try:
            # Regular RF prediction
            rf_pred = rf_model.predict_next_day(pred_df)
            rf_predictions.append(rf_pred)
            
            # Sentiment-enhanced RF prediction
            # Ensure sentiment features are included
            sentiment_rf_pred = sentiment_rf_model.predict_next_day(pred_df)
            sentiment_rf_predictions.append(sentiment_rf_pred)
            
            actuals.append(actual)
        except Exception as e:
            print(f"Error on prediction {i}: {str(e)}")
            continue
    
    # Convert predictions and actuals to numpy arrays
    rf_predictions = np.array(rf_predictions)
    sentiment_rf_predictions = np.array(sentiment_rf_predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    rf_mae = mean_absolute_error(actuals, rf_predictions)
    rf_rmse = np.sqrt(mean_squared_error(actuals, rf_predictions))
    rf_r2 = r2_score(actuals, rf_predictions)
    
    sentiment_rf_mae = mean_absolute_error(actuals, sentiment_rf_predictions)
    sentiment_rf_rmse = np.sqrt(mean_squared_error(actuals, sentiment_rf_predictions))
    sentiment_rf_r2 = r2_score(actuals, sentiment_rf_predictions)
    
    # Calculate directional accuracy
    rf_actual_direction = np.sign(np.diff(actuals))
    rf_pred_direction = np.sign(np.array(rf_predictions[1:]) - np.array(rf_predictions[:-1]))
    rf_directional_accuracy = np.mean(rf_actual_direction == rf_pred_direction) * 100
    
    sentiment_rf_actual_direction = np.sign(np.diff(actuals))
    sentiment_rf_pred_direction = np.sign(np.array(sentiment_rf_predictions[1:]) - np.array(sentiment_rf_predictions[:-1]))
    sentiment_rf_directional_accuracy = np.mean(sentiment_rf_actual_direction == sentiment_rf_pred_direction) * 100
    
    # Print results
    print("\n=== RESULTS ===")
    print("\nRegular RF Model:")
    print(f"MAE: {rf_mae:.4f}")
    print(f"RMSE: {rf_rmse:.4f}")
    print(f"R²: {rf_r2:.4f}")
    print(f"Directional Accuracy: {rf_directional_accuracy:.2f}%")
    
    print("\nSentiment-Enhanced RF Model:")
    print(f"MAE: {sentiment_rf_mae:.4f}")
    print(f"RMSE: {sentiment_rf_rmse:.4f}")
    print(f"R²: {sentiment_rf_r2:.4f}")
    print(f"Directional Accuracy: {sentiment_rf_directional_accuracy:.2f}%")
    
    # Plot results
    plt.figure(figsize=(14, 7))
    plt.plot(actuals, label='Actual', color='black')
    plt.plot(rf_predictions, label='Regular RF', color='blue')
    plt.plot(sentiment_rf_predictions, label='Sentiment-Enhanced RF', color='red')
    plt.title('Comparison of RF Models')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()
    
    # Return results
    return {
        'regular_rf': {
            'mae': rf_mae,
            'rmse': rf_rmse,
            'r2': rf_r2,
            'directional_accuracy': rf_directional_accuracy
        },
        'sentiment_rf': {
            'mae': sentiment_rf_mae,
            'rmse': sentiment_rf_rmse,
            'r2': sentiment_rf_r2,
            'directional_accuracy': sentiment_rf_directional_accuracy
        }
    }

if __name__ == "__main__":
    # Replace with your data path
    data_path = "cache_GOOGL_10y.csv"
    
    # Set up Django settings if running in Django environment
    try:
        import django
        import os
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ensemble_web.settings')
        django.setup()
    except ImportError:
        pass
    
    try:
        results = compare_models(data_path)
        
        # Print improvement percentages
        mae_improvement = ((results['regular_rf']['mae'] - results['sentiment_rf']['mae']) / 
                          results['regular_rf']['mae']) * 100
        rmse_improvement = ((results['regular_rf']['rmse'] - results['sentiment_rf']['rmse']) / 
                           results['regular_rf']['rmse']) * 100
        r2_improvement = ((results['sentiment_rf']['r2'] - results['regular_rf']['r2']) / 
                         abs(results['regular_rf']['r2'])) * 100
        dir_acc_improvement = (results['sentiment_rf']['directional_accuracy'] - 
                              results['regular_rf']['directional_accuracy'])
        
        print("\n=== IMPROVEMENTS ===")
        print(f"MAE Improvement: {mae_improvement:.2f}%")
        print(f"RMSE Improvement: {rmse_improvement:.2f}%")
        print(f"R² Improvement: {r2_improvement:.2f}%")
        print(f"Directional Accuracy Improvement: {dir_acc_improvement:.2f} percentage points")
        
    except Exception as e:
        print(f"Error: {str(e)}")
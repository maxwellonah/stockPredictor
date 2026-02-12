import pandas as pd
from rf_model import EnhancedRandomForestModel
from lstm_model import LSTMModel
import os

def test_rf_training():
    print("Testing RF Model Training...")
    try:
        # Load sample data
        df = pd.read_csv('media/stock_data/GOOGL_data.csv')
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Create and train model
        rf_model = EnhancedRandomForestModel(feature_selection_threshold=0.02, random_state=42)
        metrics = rf_model.train(df, cv=2)  # Use fewer folds for testing
        
        print("RF Training successful!")
        print("Metrics:", metrics)
        return True
    except Exception as e:
        print(f"RF Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_lstm_training():
    print("\nTesting LSTM Model Training...")
    try:
        # Load sample data
        df = pd.read_csv('media/stock_data/GOOGL_data.csv')
        
        # Convert Date to datetime and sort
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Create and train model
        features = ['Close', 'Volume']
        lstm_model = LSTMModel(time_steps=10, features=features, epochs=1, batch_size=16)  # Reduced for testing
        history = lstm_model.train(df)
        
        print("LSTM Training successful!")
        print("Training history:", history)
        return True
    except Exception as e:
        print(f"LSTM Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Run tests
    rf_success = test_rf_training()
    lstm_success = test_lstm_training()
    
    print("\nTest Summary:")
    print(f"RF Model: {'PASSED' if rf_success else 'FAILED'}")
    print(f"LSTM Model: {'PASSED' if lstm_success else 'FAILED'}")

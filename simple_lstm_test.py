import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import os

def load_and_prepare_data(filepath):
    # Load the data
    df = pd.read_csv(filepath)
    
    # Convert Date to datetime and sort
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Use only Close price for simplicity
    data = df['Close'].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data, scaler

def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps - 30):  # Predict 30 days ahead
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps + 29, 0])  # 30 days ahead
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    try:
        # Load and prepare data
        print("Loading and preparing data...")
        data, scaler = load_and_prepare_data('media/stock_data/GOOGL_data.csv')
        
        # Create sequences
        time_steps = 60
        X, y = create_sequences(data, time_steps)
        
        # Reshape input to be [samples, time_steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        print(f"Data shape - X: {X.shape}, y: {y.shape}")
        
        # Build model
        print("Building model...")
        model = build_lstm_model((time_steps, 1))
        
        # Train model
        print("Training model...")
        history = model.fit(
            X, y, 
            epochs=5,  # Reduced for testing
            batch_size=32,
            validation_split=0.1,
            verbose=1
        )
        
        # Save the model
        model_path = os.path.join('models', 'simple_lstm_model.h5')
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set memory growth to avoid GPU memory issues
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            pass
    
    main()

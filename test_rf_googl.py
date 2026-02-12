import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

def load_data():
    """Load and prepare the GOOGL stock data"""
    try:
        # Load the data
        data_path = os.path.join('media', 'stock_data', 'GOOGL_data.csv')
        print(f"Loading data from: {os.path.abspath(data_path)}")
        
        df = pd.read_csv(data_path)
        print(f"Successfully loaded {len(df)} rows")
        
        # Convert Date to datetime and sort
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
        
        print(f"Final dataset size: {len(df)} rows")
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def train_model(df):
    """Train and evaluate a simple Random Forest model"""
    try:
        # Select features and target
        features = ['Close', 'Volume', 'MA7', 'MA21', 'MA50', 'Return', 'Volume_Change']
        X = df[features]
        y = df['Target']
        
        # Split data (time-series split would be better, but this is simpler for testing)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        print(f"\nTraining data: {len(X_train)} samples")
        print(f"Testing data: {len(X_test)} samples")
        
        # Train model
        print("\nTraining Random Forest model...")
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        print("\n=== Model Performance ===")
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Testing RMSE: {test_rmse:.4f}")
        
        # Feature importance
        importance = pd.Series(
            model.feature_importances_,
            index=features
        ).sort_values(ascending=False)
        
        print("\n=== Feature Importance ===")
        print(importance)
        
        return model
        
    except Exception as e:
        print(f"Error in model training: {str(e)}")
        raise

if __name__ == "__main__":
    print("=== GOOGL Stock Price Prediction Test ===\n")
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    try:
        # Step 1: Load and prepare data
        df = load_data()
        
        # Step 2: Train and evaluate model
        model = train_model(df)
        
        # Step 3: Save the model
        model_path = os.path.join('models', 'test_rf_model.joblib')
        import joblib
        joblib.dump(model, model_path)
        print(f"\n✅ Model saved to {os.path.abspath(model_path)}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

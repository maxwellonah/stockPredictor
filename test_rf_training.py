import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def test_rf_training():
    print("Testing Random Forest training...")
    try:
        # Generate synthetic data for testing
        print("\nGenerating synthetic data...")
        np.random.seed(42)
        n_samples = 1000
        X = np.random.rand(n_samples, 5)  # 5 features
        y = 3 * X[:, 0] + 2 * X[:, 1] - 5 * X[:, 2] + np.random.normal(0, 0.1, n_samples)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        print("Training Random Forest model...")
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"\nTest MSE: {mse:.4f}")
        print("Feature importances:", model.feature_importances_)
        print("\n✅ Random Forest test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error in RF test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rf_training()

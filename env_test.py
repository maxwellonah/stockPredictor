import sys
import os
import platform
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor

def print_section(title):
    print("\n" + "="*50)
    print(f"{title}".center(50))
    print("="*50)

def test_environment():
    # Python and OS info
    print_section("SYSTEM INFORMATION")
    print(f"Python: {sys.version}")
    print(f"OS: {platform.platform()}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Check important directories
    print_section("DIRECTORY CHECK")
    dirs = ['models', 'media/models', 'media/stock_data']
    for d in dirs:
        exists = os.path.exists(d)
        print(f"{d}: {'✅ Exists' if exists else '❌ Missing'}")
        if exists:
            print(f"   Contents: {os.listdir(d) if os.path.isdir(d) else 'Not a directory'}")
    
    # Check data file
    data_file = 'media/stock_data/GOOGL_data.csv'
    print_section("DATA FILE CHECK")
    try:
        with open(data_file, 'r') as f:
            first_line = f.readline().strip()
            print(f"First line of {data_file}: {first_line}")
        df = pd.read_csv(data_file)
        print(f"Successfully read {len(df)} rows")
        print("\nFirst 5 rows:")
        print(df.head().to_string())
    except Exception as e:
        print(f"❌ Error reading data file: {str(e)}")
    
    # Test basic ML operations
    print_section("MACHINE LEARNING TEST")
    try:
        X = np.random.rand(10, 3)
        y = X.sum(axis=1) + np.random.normal(0, 0.1, 10)
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)
        pred = model.predict([[0.5, 0.5, 0.5]])
        print(f"✅ Basic ML test passed. Prediction: {pred[0]:.4f}")
    except Exception as e:
        print(f"❌ ML test failed: {str(e)}")
    
    # Test TensorFlow
    print_section("TENSORFLOW TEST")
    try:
        print(f"TensorFlow version: {tf.__version__}")
        print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
        print("✅ TensorFlow test passed")
    except Exception as e:
        print(f"❌ TensorFlow test failed: {str(e)}")
    
    print_section("ENVIRONMENT TEST COMPLETE")

if __name__ == "__main__":
    test_environment()

import pandas as pd
import numpy as np

def check_data():
    try:
        # Load the data
        print("Loading data from media/stock_data/GOOGL_data.csv...")
        df = pd.read_csv('media/stock_data/GOOGL_data.csv')
        
        # Basic info
        print("\n=== Data Info ===")
        print(f"Number of rows: {len(df)}")
        print(f"Columns: {', '.join(df.columns)}")
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Check for missing values
        print("\n=== Missing Values ===")
        print(df.isnull().sum())
        
        # Check data types
        print("\n=== Data Types ===")
        print(df.dtypes)
        
        # Basic statistics
        print("\n=== Basic Statistics ===")
        print(df.describe())
        
        # Check date range
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            print("\n=== Date Range ===")
            print(f"Start date: {df['Date'].min()}")
            print(f"End date: {df['Date'].max()}")
            print(f"Number of trading days: {len(df)}")
        
        # Check for required columns
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"\n❌ Missing required columns: {', '.join(missing_cols)}")
        else:
            print("\n✅ All required columns present")
        
        # Check for zero or negative values in price data
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if col in df.columns:
                if (df[col] <= 0).any():
                    print(f"\n⚠️  Warning: Found {len(df[df[col] <= 0])} rows with non-positive {col}")
        
        # Check volume data
        if 'Volume' in df.columns:
            if (df['Volume'] < 0).any():
                print(f"\n⚠️  Warning: Found {len(df[df['Volume'] < 0])} rows with negative Volume")
        
        print("\n=== Data Check Complete ===")
        
    except Exception as e:
        print(f"\n❌ Error during data check: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_data()

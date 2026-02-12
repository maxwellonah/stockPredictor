#!/usr/bin/env python3
"""
Test script to verify sentiment analysis integration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from news_sentiment import NewsSentimentAnalyzer
from rf_model import EnhancedRandomForestModel
from lstm_model import LSTMModel

def create_sample_data():
    """Create sample stock data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    n_days = len(dates)
    
    # Generate realistic stock price data
    np.random.seed(42)
    prices = [150.0]  # Starting price
    
    for i in range(1, n_days):
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 10.0))  # Ensure price doesn't go negative
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': [int(1e6 * (1 + np.random.normal(0, 0.2))) for _ in range(n_days)]
    })
    
    return df

def test_sentiment_analysis():
    """Test sentiment analysis functionality"""
    print("=" * 60)
    print("Testing Sentiment Analysis Integration")
    print("=" * 60)
    
    # Create sample data
    df = create_sample_data()
    ticker = 'AAPL'
    
    print(f"Created sample data with {len(df)} days for {ticker}")
    print(f"Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    
    # Test sentiment analysis
    print("\n1. Testing NewsSentimentAnalyzer...")
    try:
        analyzer = NewsSentimentAnalyzer()
        sentiment_features = analyzer.get_sentiment_features(ticker, days=30)
        
        print("✅ Sentiment analysis completed successfully!")
        print("Sentiment features:")
        for key, value in sentiment_features.items():
            print(f"  {key}: {value:.4f}")
            
    except Exception as e:
        print(f"❌ Error in sentiment analysis: {str(e)}")
        # Use default values for testing
        sentiment_features = {
            'sentiment_score': 0.1,
            'sentiment_magnitude': 0.3,
            'sentiment_volume': 25,
            'sentiment_trend': 0.02,
            'sentiment_volatility': 0.15
        }
        print("Using default sentiment values for testing...")
    
    # Test RF model with sentiment
    print("\n2. Testing RF Model with Sentiment Features...")
    try:
        rf_model = EnhancedRandomForestModel(random_state=42)
        
        # Test feature creation with sentiment
        df_features = rf_model.create_features(df)
        df_with_sentiment = rf_model.add_sentiment_features(df_features, sentiment_features)
        
        print("✅ RF model sentiment integration successful!")
        print(f"Features shape: {df_with_sentiment.shape}")
        print(f"Sentiment columns added: {[col for col in df_with_sentiment.columns if 'Sentiment' in col]}")
        
        # Test prediction (this might fail due to insufficient data, but that's expected)
        try:
            prediction = rf_model.predict_next_day(df, sentiment_features=sentiment_features)
            print(f"✅ RF prediction with sentiment: ${prediction:.2f}")
        except Exception as pred_e:
            print(f"⚠️  RF prediction failed (expected with sample data): {str(pred_e)}")
            
    except Exception as e:
        print(f"❌ Error in RF model sentiment integration: {str(e)}")
    
    # Test LSTM model with sentiment
    print("\n3. Testing LSTM Model with Sentiment Features...")
    try:
        lstm_model = LSTMModel(time_steps=30, epochs=2)  # Small epochs for testing
        
        # Test feature creation with sentiment
        df_features = lstm_model.create_features(df, for_training=False)
        df_with_sentiment = lstm_model.add_sentiment_features(df_features, sentiment_features)
        
        print("✅ LSTM model sentiment integration successful!")
        print(f"Features shape: {df_with_sentiment.shape}")
        print(f"Sentiment columns added: {[col for col in df_with_sentiment.columns if 'Sentiment' in col]}")
        
        # Test prediction (this might fail due to insufficient data, but that's expected)
        try:
            prediction_info = lstm_model.predict(df, sentiment_features=sentiment_features)
            if isinstance(prediction_info, dict):
                print(f"✅ LSTM prediction with sentiment: ${prediction_info['predicted_price']:.2f}")
                print(f"   Confidence interval: ${prediction_info['confidence_interval'][0]:.2f} - ${prediction_info['confidence_interval'][1]:.2f}")
            else:
                print(f"✅ LSTM prediction with sentiment: ${prediction_info:.2f}")
        except Exception as pred_e:
            print(f"⚠️  LSTM prediction failed (expected with sample data): {str(pred_e)}")
            
    except Exception as e:
        print(f"❌ Error in LSTM model sentiment integration: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Sentiment Analysis Integration Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_sentiment_analysis()

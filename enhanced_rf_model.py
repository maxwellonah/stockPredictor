"""
Enhanced Random Forest model with news sentiment analysis
"""

import pandas as pd
import numpy as np
from rf_model import EnhancedRandomForestModel
from news_sentiment import NewsSentimentAnalyzer
import os

# Try to import Django settings, but handle the case when they're not available
try:
    from django.conf import settings
except (ImportError, RuntimeError):
    # Create a mock settings object if Django settings aren't available
    class MockSettings:
        NEWS_API_KEY = os.environ.get('NEWS_API_KEY', 'your_default_api_key')
    settings = MockSettings()

class SentimentEnhancedRFModel(EnhancedRandomForestModel):
    """
    Random Forest model enhanced with news sentiment analysis
    """
    
    def __init__(self):
        """Initialize the model"""
        super().__init__()
        # Initialize sentiment analyzer with API key from environment
        api_key = os.environ.get('NEWS_API_KEY', 'your_default_api_key')
        self.sentiment_analyzer = NewsSentimentAnalyzer(api_key=api_key)
        self.include_sentiment = True
    
    def create_features(self, df):
        """
        Create features including sentiment analysis
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with stock data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with features
        """
        # Get base features from parent class
        df_features = super().create_features(df)
        
        # If sentiment is disabled, return base features
        if not self.include_sentiment:
            return df_features
        
        # Extract ticker symbol from data (assuming it's in the filename or metadata)
        # For this example, we'll use a default ticker
        ticker = "GOOGL"  # Default ticker
        
        # Try to extract ticker from DataFrame if available
        if hasattr(df, 'name') and df.name:
            parts = df.name.split('_')
            if len(parts) > 0:
                ticker = parts[0].upper()
        
        # Get sentiment features
        sentiment_features = self.sentiment_analyzer.get_sentiment_features(ticker)
        
        # Add sentiment features to the DataFrame
        for day in range(len(df_features)):
            for feature, value in sentiment_features.items():
                df_features.loc[df_features.index[day], feature] = value
        
        return df_features
    
    def train(self, df, include_sentiment=True):
        """
        Train the model
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with stock data
        include_sentiment : bool
            Whether to include sentiment features
            
        Returns:
        --------
        self
        """
        self.include_sentiment = include_sentiment
        return super().train(df)
    
    def predict_next_day(self, df, include_sentiment=True):
        """
        Predict the next day's price
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with stock data
        include_sentiment : bool
            Whether to include sentiment features
            
        Returns:
        --------
        float
            Predicted price
        """
        self.include_sentiment = include_sentiment
        
        # Create features
        df_features = self.create_features(df)
        
        # Get the last row for prediction
        last_row = df_features.iloc[-1:].copy()
        
        # If sentiment is disabled, remove sentiment columns
        if not include_sentiment:
            sentiment_cols = ['sentiment_score', 'sentiment_magnitude', 'sentiment_volume',
                             'sentiment_trend', 'sentiment_volatility']
            last_row = last_row.drop(sentiment_cols, axis=1, errors='ignore')
        
        return super().predict_next_day(last_row)
    
    def feature_importance(self):
        """
        Get feature importance
        
        Returns:
        --------
        dict
            Dictionary with feature importance
        """
        importance_dict = super().feature_importance()
        
        # Filter out sentiment features if they're not being used
        if not self.include_sentiment:
            importance_dict = {k: v for k, v in importance_dict.items() 
                              if k not in ['sentiment_score', 'sentiment_magnitude', 
                                          'sentiment_volume', 'sentiment_trend', 
                                          'sentiment_volatility']}
        
        return importance_dict
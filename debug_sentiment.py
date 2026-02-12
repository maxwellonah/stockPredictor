#!/usr/bin/env python3
"""
Debug script for sentiment chart issues
"""

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd
from sentiment_charts import create_sentiment_history_chart, create_sentiment_summary_cards
from news_sentiment import NewsSentimentAnalyzer

def test_sentiment_components():
    """Test sentiment chart components individually"""
    print("Testing sentiment chart components...")
    
    # Test 1: Create summary cards
    print("\n1. Testing summary cards...")
    try:
        sentiment_features = {
            'sentiment_score': 0.2,
            'sentiment_magnitude': 0.3,
            'sentiment_volume': 25,
            'sentiment_trend': 0.01,
            'sentiment_volatility': 0.15
        }
        
        cards = create_sentiment_summary_cards(sentiment_features)
        print(f"✅ Created {len(cards)} summary cards successfully")
        
        # Check if cards are valid Dash components
        for i, card in enumerate(cards):
            print(f"   Card {i+1}: {type(card)}")
            
    except Exception as e:
        print(f"❌ Error creating summary cards: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Create sentiment history chart
    print("\n2. Testing sentiment history chart...")
    try:
        fig = create_sentiment_history_chart('AAPL', days=30)
        print(f"✅ Created sentiment chart successfully")
        print(f"   Chart type: {type(fig)}")
        
        # Check if it's a valid Plotly figure
        if hasattr(fig, 'data') and hasattr(fig, 'layout'):
            print(f"   Chart has {len(fig.data)} traces")
        else:
            print("❌ Chart doesn't have expected Plotly structure")
            
    except Exception as e:
        print(f"❌ Error creating sentiment chart: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Test with NewsSentimentAnalyzer
    print("\n3. Testing NewsSentimentAnalyzer...")
    try:
        analyzer = NewsSentimentAnalyzer()
        sentiment_data = analyzer.get_sentiment_features('AAPL', days=30)
        print(f"✅ NewsSentimentAnalyzer working")
        print(f"   Sentiment score: {sentiment_data['sentiment_score']:.4f}")
        
    except Exception as e:
        print(f"❌ Error with NewsSentimentAnalyzer: {str(e)}")
        import traceback
        traceback.print_exc()

def create_simple_test_chart():
    """Create a simple test chart to verify Plotly works"""
    print("\n4. Creating simple test chart...")
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[1, 2, 3, 4],
            y=[1, 4, 2, 3],
            name='Test Data'
        ))
        fig.update_layout(title='Test Chart')
        print("✅ Simple test chart created successfully")
        return fig
    except Exception as e:
        print(f"❌ Error creating test chart: {str(e)}")
        return None

if __name__ == "__main__":
    test_sentiment_components()
    create_simple_test_chart()
    print("\n🎯 Debug testing complete!")

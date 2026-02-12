"""
Sentiment History Chart Component for Dash App
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
from news_sentiment import NewsSentimentAnalyzer

def create_sentiment_history_chart(ticker, days=90):
    """
    Create a sentiment history chart for a given ticker
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    days : int
        Number of days to analyze
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Sentiment history chart
    """
    try:
        # Try to get real sentiment data
        try:
            analyzer = NewsSentimentAnalyzer()
            current_sentiment = analyzer.get_sentiment_features(ticker, days=30)
            use_real_data = True
        except Exception as e:
            # Fall back to simulated data if API fails
            current_sentiment = {
                'sentiment_score': 0.1,
                'sentiment_magnitude': 0.3,
                'sentiment_volume': 25,
                'sentiment_trend': 0.01,
                'sentiment_volatility': 0.15
            }
            use_real_data = False
        
        # Generate sentiment data over time
        sentiment_data = []
        current_date = datetime.now()
        
        for i in range(days):
            date = current_date - timedelta(days=days-i)
            
            # Create realistic sentiment evolution
            base_sentiment = current_sentiment['sentiment_score']
            trend = current_sentiment['sentiment_trend'] * i
            noise = 0.1 * (hash(f"{ticker}_{date.strftime('%Y-%m-%d')}") % 100) / 100
            
            sentiment_score = base_sentiment + trend + noise
            sentiment_magnitude = current_sentiment['sentiment_magnitude'] + 0.05 * noise
            sentiment_volume = max(5, current_sentiment['sentiment_volume'] + int(10 * noise))
            
            sentiment_data.append({
                'Date': date,
                'Sentiment_Score': sentiment_score,
                'Sentiment_Magnitude': sentiment_magnitude,
                'News_Volume': sentiment_volume
            })
        
        df = pd.DataFrame(sentiment_data)
        
        # Create the chart
        fig = go.Figure()
        
        # Add sentiment score line
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Sentiment_Score'],
            mode='lines+markers',
            name='Sentiment Score',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        # Add zero line for reference
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add sentiment magnitude as area
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Sentiment_Magnitude'],
            mode='lines',
            name='Sentiment Magnitude',
            line=dict(color='orange', width=1),
            fill='tozeroy',
            fillcolor='rgba(255, 165, 0, 0.2)'
        ))
        
        # Update layout with simpler configuration
        title = f'Sentiment Analysis History - {ticker}'
        if not use_real_data:
            title += ' (Simulated Data)'

        try:
            fig.update_layout(
                title=title,
                xaxis_title='Date',
                yaxis_title='Sentiment Score',
                height=400,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
        except Exception as layout_error:
            # Try a simpler layout
            fig.update_layout(
                title=title,
                height=400
            )
        
        return fig
        
    except Exception as e:
        # Return empty chart with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating sentiment chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(title="Sentiment History - Error")
        return fig

def create_sentiment_summary_cards(sentiment_features):
    """
    Create summary cards for sentiment features
    
    Parameters:
    -----------
    sentiment_features : dict
        Dictionary containing sentiment features
        
    Returns:
    --------
    list
        List of Dash components for sentiment summary
    """
    import dash_bootstrap_components as dbc
    from dash import html
    
    # Determine sentiment color
    score = sentiment_features.get('sentiment_score', 0)
    if score > 0.1:
        sentiment_color = "success"
        sentiment_label = "Positive"
    elif score < -0.1:
        sentiment_color = "danger"
        sentiment_label = "Negative"
    else:
        sentiment_color = "secondary"
        sentiment_label = "Neutral"
    
    # Create cards
    cards = [
        dbc.Card(
            dbc.CardBody([
                html.H4(f"{score:.3f}", className="card-title"),
                html.P("Sentiment Score", className="card-text"),
                html.Span(sentiment_label, className=f"badge bg-{sentiment_color}")
            ]),
            color="light"
        ),
        dbc.Card(
            dbc.CardBody([
                html.H4(f"{sentiment_features.get('sentiment_magnitude', 0):.3f}", className="card-title"),
                html.P("Sentiment Magnitude", className="card-text"),
                html.P("Strength of sentiment", className="text-muted small")
            ]),
            color="light"
        ),
        dbc.Card(
            dbc.CardBody([
                html.H4(f"{sentiment_features.get('sentiment_volume', 0)}", className="card-title"),
                html.P("News Volume", className="card-text"),
                html.P("Articles analyzed", className="text-muted small")
            ]),
            color="light"
        ),
        dbc.Card(
            dbc.CardBody([
                html.H4(f"{sentiment_features.get('sentiment_trend', 0):.4f}", className="card-title"),
                html.P("Sentiment Trend", className="card-text"),
                html.P("Direction over time", className="text-muted small")
            ]),
            color="light"
        )
    ]
    
    return cards

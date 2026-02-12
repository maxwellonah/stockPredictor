"""
News sentiment analysis module for stock prediction
"""

import os
import requests
from datetime import datetime, timedelta
import pandas as pd
from textblob import TextBlob

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Try to import Django settings, but handle the case when they're not available
try:
    from django.conf import settings
except (ImportError, RuntimeError):
    # Create a mock settings object if Django settings aren't available
    class MockSettings:
        NEWS_API_KEY = os.environ.get('NEWS_API_KEY', 'your_default_api_key')
    settings = MockSettings()

class NewsSentimentAnalyzer:
    """
    Class to fetch and analyze news sentiment for stock predictions
    """
    
    def __init__(self, api_key=None):
        """Initialize with API key"""
        if api_key is not None:
            self.api_key = api_key
            return

        # Prefer Django settings only when they are configured; otherwise fall back to env.
        try:
            if hasattr(settings, "configured") and not settings.configured:
                raise RuntimeError("Django settings not configured")
            self.api_key = settings.NEWS_API_KEY
            if not self.api_key:
                raise AttributeError("NEWS_API_KEY not set")
        except Exception:
            # Fallback to environment variable or default key
            self.api_key = os.environ.get('NEWS_API_KEY', 'your_default_api_key')

    def fetch_news(self, ticker, days=90):
        """
        Fetch news articles for a given ticker symbol
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol (e.g., 'AAPL')
        days : int
            Number of days to look back for news
            
        Returns:
        --------
        list
            List of news articles with title, description, and published date
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Format dates for API
            from_date = start_date.strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d')
            
            # Construct API URL (using News API)
            url = f"https://newsapi.org/v2/everything"
            params = {
                'q': f"{ticker} stock",
                'from': from_date,
                'to': to_date,
                'language': 'en',
                'sortBy': 'publishedAt',
                'apiKey': self.api_key
            }
            
            print(f"Fetching news for {ticker} from {from_date} to {to_date}")
            print(f"API URL: {url}")
            print(f"Params: {params}")
            
            # Make API request with timeout
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                print(f"API Error: Status code {response.status_code}")
                print(f"Response: {response.text}")
                return []
            
            data = response.json()
            
            # Extract relevant information
            articles = data.get('articles', [])
            news_data = []
            
            for article in articles:
                news_data.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'published_at': article.get('publishedAt', ''),
                    'source': article.get('source', {}).get('name', '')
                })
            
            print(f"Found {len(news_data)} articles for {ticker}")
            return news_data
            
        except requests.exceptions.Timeout:
            print("Error: API request timed out")
            return []
            
        except requests.exceptions.ConnectionError as e:
            print(f"Connection Error: {str(e)}")
            print("This might be due to DNS resolution issues or network connectivity problems")
            print("Please check your internet connection and try again")
            return []
            
        except Exception as e:
            print(f"Unexpected Error: {str(e)}")
            return []
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text using TextBlob
        
        Parameters:
        -----------
        text : str
            Text to analyze
            
        Returns:
        --------
        dict
            Dictionary containing sentiment polarity and subjectivity
        """
        if not text:
            return {'polarity': 0, 'subjectivity': 0}
        
        analysis = TextBlob(text)
        return {
            'polarity': analysis.sentiment.polarity,
            'subjectivity': analysis.sentiment.subjectivity
        }
    
    def get_sentiment_features(self, ticker, days=30):
        """
        Get sentiment features for a ticker
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        days : int
            Number of days to look back
            
        Returns:
        --------
        dict
            Dictionary with sentiment features
        """
        # Fetch news
        news_data = self.fetch_news(ticker, days)
        
        if not news_data:
            return {
                'sentiment_score': 0,
                'sentiment_magnitude': 0,
                'sentiment_volume': 0,
                'sentiment_trend': 0,
                'sentiment_volatility': 0
            }

        # Analyze sentiment for each article
        sentiments = []
        for article in news_data:
            # Combine title and description for better sentiment analysis
            text = f"{article['title']} {article['description']} {article['content']}"
            sentiment = self.analyze_sentiment(text)
            published_at = article.get('published_at', '')
            
            try:
                date = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
            except:
                date = datetime.now()
            
            sentiments.append({
                'date': date,
                'polarity': sentiment['polarity'],
                'subjectivity': sentiment['subjectivity']
            })
        
        # Convert to DataFrame for easier analysis
        if not sentiments:
            return {
                'sentiment_score': 0,
                'sentiment_magnitude': 0,
                'sentiment_volume': 0,
                'sentiment_trend': 0,
                'sentiment_volatility': 0
            }

        df = pd.DataFrame(sentiments)

        # Calculate sentiment features
        sentiment_score = df['polarity'].mean()
        sentiment_magnitude = abs(df['polarity']).mean()
        sentiment_volume = len(df)

        # Calculate sentiment trend (slope of sentiment over time)
        if len(df) > 1:
            df = df.sort_values('date')
            df['day'] = (df['date'] - df['date'].min()).dt.total_seconds() / (24 * 3600)

            if len(df['day'].unique()) > 1:
                from scipy import stats
                slope, _, _, _, _ = stats.linregress(df['day'], df['polarity'])
                sentiment_trend = slope
            else:
                sentiment_trend = 0
        else:
            sentiment_trend = 0

        # Calculate sentiment volatility
        sentiment_volatility = df['polarity'].std() if len(df) > 1 else 0

        return {
            'sentiment_score': sentiment_score,
            'sentiment_magnitude': sentiment_magnitude,
            'sentiment_volume': sentiment_volume,
            'sentiment_trend': sentiment_trend,
            'sentiment_volatility': sentiment_volatility
        }

    def get_sentiment_timeseries(self, ticker, days=3, freq='h'):
        """Get aggregated sentiment time series (default hourly bins)."""
        news_data = self.fetch_news(ticker, days)

        if not news_data:
            return pd.DataFrame(
                columns=[
                    'Date',
                    'Sentiment_Score',
                    'Sentiment_Magnitude',
                    'Sentiment_Volume',
                    'Sentiment_Trend',
                    'Sentiment_Volatility',
                ]
            )

        rows = []
        for article in news_data:
            text = f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}"
            s = self.analyze_sentiment(text)
            published_at = article.get('published_at', '')

            try:
                dt = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                try:
                    dt = pd.to_datetime(published_at, utc=True).to_pydatetime()
                except Exception:
                    continue

            rows.append({'Date': dt, 'Polarity': float(s.get('polarity', 0.0))})

        if not rows:
            return pd.DataFrame(
                columns=[
                    'Date',
                    'Sentiment_Score',
                    'Sentiment_Magnitude',
                    'Sentiment_Volume',
                    'Sentiment_Trend',
                    'Sentiment_Volatility',
                ]
            )

        df = pd.DataFrame(rows)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()

        grouped = df['Polarity'].groupby(pd.Grouper(freq=freq))
        out = pd.DataFrame({
            'Sentiment_Score': grouped.mean(),
            'Sentiment_Magnitude': grouped.apply(lambda x: x.abs().mean() if len(x) else 0.0),
            'Sentiment_Volume': grouped.count(),
            'Sentiment_Volatility': grouped.std().fillna(0.0),
        })
        out['Sentiment_Trend'] = out['Sentiment_Score'].diff().fillna(0.0)

        out = out.reset_index()
        out['Date'] = pd.to_datetime(out['Date'])
        return out
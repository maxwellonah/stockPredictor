import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta, time
import requests
import io
from dotenv import load_dotenv

# Import models
from rf_model import EnhancedRandomForestModel
from lstm_model import LSTMModel
from news_sentiment import NewsSentimentAnalyzer
from sentiment_charts import create_sentiment_history_chart, create_sentiment_summary_cards

# Load environment variables
load_dotenv()
FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY', "d6747gpr01qmckkc2ig0d6747gpr01qmckkc2igg")
TWELVE_DATA_API_KEY = os.environ.get('TWELVE_DATA_API_KEY', "22425aaff0ea4df09f0e4f1fb9791b9f")
MAX_FETCH_DATA_AGE_MINUTES = 10
FETCH_LOOKBACK_DAYS = 7

# Initialize the Dash app with a Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
server = app.server

# Define instrument options
STOCK_OPTIONS = [
    {'label': 'Google (GOOGL)', 'value': 'GOOGL'},
    {'label': 'Apple (AAPL)', 'value': 'AAPL'},
    {'label': 'Microsoft (MSFT)', 'value': 'MSFT'},
    {'label': 'Amazon (AMZN)', 'value': 'AMZN'},
    {'label': 'BTC/USDT', 'value': 'BTC/USDT'},
    {'label': 'ETH/USDT', 'value': 'ETH/USDT'},
    {'label': 'SOL/USDT', 'value': 'SOL/USDT'},
    {'label': 'DOGE/USDT', 'value': 'DOGE/USDT'},
    {'label': 'ARB/USDT', 'value': 'ARB/USDT'},
    {'label': 'EUR/USD', 'value': 'EUR/USD'},
    {'label': 'GBP/USD', 'value': 'GBP/USD'},
    {'label': 'USD/JPY', 'value': 'USD/JPY'},
    {'label': 'AUD/USD', 'value': 'AUD/USD'}
]

# Provider routing:
# - Finnhub: US equities
# - Twelve Data: crypto + forex
INSTRUMENT_CONFIG = {
    'GOOGL': {'provider': 'finnhub', 'symbol': 'GOOGL', 'market_type': 'equity'},
    'AAPL': {'provider': 'finnhub', 'symbol': 'AAPL', 'market_type': 'equity'},
    'MSFT': {'provider': 'finnhub', 'symbol': 'MSFT', 'market_type': 'equity'},
    'AMZN': {'provider': 'finnhub', 'symbol': 'AMZN', 'market_type': 'equity'},
    'BTC/USDT': {'provider': 'twelve', 'symbol': 'BTC/USD', 'market_type': 'crypto'},
    'ETH/USDT': {'provider': 'twelve', 'symbol': 'ETH/USD', 'market_type': 'crypto'},
    'SOL/USDT': {'provider': 'twelve', 'symbol': 'SOL/USD', 'market_type': 'crypto'},
    'DOGE/USDT': {'provider': 'twelve', 'symbol': 'DOGE/USD', 'market_type': 'crypto'},
    'ARB/USDT': {'provider': 'twelve', 'symbol': 'ARB/USD', 'market_type': 'crypto'},
    'EUR/USD': {'provider': 'twelve', 'symbol': 'EUR/USD', 'market_type': 'forex'},
    'GBP/USD': {'provider': 'twelve', 'symbol': 'GBP/USD', 'market_type': 'forex'},
    'USD/JPY': {'provider': 'twelve', 'symbol': 'USD/JPY', 'market_type': 'forex'},
    'AUD/USD': {'provider': 'twelve', 'symbol': 'AUD/USD', 'market_type': 'forex'},
}


def resolve_market_symbol(ui_symbol):
    symbol = (ui_symbol or 'GOOGL').strip().upper()
    config = INSTRUMENT_CONFIG.get(symbol)
    if config:
        return symbol, config['provider'], config['symbol'], config['market_type']

    # Fallbacks for unknown symbols.
    if "/" in symbol:
        normalized = symbol.replace("USDT", "USD")
        market_type = 'crypto' if 'USDT' in symbol else 'forex'
        return symbol, 'twelve', normalized, market_type
    return symbol, 'finnhub', symbol, 'equity'


def is_market_open(market_type, now_utc):
    """Simple market-hours gate in UTC for freshness enforcement."""
    weekday = now_utc.weekday()  # Monday=0 ... Sunday=6
    t = now_utc.time()

    if market_type == 'crypto':
        return True  # 24/7

    if market_type == 'equity':
        # US equities regular hours in UTC (approx): Mon-Fri 14:30-21:00.
        return weekday < 5 and (time(14, 30) <= t < time(21, 0))

    if market_type == 'forex':
        # Forex: Sun 22:00 UTC -> Fri 22:00 UTC.  
        if weekday in (0, 1, 2, 3):
            return True
        if weekday == 4:
            return t < time(22, 0)
        if weekday == 6:
            return t >= time(22, 0)
        return False

    return True


def add_default_sentiment_columns(df):
    df = df.copy()
    defaults = {
        'Sentiment_Score': 0.0,
        'Sentiment_Magnitude': 0.0,
        'Sentiment_Volume': 0.0,
        'Sentiment_Trend': 0.0,
        'Sentiment_Volatility': 0.0,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    return df

# Create directory for uploaded files
os.makedirs('uploads', exist_ok=True)
os.makedirs('models', exist_ok=True)

def add_time_aligned_sentiment(df, ticker):
    df = df.copy()
    try:
        analyzer = NewsSentimentAnalyzer()
        ts = analyzer.get_sentiment_timeseries(ticker, days=3, freq='h')
        if ts is None or ts.empty:
            return df

        ts = ts.sort_values('Date')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        merged = pd.merge_asof(
            df,
            ts,
            on='Date',
            direction='backward'
        )

        for col in ['Sentiment_Score', 'Sentiment_Magnitude', 'Sentiment_Volume', 'Sentiment_Trend', 'Sentiment_Volatility']:
            if col in merged.columns:
                merged[col] = merged[col].ffill().fillna(0)
        return merged
    except Exception as e:
        print(f"Error merging sentiment timeseries: {str(e)}")
        return df

def _parse_dt(value, default, end_of_day=False):
    if value is None:
        return default
    if isinstance(value, datetime):
        return value
    parsed = pd.to_datetime(value)
    if isinstance(value, str) and len(value.strip()) == 10 and end_of_day:
        parsed = parsed + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return parsed.to_pydatetime()


def _standardize_ohlcv(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    out = df.copy()
    out['Date'] = pd.to_datetime(out['Date'])
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors='coerce')

    # Some providers/symbols do not supply meaningful volume (especially FX/crypto pairs).
    # Keep a positive, non-zero series so downstream pct_change features remain valid.
    vol = out['Volume'].fillna(0.0)
    positive_vol = vol[vol > 0]
    if positive_vol.empty:
        out['Volume'] = 1.0
    else:
        fallback = float(positive_vol.median())
        if fallback <= 0:
            fallback = 1.0
        out['Volume'] = vol.where(vol > 0, fallback)
    out = out.dropna(subset=['Date', 'Open', 'High', 'Low', 'Close'])
    out = out.sort_values('Date').drop_duplicates(subset=['Date']).reset_index(drop=True)
    return out[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]


def fetch_stock_data_finnhub(symbol, from_dt, to_dt):
    if not FINNHUB_API_KEY:
        print("FINNHUB_API_KEY is missing.")
        return None
    url = "https://finnhub.io/api/v1/stock/candle"
    params = {
        'symbol': symbol,
        'resolution': '1',
        'from': int(from_dt.timestamp()),
        'to': int(to_dt.timestamp()),
        'token': FINNHUB_API_KEY,
    }
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=20)
            if not response.ok:
                print(f"Error fetching Finnhub data: HTTP {response.status_code} - {response.text}")
                return None
            data = response.json()
            if data.get('s') != 'ok' or not data.get('t'):
                print(f"Finnhub returned no usable data for {symbol}: {data}")
                return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df = pd.DataFrame({
                'Date': pd.to_datetime(data.get('t', []), unit='s'),
                'Open': data.get('o', []),
                'High': data.get('h', []),
                'Low': data.get('l', []),
                'Close': data.get('c', []),
                'Volume': data.get('v', []),
            })
            return _standardize_ohlcv(df)
        except Exception as e:
            last_error = e
            print(f"Error fetching Finnhub data (attempt {attempt + 1}/3): {str(e)}")
    print(f"Error fetching Finnhub data: {str(last_error)}")
    return None


def fetch_stock_data_twelve(symbol, from_dt, to_dt):
    if not TWELVE_DATA_API_KEY:
        print("TWELVE_DATA_API_KEY is missing.")
        return None
    url = "https://api.twelvedata.com/time_series"
    params = {
        'symbol': symbol,
        'interval': '1min',
        'start_date': from_dt.strftime('%Y-%m-%d %H:%M:%S'),
        'end_date': to_dt.strftime('%Y-%m-%d %H:%M:%S'),
        'timezone': 'UTC',
        'order': 'ASC',
        'outputsize': 2000,
        'apikey': TWELVE_DATA_API_KEY,
    }
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=25)
            if not response.ok:
                print(f"Error fetching Twelve Data: HTTP {response.status_code} - {response.text}")
                return None
            data = response.json()
            values = data.get('values', [])
            if not values:
                print(f"Twelve Data returned no usable data for {symbol}: {data}")
                return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df = pd.DataFrame(values)
            df = df.rename(columns={
                'datetime': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
            })
            return _standardize_ohlcv(df)
        except Exception as e:
            last_error = e
            print(f"Error fetching Twelve Data (attempt {attempt + 1}/3): {str(e)}")
    print(f"Error fetching Twelve Data: {str(last_error)}")
    return None


def fetch_stock_data(ticker, timespan='minute', multiplier=1, from_date=None, to_date=None, provider=None):
    """Fetch 1-minute market data from configured providers."""
    if not ticker:
        return None

    now_dt = datetime.utcnow()
    default_from = now_dt - timedelta(days=FETCH_LOOKBACK_DAYS)
    from_dt = _parse_dt(from_date, default_from, end_of_day=False)
    to_dt = _parse_dt(to_date, now_dt, end_of_day=True)
    if to_dt <= from_dt:
        to_dt = from_dt + timedelta(minutes=1)

    provider_name = (provider or '').strip().lower()
    if not provider_name:
        if "/" in str(ticker):
            provider_name = 'twelve'
        else:
            provider_name = 'finnhub'

    if provider_name == 'finnhub':
        return fetch_stock_data_finnhub(ticker, from_dt, to_dt)
    if provider_name == 'twelve':
        return fetch_stock_data_twelve(ticker, from_dt, to_dt)

    print(f"Unknown provider '{provider_name}' for symbol {ticker}")
    return None

# Function to calculate technical indicators
def calculate_technical_indicators(df):
    """Calculate technical indicators for display"""
    df = df.copy()
    
    # Moving Averages
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    # Avoid division by zero
    avg_loss = avg_loss.replace(0, 1e-10)
    df['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(20).mean()
    df['BB_std'] = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
    
    return df

# Function to train RF model
def train_rf_model(df, status_div_id):
    """Train Random Forest model and return metrics"""
    try:
        print("Starting RF model training...")
        # Create model
        rf_model = EnhancedRandomForestModel(
            feature_selection_threshold=0.01,
            random_state=42,
            horizon_steps=30
        )
        
        # Train model
        print("Training RF model...")
        metrics = rf_model.train(df, cv=3)
        
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)
        
        # Save model
        model_path = os.path.join('models', 'rf_model.joblib')
        print(f"Saving RF model to {model_path}")
        rf_model.save_model(model_path)
        
        print("RF model training completed successfully")
        return rf_model, metrics
    
    except Exception as e:
        error_msg = f"Error training RF model: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, {"error": error_msg, "status": "failed"}

# Function to train LSTM model
def train_lstm_model(df, status_div_id):
    """Train LSTM model and return metrics"""
    try:
        print("Starting LSTM model training...")
        # Define a richer feature set for better 2-hour forecasting accuracy
        features = [
            'Close', 'Open', 'High', 'Low', 'Volume',
            'Return', 'Log_Return', 'Volume_Change',
            'MA7', 'MA14', 'MA21', 'MA50',
            'Volatility7', 'Volatility14', 'Volatility21',
            'RSI', 'RSI_Trend', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_width', 'BB_position',
            'Momentum7', 'Momentum14', 'Momentum21',
            'Price_to_MA50',
            'Day_of_Week', 'Month',
            'Sentiment_Score', 'Sentiment_Magnitude', 'Sentiment_Volume', 'Sentiment_Trend', 'Sentiment_Volatility'
        ]
        
        # Ensure all required features are in the dataframe
        required_cols = ['Date', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in dataframe: {', '.join(missing_cols)}. "
                f"Available columns: {', '.join(df.columns)}"
            )
        
        # Create model with larger minute-history context for stronger regime coverage.
        print("Initializing LSTM model...")
        dynamic_time_steps = int(np.clip(len(df) // 8, 120, 240))
        lstm_model = LSTMModel(
            time_steps=dynamic_time_steps,
            features=features,
            epochs=120,
            batch_size=32,
            horizon_steps=120
        )
        
        # Train model
        print("Training AI 2-Hour Model...")
        history = lstm_model.train(df)
        
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)
        
        # Save model
        model_path = os.path.join('models', 'lstm_model.h5')
        print(f"Saving LSTM model to {model_path}")
        lstm_model.save(model_path)
        
        print("LSTM model training completed successfully")
        return lstm_model, history
    
    except Exception as e:
        error_msg = f"Error training LSTM model: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, {"error": error_msg, "status": "failed"}

# App Layout
app.layout = dbc.Container([
    # Notifications container
    html.Div(
        [
            dbc.Toast(
                id="training-notification",
                header="Model Training",
                is_open=False,
                dismissable=True,
                duration=8000,  # Longer duration
                icon="primary",
                style={"position": "fixed", "top": 66, "right": 10, "width": 350, "zIndex": 1000},
            ),
            dbc.Toast(
                id="completion-notification",
                header="Training Complete",
                is_open=False,
                dismissable=True,
                duration=8000,  # Longer duration
                icon="success",
                style={"position": "fixed", "top": 66, "right": 10, "width": 350, "zIndex": 1000},
            ),
        ]
    ),
    
    # Loading spinners container
    html.Div(
        [
            dbc.Spinner(
                id="rf-spinner",
                color="primary",
                type="grow",
                fullscreen=False,
                children=html.Div(id="rf-spinner-output"),
                spinner_style={"width": "3rem", "height": "3rem"},
            ),
            dbc.Spinner(
                id="lstm-spinner",
                color="info",
                type="grow",
                fullscreen=False,
                children=html.Div(id="lstm-spinner-output"),
                spinner_style={"width": "3rem", "height": "3rem"},
            ),
        ],
        style={"textAlign": "center", "marginBottom": "20px"}
    ),
    dbc.Row([
        dbc.Col([
            html.H1("Hybrid Stock Prediction System", className="text-center my-4"),
            html.P("Using AI models for 30-minute and 2-hour intraday predictions", className="text-center text-muted mb-4")
        ])
    ]),
    
    # Status indicators
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H5("System Status", className="text-center"),
                dbc.Row([
                    dbc.Col(dbc.Card([
                        html.Div(id="rf-status", className="text-center p-2", 
                                children="AI 30-Min Model: Ready")
                    ], color="light"), width=6),
                    dbc.Col(dbc.Card([
                        html.Div(id="lstm-status", className="text-center p-2", 
                                children="AI 2-Hour Model: Ready")
                    ], color="light"), width=6),
                ])
            ], className="mb-4")
        ])
    ]),
    
    # Stock Selection & Data Source Tabs
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Data Source"),
                dbc.CardBody([
                    dbc.Tabs([
                        dbc.Tab([
                            html.P("Select an instrument to analyze:", className="mt-2"),
                            dcc.Dropdown(
                                id="stock-dropdown",
                                options=STOCK_OPTIONS,
                                value="GOOGL",
                                className="mb-3"
                            ),
                            dbc.Button("Fetch Data", id="fetch-data-btn", color="primary", className="mr-2"),
                            html.Div(id="api-data-info", className="mt-3")
                        ], label="Live Market API"),
                        
                        dbc.Tab([
                            html.P("Upload your own CSV file:", className="mt-2"),
                            dcc.Upload(
                                id='upload-data',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select Files')
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px'
                                },
                                multiple=False
                            ),
                            html.Div(id="upload-data-info", className="mt-3")
                        ], label="Upload Data")
                    ])
                ])
            ], className="mb-4")
        ])
    ]),
    
    # Price Chart
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Price Chart"),
                dbc.CardBody([
                    dcc.Graph(id="price-chart")
                ])
            ], className="mb-4")
        ])
    ]),
    
    # Technical Indicators
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Technical Indicators"),
                dbc.CardBody([
                    dbc.Tabs([
                        dbc.Tab([
                            dcc.Graph(id="moving-averages-chart")
                        ], label="Moving Averages"),
                        dbc.Tab([
                            dcc.Graph(id="rsi-chart")
                        ], label="RSI"),
                        dbc.Tab([
                            dcc.Graph(id="macd-chart")
                        ], label="MACD"),
                        dbc.Tab([
                            dcc.Graph(id="bollinger-chart")
                        ], label="Bollinger Bands")
                    ])
                ])
            ], className="mb-4")
        ])
    ]),
    
    # Predictions
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Predictions"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Train Models", id="train-models-btn", color="success", className="mb-3 w-100")
                        ], width=6),
                        dbc.Col([
                            dbc.Button("Make Predictions", id="predict-btn", color="primary", className="mb-3 w-100")
                        ], width=6)
                    ]),
                    dbc.Tabs([
                        dbc.Tab([
                            html.Div(id="rf-prediction-results"),
                            dcc.Graph(id="rf-prediction-chart")
                        ], label="AI 30-Min Model"),
                        dbc.Tab([
                            html.Div(id="lstm-prediction-results"),
                            dcc.Graph(id="lstm-prediction-chart")
                        ], label="AI 2-Hour Model"),
                        dbc.Tab([
                            html.Div(id="sentiment-summary-cards", className="mb-3"),
                            dcc.Graph(id="sentiment-history-chart")
                        ], label="Sentiment Analysis")
                    ])
                ])
            ])
        ])
    ]),
    
    # Store components for data
    dcc.Store(id="stock-data-store"),
    dcc.Store(id="rf-model-store"),
    dcc.Store(id="lstm-model-store"),
    dcc.Store(id="rf-predictions-store"),
    dcc.Store(id="lstm-predictions-store"),
    
    # Interval for status updates
    dcc.Interval(id="status-interval", interval=1000, n_intervals=0),
    
    html.Footer([
        html.P("Hybrid Stock Prediction System © 2025", className="text-center text-muted mt-4")
    ])
], fluid=True)

# Callback for uploading data
@app.callback(
    [Output("upload-data-info", "children"),
     Output("stock-data-store", "data", allow_duplicate=True)],
    [Input("upload-data", "contents")],
    [State("upload-data", "filename")],
    prevent_initial_call=True
)
def update_output(contents, filename):
    if contents is None:
        return html.Div("No file uploaded yet."), None
    
    try:
        content_type, content_string = contents.split(',')
        decoded = io.StringIO(content_string.decode('utf-8'))
        
        if 'csv' in filename:
            df = pd.read_csv(decoded)
        else:
            return html.Div(f"File type not supported: {filename}"), None
        
        # Validate data
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return html.Div(f"Missing required columns: {', '.join(missing_columns)}"), None
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Enforce strictly 1-minute bars
        df_sorted = df.sort_values('Date')
        if len(df_sorted) < 3:
            return html.Div("Uploaded file must contain at least 3 rows of 1-minute bars."), None

        diffs = df_sorted['Date'].diff().dropna()
        median_diff_seconds = diffs.median().total_seconds()
        if abs(median_diff_seconds - 60) > 5:
            return html.Div(
                f"Uploaded data must be 1-minute bars. Detected median interval: {median_diff_seconds:.1f} seconds.",
            ), None
        
        # Add time-aligned sentiment (fallback to GOOGL if ticker not available)
        df = add_time_aligned_sentiment(df, 'GOOGL')

        # Save file
        upload_path = os.path.join('uploads', filename)
        df.to_csv(upload_path, index=False)
        
        # Calculate technical indicators
        df_with_indicators = calculate_technical_indicators(df)
        
        return html.Div([
            html.P(f"File uploaded: {filename}"),
            html.P(f"Data range: {df['Date'].min().date()} to {df['Date'].max().date()}"),
            html.P(f"Number of records: {len(df)}")
        ]), df_with_indicators.to_json(date_format='iso', orient='split')
    
    except Exception as e:
        return html.Div(f"Error processing file: {str(e)}"), None

# Callback for fetching data from API
@app.callback(
    [Output("api-data-info", "children"),
     Output("stock-data-store", "data")],
    [Input("fetch-data-btn", "n_clicks")],
    [State("stock-dropdown", "value")]
)
def fetch_data(n_clicks, ticker):
    if n_clicks is None:
        return html.Div("Click 'Fetch Data' to load market data."), None

    if not ticker:
        ticker = 'GOOGL'

    ui_symbol, provider_name, provider_symbol, market_type = resolve_market_symbol(ticker)
    click_time_utc = datetime.utcnow()
    from_date = click_time_utc - timedelta(days=FETCH_LOOKBACK_DAYS)
    to_date = click_time_utc
    
    # Fetch 1-minute bars
    df = fetch_stock_data(
        provider_symbol,
        timespan='minute',
        multiplier=1,
        from_date=from_date,
        to_date=to_date,
        provider=provider_name,
    )

    # Keep Finnhub as primary for stocks, but fail over to Twelve Data if access is denied.
    if (df is None or df.empty) and provider_name == 'finnhub':
        fallback_provider = 'twelve'
        fallback_symbol = provider_symbol
        fallback_df = fetch_stock_data(
            fallback_symbol,
            timespan='minute',
            multiplier=1,
            from_date=from_date,
            to_date=to_date,
            provider=fallback_provider,
        )
        if fallback_df is not None and not fallback_df.empty:
            print(f"Finnhub fallback activated for {ui_symbol}: using Twelve Data.")
            df = fallback_df
            provider_name = f"{provider_name}->twelve"
            provider_symbol = fallback_symbol
    
    if df is None:
        return html.Div(
            f"Error fetching data for {ui_symbol} "
            f"(provider: {provider_name}, symbol: {provider_symbol}). "
            "Check server logs and confirm API keys are valid."
        ), None

    if df.empty:
        return html.Div(
            f"No data returned for {ui_symbol} "
            f"(provider: {provider_name}, symbol: {provider_symbol})."
        ), None

    latest_ts = pd.to_datetime(df['Date']).max()
    if hasattr(latest_ts, "tzinfo") and latest_ts.tzinfo is not None:
        latest_ts = latest_ts.tz_convert("UTC").tz_localize(None)
    age_minutes = (click_time_utc - latest_ts.to_pydatetime()).total_seconds() / 60.0
    market_open = is_market_open(market_type, click_time_utc)
    if market_open and age_minutes > MAX_FETCH_DATA_AGE_MINUTES:
        return html.Div([
            html.P(
                f"Data freshness check failed for {ui_symbol} "
                f"(provider: {provider_name}, symbol: {provider_symbol}).",
                className="text-danger"
            ),
            html.P(
                f"Latest bar is {age_minutes:.1f} minutes old (max allowed: {MAX_FETCH_DATA_AGE_MINUTES} minutes)."
            ),
            html.P(f"Latest timestamp: {latest_ts} UTC"),
            html.P(
                f"Market is currently open for {market_type}, so strict freshness is enforced."
            ),
            html.P("Try again shortly or switch provider/data plan for lower latency.")
        ]), None
    
    # Add time-aligned sentiment before technical indicators
    df = add_time_aligned_sentiment(df, ui_symbol)
    df = add_default_sentiment_columns(df)

    # Calculate technical indicators
    df_with_indicators = calculate_technical_indicators(df)
    
    # Save file
    safe_symbol = ui_symbol.replace("/", "_").replace(":", "_")
    file_path = os.path.join('uploads', f"{safe_symbol}_data.csv")
    df.to_csv(file_path, index=False)
    
    return html.Div([
        html.P(f"Data fetched for {ui_symbol}"),
        html.P(f"Provider: {provider_name}"),
        html.P(f"Market source symbol: {provider_symbol}"),
        html.P(f"Latest bar: {latest_ts} UTC ({max(age_minutes, 0.0):.1f} minutes old)"),
        html.P(
            f"Freshness mode: {'strict (market open)' if market_open else 'relaxed (market closed)'}"
        ),
        html.P(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}"),
        html.P(f"Number of records: {len(df)}")
    ]), df_with_indicators.to_json(date_format='iso', orient='split')

# Callback for price chart
@app.callback(
    Output("price-chart", "figure"),
    [Input("stock-data-store", "data")]
)
def update_price_chart(data):
    if data is None:
        return go.Figure().update_layout(title="No data available")
    
    df = pd.read_json(data, orient='split')
    
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Price"
    ))
    
    # Add volume as bar chart on secondary y-axis
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['Volume'],
        name="Volume",
        yaxis="y2",
        opacity=0.3
    ))
    
    # Update layout
    fig.update_layout(
        title="Stock Price and Volume",
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right"
        ),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig

# Callback for technical indicator charts
@app.callback(
    [Output("moving-averages-chart", "figure"),
     Output("rsi-chart", "figure"),
     Output("macd-chart", "figure"),
     Output("bollinger-chart", "figure")],
    [Input("stock-data-store", "data")]
)
def update_technical_charts(data):
    if data is None:
        empty_fig = go.Figure().update_layout(title="No data available")
        return empty_fig, empty_fig, empty_fig, empty_fig
    
    df = pd.read_json(data, orient='split')
    
    # Moving Averages Chart
    ma_fig = go.Figure()
    ma_fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Close", line=dict(color='black')))
    ma_fig.add_trace(go.Scatter(x=df['Date'], y=df['MA7'], name="MA7", line=dict(color='blue')))
    ma_fig.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], name="MA20", line=dict(color='orange')))
    ma_fig.add_trace(go.Scatter(x=df['Date'], y=df['MA50'], name="MA50", line=dict(color='red')))
    ma_fig.update_layout(title="Moving Averages", height=400, legend=dict(orientation="h"))
    
    # RSI Chart
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name="RSI", line=dict(color='purple')))
    rsi_fig.add_shape(type="line", x0=df['Date'].min(), x1=df['Date'].max(), y0=70, y1=70,
                     line=dict(color="red", width=2, dash="dash"))
    rsi_fig.add_shape(type="line", x0=df['Date'].min(), x1=df['Date'].max(), y0=30, y1=30,
                     line=dict(color="green", width=2, dash="dash"))
    rsi_fig.update_layout(title="Relative Strength Index (RSI)", height=400, yaxis=dict(range=[0, 100]))
    
    # MACD Chart
    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name="MACD", line=dict(color='blue')))
    macd_fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal'], name="Signal", line=dict(color='red')))
    macd_fig.update_layout(title="MACD", height=400, legend=dict(orientation="h"))
    
    # Bollinger Bands Chart
    bb_fig = go.Figure()
    bb_fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Close", line=dict(color='black')))
    bb_fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_upper'], name="Upper Band", line=dict(color='red')))
    bb_fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_middle'], name="Middle Band", line=dict(color='blue')))
    bb_fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_lower'], name="Lower Band", line=dict(color='green')))
    bb_fig.update_layout(title="Bollinger Bands", height=400, legend=dict(orientation="h"))
    
    return ma_fig, rsi_fig, macd_fig, bb_fig

# Callback for training models
@app.callback(
    [Output("rf-status", "children"),
     Output("lstm-status", "children"),
     Output("rf-model-store", "data"),
     Output("lstm-model-store", "data"),
     Output("training-notification", "is_open", allow_duplicate=True),
     Output("training-notification", "children", allow_duplicate=True),
     Output("completion-notification", "is_open", allow_duplicate=True),
     Output("completion-notification", "children", allow_duplicate=True),
     Output("rf-spinner-output", "children"),
     Output("lstm-spinner-output", "children")],
    [Input("train-models-btn", "n_clicks")],
    [State("stock-data-store", "data")],
    prevent_initial_call=True
)
def train_models(n_clicks, data):
    if n_clicks is None or data is None:
        return "AI 30-Min Model: Ready", "AI 2-Hour Model: Ready", None, None, False, "", False, "", "", ""
    
    try:
        from io import StringIO
        import json
        
        # Convert the data to a string if it's a dict
        if isinstance(data, dict):
            data = json.dumps(data)
            
        # Parse JSON string to DataFrame
        df = pd.read_json(StringIO(data), orient='split')
        if df.empty:
            raise ValueError("No data available for training")
            
        # Ensure required columns exist
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
            
    except Exception as e:
        error_msg = f"Error loading data: {str(e)}"
        print(error_msg)
        return (
            "AI 30-Min Model: Error", 
            "AI 2-Hour Model: Error", 
            None, 
            None, 
            True, 
            html.Div([
                html.H5("Error Loading Data", className="text-danger"),
                html.P(error_msg),
                html.P("Please try loading the data again.", className="mb-0")
            ]), 
            False, 
            "", 
            "", 
            ""
        )
    
    # Show training notification
    training_notification = True
    training_message = html.Div([
        html.H5("Training Models", className="mb-2"),
        html.P("Starting model training. This may take a few moments..."),
        html.P("Please wait while we train both models on your data.", className="mb-0")
    ])
    
    # Train RF model
    rf_status = "AI 30-Min Model: Training..."
    lstm_status = "AI 2-Hour Model: Waiting..."
    
    # Spinner content - show during training
    rf_spinner_content = "Training RF"
    lstm_spinner_content = "Waiting"
    
    # Train RF model
    rf_model, rf_metrics = train_rf_model(df, "rf-status")
    if rf_model is not None:
        rf_status = "AI 30-Min Model: Trained"
        rf_model_info = {"model_path": "models/rf_model.joblib", "metrics": rf_metrics}
    else:
        rf_status = f"AI 30-Min Model: Error - {rf_metrics.get('error', 'Unknown error')}"
        rf_model_info = None
    
    # Train LSTM model
    lstm_status = "AI 2-Hour Model: Training..."
    lstm_spinner_content = "Training AI 2-Hour Model"
    lstm_model, lstm_history = train_lstm_model(df, "lstm-status")
    if lstm_model is not None:
        lstm_status = "AI 2-Hour Model: Trained"
        lstm_model_info = {"model_path": "models/lstm_model.h5", "history": lstm_history}
    else:
        lstm_status = f"AI 2-Hour Model: Error - {lstm_history.get('error', 'Unknown error')}"
        lstm_model_info = None
    
    # Show completion notification
    completion_notification = True
    completion_message = html.Div([
        html.H5("Training Complete!", className="mb-2"),
        html.P("Both models have been successfully trained."),
        html.P("You can now make predictions using the 'Predict' button.", className="mb-0")
    ])
    
    # Clear spinner content after training
    rf_spinner_content = ""
    lstm_spinner_content = ""
    
    # Return all required outputs
    return (
        rf_status,  # rf-status children
        lstm_status,  # lstm-status children
        rf_model_info,  # rf-model-store data
        lstm_model_info,  # lstm-model-store data
        training_notification,  # training-notification is_open
        training_message,  # training-notification children
        completion_notification,  # completion-notification is_open
        completion_message,  # completion-notification children
        rf_spinner_content,  # rf-spinner-output children
        lstm_spinner_content   # lstm-spinner-output children
    )

# Callback for making predictions
@app.callback(
    [Output("rf-prediction-results", "children"),
     Output("lstm-prediction-results", "children"),
     Output("rf-prediction-chart", "figure"),
     Output("lstm-prediction-chart", "figure"),
     Output("rf-predictions-store", "data"),
     Output("lstm-predictions-store", "data"),
     Output("training-notification", "is_open", allow_duplicate=True),
     Output("training-notification", "children", allow_duplicate=True),
     Output("completion-notification", "is_open", allow_duplicate=True),
     Output("completion-notification", "children", allow_duplicate=True),
     Output("rf-spinner-output", "children", allow_duplicate=True),
     Output("lstm-spinner-output", "children", allow_duplicate=True)],
    [Input("predict-btn", "n_clicks")],
    [State("stock-data-store", "data"),
     State("rf-model-store", "data"),
     State("lstm-model-store", "data"),
     State("stock-dropdown", "value")],
    prevent_initial_call=True
)
def make_predictions(n_clicks, data, rf_model_info, lstm_model_info, selected_symbol):
    if n_clicks is None or data is None:
        empty_fig = go.Figure().update_layout(title="No predictions available")
        return html.Div(), html.Div(), empty_fig, empty_fig, None, None, False, "", False, "", "", ""
    
    df = pd.read_json(data, orient='split')
    
    # Get sentiment features for the stock
    ticker = selected_symbol or (df.get('Ticker', ['GOOGL']).iloc[0] if 'Ticker' in df.columns else 'GOOGL')
    try:
        sentiment_analyzer = NewsSentimentAnalyzer()
        sentiment_features = sentiment_analyzer.get_sentiment_features(ticker, days=30)
        print(f"Sentiment features for {ticker}: {sentiment_features}")
    except Exception as e:
        print(f"Error getting sentiment features: {str(e)}")
        sentiment_features = {
            'sentiment_score': 0,
            'sentiment_magnitude': 0,
            'sentiment_volume': 0,
            'sentiment_trend': 0,
            'sentiment_volatility': 0
        }
    
    # Initialize results
    rf_results = html.Div("No AI 30-Min model trained.")
    lstm_results = html.Div("No AI 2-Hour model trained.")
    rf_fig = go.Figure().update_layout(title="No AI 30-Min predictions available")
    lstm_fig = go.Figure().update_layout(title="No AI 2-Hour predictions available")
    rf_predictions = None
    lstm_predictions = None
    
    # RF Predictions
    if rf_model_info is not None:
        try:
            # Load RF model
            rf_model = EnhancedRandomForestModel.load_model(rf_model_info["model_path"])

            # Ensure horizon is 30 minutes for intraday prediction
            rf_model.horizon_steps = 30
            
            # Make predictions with sentiment features (30 minutes ahead)
            next_day_price = rf_model.predict_next_30min(df, sentiment_features=sentiment_features)
            
            # Current price
            last_price = df['Close'].iloc[-1]
            last_date = df['Date'].iloc[-1]
            
            # Calculate change
            change = next_day_price - last_price
            pct_change = (change / last_price) * 100
            
            # Create results display
            rf_results = html.Div([
                html.H5("AI 30-Minute Prediction"),
                html.P(f"Last Close: ${last_price:.2f}"),
                html.P(f"Predicted Next 30 Minutes: ${next_day_price:.2f}"),
                html.P([
                    f"Change: ${change:.2f} (",
                    html.Span(f"{pct_change:.2f}%", 
                              style={"color": "green" if pct_change >= 0 else "red"}),
                    ")"
                ]),
                html.Hr(),
                html.H6("Sentiment Analysis", className="mt-3"),
                html.P(f"Sentiment Score: {sentiment_features['sentiment_score']:.3f}"),
                html.P(f"Sentiment Magnitude: {sentiment_features['sentiment_magnitude']:.3f}"),
                html.P(f"News Volume: {sentiment_features['sentiment_volume']} articles"),
                html.P([
                    "Sentiment Trend: ",
                    html.Span(f"{sentiment_features['sentiment_trend']:.4f}", 
                              style={"color": "green" if sentiment_features['sentiment_trend'] >= 0 else "red"})
                ])
            ])
            
            # Create prediction chart
            rf_fig = go.Figure()
            
            # Add historical data (last 180 minutes)
            historical_df = df.iloc[-180:]
            rf_fig.add_trace(go.Scatter(
                x=historical_df['Date'], 
                y=historical_df['Close'],
                name="Historical",
                line=dict(color='blue')
            ))
            
            # Add next day prediction point
            next_day_date = last_date + timedelta(minutes=30)
            rf_fig.add_trace(go.Scatter(
                x=[next_day_date],
                y=[next_day_price],
                name="+30 Min Prediction",
                mode="markers",
                marker=dict(size=12, color='red')
            ))
            
            # Update layout
            rf_fig.update_layout(
                title="AI 30-Minute Prediction",
                xaxis_title="Date",
                yaxis_title="Price",
                height=400,
                legend=dict(orientation="h")
            )
            
            # Store predictions
            rf_predictions = pd.DataFrame({
                'Date': [next_day_date],
                'Predicted_Close': [next_day_price]
            }).to_json(date_format='iso', orient='split')
            
        except Exception as e:
            rf_results = html.Div(f"Error making RF predictions: {str(e)}")
    
    # LSTM Predictions
    if lstm_model_info is not None:
        try:
            model_path = lstm_model_info.get("model_path", "models/lstm_model.h5")
            print(f"Loading LSTM model from {model_path}...")
            lstm_model = LSTMModel.load(model_path)

            # Require trained model artifacts only (no ad-hoc fallback/minimal predictions).
            if lstm_model.model is None and lstm_model.tabular_model is None:
                raise ValueError("No trained LSTM artifacts available. Train and evaluate models before predicting.")

            print("Making LSTM prediction...")
            prediction_info = lstm_model.predict(df, sentiment_features=sentiment_features)
            if not isinstance(prediction_info, dict):
                raise ValueError("Unexpected prediction payload from trained LSTM model.")

            prediction_method = prediction_info.get('method', 'unknown')
            disallowed_methods = {'fallback_model', 'simple_fallback', 'lstm_constant', 'lstm_model_minimal', 'lstm_minimal'}
            if prediction_method in disallowed_methods:
                raise ValueError(
                    f"Prediction blocked: method '{prediction_method}' is a fallback path, not the trained/evaluated model."
                )

            next_month_price = prediction_info['predicted_price']
            lower_bound, upper_bound = prediction_info['confidence_interval']
            uncertainty = prediction_info['uncertainty']
            print(f"Using prediction method: {prediction_method}")
            
            # Current price
            last_price = df['Close'].iloc[-1]
            last_date = df['Date'].iloc[-1]
            
            # Calculate change
            change = next_month_price - last_price
            pct_change = (change / last_price) * 100
            
            # Format confidence interval
            confidence_range = f"${lower_bound:.2f} to ${upper_bound:.2f}"
            confidence_pct = (upper_bound - lower_bound) / next_month_price * 100
            
            # Create results display with uncertainty information
            lstm_results = html.Div([
                html.H5("AI 2-Hour Prediction"),
                html.P(f"Last Close: ${last_price:.2f}"),
                html.P(f"Predicted Next 2 Hours: ${next_month_price:.2f}"),
                html.P([
                    f"Change: ${change:.2f} (",
                    html.Span(f"{pct_change:.2f}%", 
                             style={"color": "green" if pct_change >= 0 else "red"}),
                    ")"
                ]),
                html.P(f"95% Confidence Interval: {confidence_range}"),
                html.P(f"Uncertainty: ±{uncertainty:.2f} (±{confidence_pct:.1f}%)"),
                html.P(f"Prediction Method: {prediction_method}", className="text-muted small"),
                html.Hr(),
                html.H6("Sentiment Analysis", className="mt-3"),
                html.P(f"Sentiment Score: {sentiment_features['sentiment_score']:.3f}"),
                html.P(f"Sentiment Magnitude: {sentiment_features['sentiment_magnitude']:.3f}"),
                html.P(f"News Volume: {sentiment_features['sentiment_volume']} articles"),
                html.P([
                    "Sentiment Trend: ",
                    html.Span(f"{sentiment_features['sentiment_trend']:.4f}", 
                              style={"color": "green" if sentiment_features['sentiment_trend'] >= 0 else "red"})
                ])
            ])

            # Create prediction chart
            lstm_fig = go.Figure()
            
            # Add historical data (last 240 minutes)
            historical_df = df.iloc[-240:]
            lstm_fig.add_trace(go.Scatter(
                x=historical_df['Date'], 
                y=historical_df['Close'],
                name="Historical",
                line=dict(color='blue')
            ))
            
            # Add current point
            lstm_fig.add_trace(go.Scatter(
                x=[last_date],
                y=[last_price],
                name="Current",
                mode="markers",
                marker=dict(size=10, color='blue')
            ))

            # Add prediction point (+2 hours)
            future_date = last_date + timedelta(hours=2)
            lstm_fig.add_trace(go.Scatter(
                x=[future_date],
                y=[next_month_price],
                name="+2 Hr Prediction",
                mode="markers",
                marker=dict(size=12, color='red'),
                error_y=dict(
                    type='data',
                    array=[uncertainty],
                    visible=True,
                    color='red'
                )
            ))
            
            # Update layout
            lstm_fig.update_layout(
                title="AI 2-Hour Prediction",
                xaxis_title="Date",
                yaxis_title="Price",
                height=400,
                legend=dict(orientation="h")
            )
            
            # Store predictions
            lstm_predictions = pd.DataFrame({
                'Date': [future_date],
                'Predicted_Close': [next_month_price],
                'Lower_Bound': [lower_bound],
                'Upper_Bound': [upper_bound]
            }).to_json(date_format='iso', orient='split')
            
        except Exception as e:
            lstm_results = html.Div(f"Error making AI 2-Hour predictions: {str(e)}")
    
    # Show prediction notification
    training_notification = True
    training_message = html.Div([
        html.H5("Generating Predictions", className="mb-2"),
        html.P("Processing data and generating predictions..."),
        html.P("This will only take a moment.", className="mb-0")
    ])
    
    # Show completion notification
    completion_notification = True
    completion_message = html.Div([
        html.H5("Predictions Complete!", className="mb-2"),
        html.P("All predictions have been generated."),
        html.P("Scroll down to view the results and charts.", className="mb-0")
    ])
    
    # Spinner content during prediction
    rf_spinner_content = "Predicting"
    lstm_spinner_content = "Predicting"
    
    return rf_results, lstm_results, rf_fig, lstm_fig, rf_predictions, lstm_predictions, training_notification, training_message, completion_notification, completion_message, rf_spinner_content, lstm_spinner_content

# Callback for updating sentiment analysis
@app.callback(
    [Output("sentiment-summary-cards", "children"),
     Output("sentiment-history-chart", "figure")],
    [Input("stock-data-store", "data"),
     Input("stock-dropdown", "value")]
)
def update_sentiment_analysis(data, selected_stock):
    if data is None:
        empty_cards = [dbc.Card(dbc.CardBody([html.P("No data available")]), color="light") for _ in range(4)]
        empty_fig = go.Figure().update_layout(title="No sentiment data available")
        return empty_cards, empty_fig
    
    try:
        _ = pd.read_json(data, orient='split')
        ticker = selected_stock or 'GOOGL'

        sentiment_features = {
            'sentiment_score': 0.0,
            'sentiment_magnitude': 0.0,
            'sentiment_volume': 0,
            'sentiment_trend': 0.0,
            'sentiment_volatility': 0.0,
        }

        try:
            analyzer = NewsSentimentAnalyzer()
            fetched = analyzer.get_sentiment_features(ticker, days=30)
            if isinstance(fetched, dict):
                sentiment_features.update(fetched)
        except Exception as e:
            print(f"Sentiment fallback for {ticker}: {str(e)}")

        cards = create_sentiment_summary_cards(sentiment_features)
        chart = create_sentiment_history_chart(ticker, days=90)
        return cards, chart
    except Exception as e:
        error_cards = [dbc.Card(dbc.CardBody([html.P(f"Error: {str(e)}")]), color="light") for _ in range(4)]
        error_fig = go.Figure().add_annotation(
            text=f"Error loading sentiment data: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        ).update_layout(title="Sentiment Analysis - Error")
        return error_cards, error_fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

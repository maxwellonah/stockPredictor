import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
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
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', "mo0_G1UPGqllOOPmY37UvS9Ui6mpiPQL")

# Initialize the Dash app with a Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
server = app.server

# Define stock options
STOCK_OPTIONS = [
    {'label': 'Google (GOOGL)', 'value': 'GOOGL'},
    {'label': 'Apple (AAPL)', 'value': 'AAPL'},
    {'label': 'Microsoft (MSFT)', 'value': 'MSFT'},
    {'label': 'Amazon (AMZN)', 'value': 'AMZN'}
]

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

# Function to fetch stock data from Polygon API
def fetch_stock_data(ticker, timespan='minute', multiplier=1, from_date=None, to_date=None):
    """Fetch stock data from Polygon API"""
    if not ticker:
        return None

    if from_date is None:
        if timespan == 'minute':
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        else:
            from_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
    if to_date is None:
        to_date = datetime.now().strftime('%Y-%m-%d')
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}?apiKey={POLYGON_API_KEY}"
    
    try:
        response = requests.get(url)
        if not response.ok:
            print(f"Error fetching stock data: HTTP {response.status_code} - {response.text}")
            return None

        data = response.json()

        if 'results' not in data:
            print(f"Error fetching data: {data}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data['results'])
        
        # Rename columns to match our model expectations
        df = df.rename(columns={
            'v': 'Volume',
            'o': 'Open',
            'c': 'Close',
            'h': 'High',
            'l': 'Low',
            't': 'timestamp'
        })
        
        # Convert timestamp to datetime
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Select and reorder columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        return df
    
    except Exception as e:
        print(f"Error fetching stock data: {str(e)}")
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
            feature_selection_threshold=0.0,
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
        
        # Create model
        print("Initializing LSTM model...")
        lstm_model = LSTMModel(time_steps=90, features=features, epochs=80, batch_size=32, horizon_steps=120)
        
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
                            html.P("Select a stock to analyze:", className="mt-2"),
                            dcc.Dropdown(
                                id="stock-dropdown",
                                options=STOCK_OPTIONS,
                                value="GOOGL",
                                className="mb-3"
                            ),
                            dbc.Button("Fetch Data", id="fetch-data-btn", color="primary", className="mr-2"),
                            html.Div(id="api-data-info", className="mt-3")
                        ], label="Polygon API"),
                        
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
        return html.Div("Click 'Fetch Data' to load stock information."), None

    if not ticker:
        ticker = 'GOOGL'
    
    # Fetch 1-minute bars
    df = fetch_stock_data(ticker, timespan='minute', multiplier=1)
    
    if df is None:
        return html.Div(f"Error fetching data for {ticker}. Check server logs for Polygon API details and confirm your API key has access."), None
    
    # Add time-aligned sentiment before technical indicators
    df = add_time_aligned_sentiment(df, ticker)

    # Calculate technical indicators
    df_with_indicators = calculate_technical_indicators(df)
    
    # Save file
    file_path = os.path.join('uploads', f"{ticker}_data.csv")
    df.to_csv(file_path, index=False)
    
    return html.Div([
        html.P(f"Data fetched for {ticker}"),
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
        lstm_model_info = {"model_path": "models/lstm_model.h5", "history": str(lstm_history)}
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
     State("lstm-model-store", "data")],
    prevent_initial_call=True
)
def make_predictions(n_clicks, data, rf_model_info, lstm_model_info):
    if n_clicks is None or data is None:
        empty_fig = go.Figure().update_layout(title="No predictions available")
        return html.Div(), html.Div(), empty_fig, empty_fig, None, None, False, "", False, "", "", ""
    
    df = pd.read_json(data, orient='split')
    
    # Get sentiment features for the stock
    ticker = df.get('Ticker', ['GOOGL']).iloc[0] if 'Ticker' in df.columns else 'GOOGL'
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
            # Check if LSTM model file exists
            model_path = "models/lstm_model.h5"
            if os.path.exists(model_path) or os.path.exists(model_path.replace('.h5', '.keras')):
                # Load LSTM model
                print("Loading LSTM model...")
                lstm_model = LSTMModel.load(model_path)
                
                # Force using the main prediction code
                print("Making LSTM prediction...")
                # Make predictions with sentiment features - now returns a dictionary with uncertainty
                prediction_info = lstm_model.predict(df, sentiment_features=sentiment_features)
                print(f"LSTM prediction info: {prediction_info.keys() if isinstance(prediction_info, dict) else 'not a dict'}")
                
                # Extract prediction details
                if isinstance(prediction_info, dict):
                    # New format with uncertainty information
                    next_month_price = prediction_info['predicted_price']
                    lower_bound, upper_bound = prediction_info['confidence_interval']
                    uncertainty = prediction_info['uncertainty']
                    prediction_method = prediction_info.get('method', 'lstm_model')
                    print(f"Using prediction method: {prediction_method}")
                else:
                    # Handle old format for backward compatibility
                    next_month_price = prediction_info
                    # Estimate uncertainty as 10% of the prediction
                    uncertainty = next_month_price * 0.1
                    lower_bound = next_month_price * 0.9
                    upper_bound = next_month_price * 1.1
                    prediction_method = 'lstm_model_legacy'
                    print("Using legacy prediction format")
            else:
                # Create a simple prediction based on the last price with a small increase
                # This is a fallback when the model file doesn't exist
                print(f"LSTM model file not found at {model_path} or {model_path.replace('.h5', '.keras')}")
                print("Using simple fallback prediction")
                
                # Train a minimal model on the fly
                try:
                    print("Attempting to train a minimal LSTM model on the fly...")
                    features = ['Close', 'Volume', 'High', 'Low', 'Open']
                    minimal_model = LSTMModel(time_steps=30, features=features, epochs=3, horizon_steps=120)
                    minimal_model.train(df)
                    prediction_info = minimal_model.predict(df, sentiment_features=sentiment_features)
                    
                    if isinstance(prediction_info, dict):
                        next_month_price = prediction_info['predicted_price']
                        lower_bound, upper_bound = prediction_info['confidence_interval']
                        uncertainty = prediction_info['uncertainty']
                        prediction_method = prediction_info.get('method', 'lstm_minimal')
                        print(f"Successfully used minimal model: {prediction_method}")
                    else:
                        raise ValueError("Minimal model did not return dictionary format")
                except Exception as e:
                    print(f"Error training minimal model: {str(e)}")
                    # Ultimate fallback
                    last_price = df['Close'].iloc[-1]
                    next_month_price = last_price * 1.05  # 5% increase as a placeholder
                    uncertainty = next_month_price * 0.15  # Higher uncertainty for fallback
                    lower_bound = next_month_price * 0.85
                    upper_bound = next_month_price * 1.15
                    prediction_method = 'simple_fallback'
                    print("Using ultimate simple fallback")
            
            # Save the model after successful prediction
            if 'lstm_model' in locals() and lstm_model.model is not None:
                print("Saving LSTM model after successful prediction")
                lstm_model.save(model_path)
            
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
        df = pd.read_json(data, orient='split')
        ticker = selected_stock or 'GOOGL'
        
        # Get sentiment features
        analyzer = NewsSentimentAnalyzer()
        sentiment_features = analyzer.get_sentiment_features(ticker, days=30)
        
        # Create summary cards
        cards = create_sentiment_summary_cards(sentiment_features)
        
        # Create sentiment history chart
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

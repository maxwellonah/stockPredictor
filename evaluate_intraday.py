import argparse
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta

import app
from rf_model import EnhancedRandomForestModel
from lstm_model import LSTMModel


def _directional_accuracy(y_true, y_pred, base_price):
    actual_dir = np.sign(np.asarray(y_true) - np.asarray(base_price))
    pred_dir = np.sign(np.asarray(y_pred) - np.asarray(base_price))
    return float((actual_dir == pred_dir).mean() * 100.0)


def evaluate_rf_intraday(df, horizon_steps=5, test_size=0.2, random_state=42):
    d = df.copy().dropna(subset=["Close"]).reset_index(drop=True)
    model = EnhancedRandomForestModel(
        feature_selection_threshold=0.01,
        random_state=random_state,
        horizon_steps=horizon_steps,
    )

    full_feat = model.create_features(d)
    split_feat = int(len(full_feat) * (1.0 - test_size))
    split_feat = min(max(split_feat, 1), len(full_feat) - 1)
    split_date = full_feat["Date"].iloc[split_feat]
    train_df = d[d["Date"] <= split_date].copy().reset_index(drop=True)

    train_metrics = model.train(train_df, cv=3)

    # Build features on full timeline and evaluate on aligned tail split.
    full_feat = model.create_features(d)
    eval_df = full_feat.iloc[split_feat:].copy().reset_index(drop=True)
    if eval_df.empty:
        raise ValueError("No RF evaluation samples available after feature alignment.")

    X_test = eval_df.drop(columns=["Date", "Close", "Next_Day_Close"], errors="ignore")
    y_true = eval_df["Next_Day_Close"].values
    y_pred = model.predict_from_features(X_test, eval_df["Close"].values, apply_direction=True)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    denom = np.maximum(np.abs(y_true), 1e-8)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    dir_acc = _directional_accuracy(y_true, y_pred, base_price=eval_df["Close"].values)

    return {
        "rows": int(len(d)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(eval_df)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "directional_accuracy": dir_acc,
        "best_params": train_metrics.get("best_params", {}),
    }


def evaluate_lstm_intraday(df, horizon_steps=10, test_size=0.2, epochs=3, batch_size=64, time_steps=60):
    d = df.copy().dropna(subset=["Close"]).reset_index(drop=True)

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

    model = LSTMModel(
        time_steps=time_steps,
        features=features,
        epochs=epochs,
        batch_size=batch_size,
        horizon_steps=horizon_steps,
    )

    full_feat = model.create_features(d)
    split_feat = int(len(full_feat) * (1.0 - test_size))
    split_feat = min(max(split_feat, 1), len(full_feat) - 1)
    split_date = full_feat["Date"].iloc[split_feat]
    train_df = d[d["Date"] <= split_date].copy().reset_index(drop=True)

    model.train(train_df)

    # Build aligned feature/target dataset and split after alignment.
    full_feat = model.create_features(d)
    eval_columns = list(dict.fromkeys(model.features + ['Close']))
    eval_df = full_feat[eval_columns].copy()
    eval_df['Target_Close'] = full_feat['Close'].shift(-horizon_steps)
    eval_df = eval_df.dropna().reset_index(drop=True)
    split_eval = min(max(split_feat, 1), len(eval_df) - 1)
    eval_df = eval_df.iloc[split_eval:].copy().reset_index(drop=True)

    if eval_df.empty:
        raise ValueError("No evaluation samples available for LSTM after feature/target alignment.")

    X_test = eval_df[model.features].values
    y_test_inv = eval_df['Target_Close'].values
    base_prices = eval_df['Close'].values

    y_pred_inv = model.predict_tabular_batch(eval_df)

    mse = float(mean_squared_error(y_test_inv, y_pred_inv))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test_inv, y_pred_inv))
    denom = np.maximum(np.abs(y_test_inv), 1e-8)
    mape = float(np.mean(np.abs((y_test_inv - y_pred_inv) / denom)) * 100.0)
    r2 = float(r2_score(y_test_inv, y_pred_inv))
    dir_acc = _directional_accuracy(y_test_inv, y_pred_inv, base_prices)

    out = {
        "rows": int(len(d)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(eval_df)),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "directional_accuracy": dir_acc,
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="GOOGL")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--rf_horizon", type=int, default=5)
    parser.add_argument("--lstm_horizon", type=int, default=10)
    parser.add_argument("--lstm_epochs", type=int, default=15)
    parser.add_argument("--lstm_time_steps", type=int, default=60)
    parser.add_argument("--csv", default=None, help="Optional local CSV path with Date,Open,High,Low,Close,Volume")
    args = parser.parse_args()

    ticker = args.ticker

    print(f"Evaluating intraday models for {ticker} on 1-minute bars")
    print(f"RF horizon: {args.rf_horizon} minutes | LSTM horizon: {args.lstm_horizon} minutes | LSTM time_steps: {args.lstm_time_steps}")

    if args.csv:
        import pandas as pd
        df = pd.read_csv(args.csv)
        df['Date'] = pd.to_datetime(df['Date'])
        if 'Sentiment_Score' not in df.columns:
            df = app.add_time_aligned_sentiment(df, ticker)
    else:
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
        df = app.fetch_stock_data(ticker, timespan="minute", multiplier=1, from_date=from_date, to_date=to_date)
        if df is None or df.empty:
            raise SystemExit("Failed to fetch 1-minute data")
        df = app.add_time_aligned_sentiment(df, ticker)

    df = app.calculate_technical_indicators(df)
    df = df.sort_values("Date").reset_index(drop=True)

    rf_metrics = evaluate_rf_intraday(df, horizon_steps=args.rf_horizon)
    print("RF_5MIN" if args.rf_horizon == 5 else f"RF_{args.rf_horizon}")
    for k in ["rows", "train_rows", "test_rows", "mae", "rmse", "r2", "mape", "directional_accuracy"]:
        print(f"{k}: {rf_metrics[k]}")
    print(f"best_params: {rf_metrics.get('best_params', {})}")

    lstm_metrics = evaluate_lstm_intraday(
        df,
        horizon_steps=args.lstm_horizon,
        epochs=args.lstm_epochs,
        time_steps=args.lstm_time_steps,
    )
    print("LSTM_10MIN" if args.lstm_horizon == 10 else f"LSTM_{args.lstm_horizon}")
    for k in ["rows", "train_rows", "test_rows", "mae", "rmse", "r2", "mape", "directional_accuracy"]:
        print(f"{k}: {lstm_metrics[k]}")


if __name__ == "__main__":
    main()

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


def evaluate_rf_intraday(df, horizon_steps=30, test_size=0.2, random_state=42):
    d = df.copy().dropna(subset=["Close"]).reset_index(drop=True)

    split = int(len(d) * (1.0 - test_size))
    train_df = d.iloc[:split].copy()
    test_df = d.iloc[split:].copy()

    model = EnhancedRandomForestModel(
        feature_selection_threshold=0.0,
        random_state=random_state,
        horizon_steps=horizon_steps,
    )
    train_metrics = model.train(train_df, cv=3)

    feats_test = model.create_features(test_df)
    X_test = feats_test.drop(columns=["Date", "Close", "Next_Day_Close"], errors="ignore")

    # Align test columns to training feature order
    X_test = X_test.reindex(columns=model.feature_columns, fill_value=0)
    if model.selected_features is not None:
        X_test = X_test[model.selected_feature_names]

    y_true = feats_test["Next_Day_Close"].values
    y_pred = model.pipeline.predict(X_test.values)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)
    dir_acc = _directional_accuracy(y_true, y_pred, base_price=feats_test["Close"].values)

    return {
        "rows": int(len(d)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "directional_accuracy": dir_acc,
        "best_params": train_metrics.get("best_params", {}),
    }


def evaluate_lstm_intraday(df, horizon_steps=120, test_size=0.2, epochs=3, batch_size=64, time_steps=60):
    d = df.copy().dropna(subset=["Close"]).reset_index(drop=True)

    split = int(len(d) * (1.0 - test_size))
    train_df = d.iloc[:split].copy()
    test_df = d.iloc[split:].copy()

    features = [
        "Close",
        "Volume",
        "MA7",
        "MA20",
        "RSI",
        "MACD",
        "Sentiment_Score",
        "Sentiment_Magnitude",
        "Sentiment_Volume",
        "Sentiment_Trend",
        "Sentiment_Volatility",
    ]

    model = LSTMModel(
        time_steps=time_steps,
        features=features,
        epochs=epochs,
        batch_size=batch_size,
        horizon_steps=horizon_steps,
    )

    model.train(train_df)

    # Build evaluation sequences using context from end of train so sequences span boundary
    df_train_feat = model.create_features(train_df)
    df_test_feat = model.create_features(test_df)

    context_len = time_steps + horizon_steps
    tail = df_train_feat.iloc[-context_len:].copy() if len(df_train_feat) >= context_len else df_train_feat.copy()
    combined_close = np.concatenate([
        tail['Close'].values,
        df_test_feat['Close'].values,
    ], axis=0)

    combined = np.concatenate([
        model.scaler.transform(tail[model.features].values),
        model.scaler.transform(df_test_feat[model.features].values),
    ], axis=0)

    # Create sequences from combined, but keep only those whose target falls inside test region
    X_all, y_all = model.create_sequences(combined)

    # In create_sequences, y index is i + time_steps + horizon_steps
    # Combined index mapping:
    # - tail occupies [0, tail_len)
    # - test occupies [tail_len, tail_len + len(test_feat))
    tail_len = len(tail)
    test_start = tail_len

    seq_indices = np.arange(len(y_all))
    target_indices = seq_indices + time_steps + horizon_steps
    mask = target_indices >= test_start
    X_test = X_all[mask]
    y_test = y_all[mask]

    # Base price for directional accuracy: last close of the input window
    base_indices = (seq_indices + time_steps - 1)[mask]
    base_prices = combined_close[base_indices]

    y_pred = model.model.predict(X_test, verbose=0).flatten()

    # Inverse transform predictions and actual values
    dummy = np.zeros((len(y_pred), len(model.features)))
    dummy[:, model.target_index] = y_pred
    y_pred_inv = model.scaler.inverse_transform(dummy)[:, model.target_index]

    dummy = np.zeros((len(y_test), len(model.features)))
    dummy[:, model.target_index] = y_test
    y_test_inv = model.scaler.inverse_transform(dummy)[:, model.target_index]

    mse = float(np.mean((y_pred_inv - y_test_inv) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test_inv, y_pred_inv))
    mape = float(np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100.0)
    dir_acc = _directional_accuracy(y_test_inv, y_pred_inv, base_price=base_prices)

    out = {
        "rows": int(len(d)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "directional_accuracy": dir_acc,
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="GOOGL")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--rf_horizon", type=int, default=30)
    parser.add_argument("--lstm_horizon", type=int, default=120)
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
    print("RF_30MIN" if args.rf_horizon == 30 else f"RF_{args.rf_horizon}")
    for k in ["rows", "train_rows", "test_rows", "mae", "rmse", "r2", "mape", "directional_accuracy"]:
        print(f"{k}: {rf_metrics[k]}")
    print(f"best_params: {rf_metrics.get('best_params', {})}")

    lstm_metrics = evaluate_lstm_intraday(
        df,
        horizon_steps=args.lstm_horizon,
        epochs=args.lstm_epochs,
        time_steps=args.lstm_time_steps,
    )
    print("LSTM_2HR" if args.lstm_horizon == 120 else f"LSTM_{args.lstm_horizon}")
    for k, v in lstm_metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

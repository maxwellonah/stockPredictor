import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D, Attention
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
import joblib
import os
import math

class LSTMModel:
    def __init__(self, time_steps=60, features=None, epochs=100, batch_size=64, horizon_steps=10):
        self.time_steps = time_steps
        self.features = features or ['Close']
        self.epochs = epochs
        self.batch_size = batch_size
        self.horizon_steps = horizon_steps
        self.model = None
        self.scaler = StandardScaler()
        self.target_index = self.features.index('Close') if 'Close' in self.features else 0
        np.random.seed(42)
        tf.random.set_seed(42)
        self.directional_alpha = 1.0
        self.direction_model = None
        self.tabular_model = None
        self.rf_120_model = None
        self.reconstruction_model = None
        self.ensemble_weights = (0.6, 0.3, 0.1)  # hgb_delta, rf_delta, persistence
        self.direction_threshold = 0.5
        self.price_blend_weight = 1.0
        self.use_direction_adjustment = True

    def create_features(self, df, for_training=True):
        df = df.copy()
        
        if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Basic price features
        df['Return'] = df['Close'].pct_change()
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Price dynamics
        for lag in [1, 3, 5, 7, 14, 21, 30, 60, 90, 120, 180, 240]:
            df[f'Return_Lag_{lag}'] = df['Return'].shift(lag)
            df[f'Log_Return_Lag_{lag}'] = df['Log_Return'].shift(lag)
            if lag <= 7:  # Only calculate for shorter lags
                df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        
        # Volume features
        df['Volume_MA7'] = df['Volume'].rolling(7).mean()
        df['Volume_MA30'] = df['Volume'].rolling(30).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA30']
        df['Volume_Trend'] = (df['Volume_MA7'] / df['Volume_MA30']).pct_change(5)
        
        # Moving averages and volatility
        for window in [7, 14, 21, 50, 100, 200, 390]:
            df[f'MA{window}'] = df['Close'].rolling(window).mean()
            if window <= 50:  # Only calculate for shorter windows
                df[f'Volatility{window}'] = df['Return'].rolling(window).std()
        for window in [60, 120, 240]:
            df[f'Volatility{window}'] = df['Return'].rolling(window).std()
        
        # Price relative to moving averages
        df['Price_to_MA50'] = df['Close'] / df['MA50']
        df['Price_to_MA200'] = df['Close'] / df['MA200']
        
        # Moving average crossovers
        df['MA_Cross_7_21'] = (df['MA7'] > df['MA21']).astype(int)
        df['MA_Cross_21_50'] = (df['MA21'] > df['MA50']).astype(int)
        df['MA_Cross_50_200'] = (df['MA50'] > df['MA200']).astype(int)
        
        # Momentum indicators
        for period in [7, 14, 21, 30]:
            df[f'Momentum{period}'] = df['Close'].pct_change(period)
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        # Avoid division by zero
        avg_loss = avg_loss.replace(0, 1e-10)
        df['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))
        
        # RSI momentum
        df['RSI_MA7'] = df['RSI'].rolling(7).mean()
        df['RSI_Trend'] = df['RSI'] - df['RSI_MA7']
        
        # MACD (Moving Average Convergence Divergence)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        df['MACD_Hist_Change'] = df['MACD_Hist'].pct_change(3)
        
        # Bollinger Bands
        df['MA20'] = df['Close'].rolling(20).mean()
        df['BB_std'] = df['Close'].rolling(20).std()
        df['BB_upper'] = df['MA20'] + 2 * df['BB_std']
        df['BB_lower'] = df['MA20'] - 2 * df['BB_std']
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['MA20']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Price patterns
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        df['High_Close_Ratio'] = df['High'] / df['Close']
        
        # Date features (if Date column exists)
        if 'Date' in df.columns:
            df['Day_of_Week'] = df['Date'].dt.dayofweek
            df['Month'] = df['Date'].dt.month
            df['Quarter'] = df['Date'].dt.quarter
            df['Day_of_Month'] = df['Date'].dt.day
            
            # Cyclical encoding of time features
            df['Day_sin'] = np.sin(2 * np.pi * df['Day_of_Month'] / 31)
            df['Day_cos'] = np.cos(2 * np.pi * df['Day_of_Month'] / 31)
            df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
            df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        # Market regime features
        df['Trend_21_50'] = np.where(df['MA21'] > df['MA50'], 1, -1)
        df['Volatility_Regime'] = np.where(df['Volatility21'] > df['Volatility21'].rolling(50).mean(), 1, 0)
        
        if 'Sentiment_Score' not in df.columns:
            df['Sentiment_Score'] = 0
        if 'Sentiment_Magnitude' not in df.columns:
            df['Sentiment_Magnitude'] = 0
        if 'Sentiment_Volume' not in df.columns:
            df['Sentiment_Volume'] = 0
        if 'Sentiment_Trend' not in df.columns:
            df['Sentiment_Trend'] = 0
        if 'Sentiment_Volatility' not in df.columns:
            df['Sentiment_Volatility'] = 0
        
        # Target for the configured intraday horizon (1-minute bars).
        df['Next_Month_Close'] = df['Close'].shift(-self.horizon_steps)
        df['Target_Delta'] = df['Next_Month_Close'] - df['Close']
        df['Target_Return'] = df['Target_Delta'] / df['Close'].replace(0, np.nan)
        
        # Drop rows with missing values
        if not for_training:
            return df

        required_cols = list(dict.fromkeys(self.features + ['Next_Month_Close']))
        return df.dropna(subset=required_cols)

    def create_sequences(self, data):
        X, y = [], []
        max_possible = len(data) - self.time_steps - self.horizon_steps
        if max_possible <= 0:
            raise ValueError("Insufficient data for sequence creation")
        
        for i in range(max_possible):
            X.append(data[i:i+self.time_steps])
            y.append(data[i+self.time_steps+self.horizon_steps, self.target_index])
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        # Clear previous TF session to avoid memory issues
        tf.keras.backend.clear_session()
        
        # Use functional API for a hybrid CNN-LSTM model with attention
        inputs = tf.keras.Input(shape=input_shape)
        
        # CNN branch for feature extraction
        cnn = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
        cnn = MaxPooling1D(pool_size=2)(cnn)
        cnn = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(cnn)
        cnn = MaxPooling1D(pool_size=2)(cnn)
        
        # LSTM branch
        lstm = Bidirectional(LSTM(128, return_sequences=True, 
                          kernel_regularizer=l2(0.005)))(inputs)
        lstm = Dropout(0.3)(lstm)
        
        # Second Bidirectional LSTM layer
        lstm = Bidirectional(LSTM(64, return_sequences=True,
                          kernel_regularizer=l2(0.005)))(lstm)
        
        # Apply attention mechanism
        attention_layer = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=16, dropout=0.1
        )
        attention_output = attention_layer(lstm, lstm)
        attention_output = tf.keras.layers.LayerNormalization()(attention_output + lstm)
        
        # Global pooling to reduce sequence dimension
        gap_cnn = GlobalAveragePooling1D()(cnn)
        gap_lstm = GlobalAveragePooling1D()(attention_output)
        
        # Concatenate CNN and LSTM branches
        x = Concatenate()([gap_cnn, gap_lstm])
        x = Dropout(0.2)(x)
        
        # Dense layers with residual connections
        dense1 = Dense(64, activation='relu', kernel_regularizer=l2(0.005))(x)
        dense1 = Dropout(0.2)(dense1)
        dense2 = Dense(32, activation='relu', kernel_regularizer=l2(0.005))(dense1)
        
        # Output layer
        outputs = Dense(1)(dense2)
        
        # Create model
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0), loss=tf.keras.losses.Huber(delta=1.0), metrics=['mae'])

    def train(self, train_df, val_df=None):
        df_features = self.create_features(train_df)

        min_required = self.time_steps + self.horizon_steps + 1
        if len(df_features) < min_required:
            raise ValueError(f"Requires at least {min_required} data points")

        missing_features = [f for f in self.features if f not in df_features.columns]
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")

        X_tab, y_delta_tab, y_close_tab, base_tab = self._build_tabular_dataset(df_features)
        if len(y_close_tab) < 500:
            raise ValueError(f"Insufficient samples for {self.horizon_steps}-minute training")

        split_tab = int(len(X_tab) * 0.85)
        X_train_tab, X_val_tab = X_tab[:split_tab], X_tab[split_tab:]
        y_train_delta, y_val_delta = y_delta_tab[:split_tab], y_delta_tab[split_tab:]
        y_train_close, y_val_close = y_close_tab[:split_tab], y_close_tab[split_tab:]
        base_train_tab = base_tab[:split_tab]
        base_val_tab = base_tab[split_tab:]

        # Learn robust ensemble weights on rolling splits before final fit.
        self.ensemble_weights = self._fit_rolling_ensemble_weights(X_train_tab, y_train_delta, base_train_tab)

        # Main delta models
        self.tabular_model = HistGradientBoostingRegressor(
            loss='squared_error',
            learning_rate=0.03,
            max_iter=350,
            max_leaf_nodes=63,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=42,
        )
        self.rf_120_model = RandomForestRegressor(
            n_estimators=550,
            max_depth=18,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        self.tabular_model.fit(X_train_tab, y_train_delta)
        self.rf_120_model.fit(X_train_tab, y_train_delta)

        self.scaler = RobustScaler()
        self.scaler.fit(df_features[self.features].values)

        val_delta_hgb = self.tabular_model.predict(X_val_tab)
        val_delta_rf = self.rf_120_model.predict(X_val_tab)
        val_delta_ens = self._blend_delta(val_delta_hgb, val_delta_rf)
        val_pred = base_val_tab + val_delta_ens

        # Separate reconstruction model: map base + component deltas -> calibrated close.
        recon_X_train = np.column_stack([
            base_train_tab,
            self.tabular_model.predict(X_train_tab),
            self.rf_120_model.predict(X_train_tab),
        ])
        recon_y_train = y_train_close
        self.reconstruction_model = Ridge(alpha=1.0, random_state=42)
        self.reconstruction_model.fit(recon_X_train, recon_y_train)

        recon_X_val = np.column_stack([base_val_tab, val_delta_hgb, val_delta_rf])
        val_pred = self.reconstruction_model.predict(recon_X_val)

        y_dir_train = (y_train_delta > 0).astype(int)
        self.direction_model = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        self.direction_model.fit(X_train_tab, y_dir_train)

        # Holdout calibration: blend with persistence baseline.
        self.price_blend_weight = 1.0
        best_blend = 1.0
        best_rmse = np.inf
        best_obj = np.inf
        for blend in np.linspace(0.7, 1.0, 13):
            blended = blend * val_pred + (1.0 - blend) * base_val_tab
            rmse = np.sqrt(mean_squared_error(y_val_close, blended))
            dir_acc = np.mean(np.sign(blended - base_val_tab) == np.sign(y_val_close - base_val_tab))
            obj = rmse * (1.0 + 0.3 * (1.0 - dir_acc))
            if obj < best_obj:
                best_obj = obj
                best_rmse = rmse
                best_blend = float(blend)
        self.price_blend_weight = best_blend
        val_pred = self.price_blend_weight * val_pred + (1.0 - self.price_blend_weight) * base_val_tab

        # Direction threshold calibration with RMSE safety check.
        self.direction_threshold = 0.5
        self.use_direction_adjustment = False
        base_dir_acc = np.mean(np.sign(val_pred - base_val_tab) == np.sign(y_val_close - base_val_tab))
        if hasattr(self.direction_model, "predict_proba"):
            dir_proba = self.direction_model.predict_proba(X_val_tab)[:, 1]
            true_dir = (y_val_delta > 0).astype(int)
            best_thr, best_acc = 0.5, -1.0
            for thr in np.linspace(0.40, 0.65, 26):
                pred_dir = (dir_proba >= thr).astype(int)
                acc = np.mean(pred_dir == true_dir)
                if acc > best_acc:
                    best_acc = acc
                    best_thr = float(thr)
            self.direction_threshold = best_thr

            dir_up = (dir_proba >= self.direction_threshold).astype(int)
            adjusted = np.where((dir_up == 1) & (val_pred < base_val_tab), base_val_tab + np.abs(val_pred - base_val_tab), val_pred)
            adjusted = np.where((dir_up == 0) & (adjusted > base_val_tab), base_val_tab - np.abs(adjusted - base_val_tab), adjusted)
            adjusted_rmse = np.sqrt(mean_squared_error(y_val_close, adjusted))
            adjusted_dir_acc = np.mean(np.sign(adjusted - base_val_tab) == np.sign(y_val_close - base_val_tab))
            self.use_direction_adjustment = (adjusted_rmse <= (best_rmse * 0.995)) and (adjusted_dir_acc >= (base_dir_acc + 0.01))
            if self.use_direction_adjustment:
                val_pred = adjusted

        dir_acc = float(np.mean(np.sign(val_pred - base_val_tab) == np.sign(y_val_close - base_val_tab)))
        mae = float(mean_absolute_error(y_val_close, val_pred))
        rmse = float(np.sqrt(mean_squared_error(y_val_close, val_pred)))
        r2 = float(r2_score(y_val_close, val_pred))
        return {
            'model': 'delta_hgb_rf120_ensemble',
            'val_mae': mae,
            'val_rmse': rmse,
            'val_r2': r2,
            'val_directional_accuracy': dir_acc,
            'price_blend_weight': self.price_blend_weight,
            'direction_threshold': self.direction_threshold,
            'direction_adjustment_enabled': self.use_direction_adjustment,
            'ensemble_weights': {
                'lstm_tabular_delta': self.ensemble_weights[0],
                'rf120_delta': self.ensemble_weights[1],
                'persistence': self.ensemble_weights[2],
            },
        }

    def _build_tabular_dataset(self, df_features):
        feature_df = df_features[self.features].copy()
        feature_df['Target_Close'] = df_features['Close'].shift(-self.horizon_steps)
        feature_df['Target_Delta'] = feature_df['Target_Close'] - df_features['Close']
        feature_df['Base_Close'] = df_features['Close']
        feature_df = feature_df.dropna().reset_index(drop=True)
        X = feature_df[self.features].values
        y_delta = feature_df['Target_Delta'].values
        y_close = feature_df['Target_Close'].values
        base = feature_df['Base_Close'].values
        return X, y_delta, y_close, base

    def _fit_rolling_ensemble_weights(self, X_train, y_delta_train, base_train):
        """
        Learn ensemble weights on rolling validation folds:
        HGB delta model + RF delta model + persistence for the configured horizon.
        """
        if len(X_train) < 400:
            return (0.6, 0.3, 0.1)

        splitter = TimeSeriesSplit(n_splits=4)
        fold_hgb = []
        fold_rf = []
        fold_base = []
        fold_y = []

        for tr_idx, va_idx in splitter.split(X_train):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_delta_train[tr_idx], y_delta_train[va_idx]
            base_va = base_train[va_idx]

            hgb = HistGradientBoostingRegressor(
                loss='squared_error',
                learning_rate=0.03,
                max_iter=200,
                max_leaf_nodes=63,
                min_samples_leaf=20,
                l2_regularization=0.1,
                random_state=42,
            )
            rf = RandomForestRegressor(
                n_estimators=350,
                max_depth=16,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
            hgb.fit(X_tr, y_tr)
            rf.fit(X_tr, y_tr)

            fold_hgb.append(hgb.predict(X_va))
            fold_rf.append(rf.predict(X_va))
            fold_base.append(base_va)
            fold_y.append(y_va + base_va)

        pred_hgb = np.concatenate(fold_hgb)
        pred_rf = np.concatenate(fold_rf)
        base = np.concatenate(fold_base)
        y_true = np.concatenate(fold_y)

        best = (0.6, 0.3, 0.1)
        best_obj = np.inf
        for w_hgb in np.linspace(0.3, 0.9, 13):
            for w_rf in np.linspace(0.1, 0.7, 13):
                w_p = 1.0 - w_hgb - w_rf
                if w_p < 0.0 or w_p > 0.25:
                    continue
                pred_delta = (w_hgb * pred_hgb) + (w_rf * pred_rf)
                pred_close = base + pred_delta
                rmse = np.sqrt(mean_squared_error(y_true, pred_close))
                dir_acc = np.mean(np.sign(y_true - base) == np.sign(pred_close - base))
                # Reward directional stability while minimizing RMSE.
                obj = rmse * (1.0 + 0.35 * (1.0 - dir_acc))
                if obj < best_obj:
                    best_obj = obj
                    best = (float(w_hgb), float(w_rf), float(max(0.0, w_p)))
        return best

    def _blend_delta(self, delta_hgb, delta_rf):
        w_hgb, w_rf, _ = self.ensemble_weights
        return (w_hgb * delta_hgb) + (w_rf * delta_rf)

    def _sanitize_feature_frame(self, feature_frame):
        safe = feature_frame.copy()
        safe = safe.apply(pd.to_numeric, errors='coerce')
        safe = safe.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        return safe

    def _volatility_cap_pct(self, df_features):
        for col in ['Volatility120', 'Volatility60', 'Volatility21', 'Volatility14']:
            if col in df_features.columns:
                series = pd.to_numeric(df_features[col], errors='coerce')
                if series.notna().any():
                    vol = float(series.dropna().iloc[-1])
                    horizon_vol = max(vol, 0.0) * np.sqrt(max(self.horizon_steps, 1))
                    return float(np.clip(4.0 * horizon_vol * 100.0, 0.5, 12.0))
        return 6.0

    def _predict_with_tabular_model(self, df_features):
        row_df = self._sanitize_feature_frame(df_features[self.features].iloc[-1:].copy())
        row = row_df.values.astype(float)
        close_series = pd.to_numeric(df_features['Close'], errors='coerce').dropna()
        if close_series.empty:
            raise ValueError(
                f"Cannot predict {self.horizon_steps}-minute price: missing valid Close values."
            )
        last_close = float(close_series.iloc[-1])
        pred_delta_hgb = float(self.tabular_model.predict(row)[0])
        pred_delta_rf = float(self.rf_120_model.predict(row)[0]) if self.rf_120_model is not None else pred_delta_hgb
        pred_delta = float(self._blend_delta(np.array([pred_delta_hgb]), np.array([pred_delta_rf]))[0])
        pred = last_close + pred_delta
        if self.reconstruction_model is not None:
            recon_row = np.array([[last_close, pred_delta_hgb, pred_delta_rf]])
            pred = float(self.reconstruction_model.predict(recon_row)[0])
        blend = float(np.clip(getattr(self, "price_blend_weight", 1.0), 0.0, 1.0))
        pred = blend * pred + (1.0 - blend) * last_close
        cap_pct = self._volatility_cap_pct(df_features)
        lower = last_close * (1.0 - cap_pct / 100.0)
        upper = last_close * (1.0 + cap_pct / 100.0)
        pred = float(np.clip(pred, lower, upper))
        if self.direction_model is not None and getattr(self, "use_direction_adjustment", True):
            if hasattr(self.direction_model, "predict_proba"):
                dir_up = int(self.direction_model.predict_proba(row)[0, 1] >= float(getattr(self, "direction_threshold", 0.5)))
            else:
                dir_up = int(self.direction_model.predict(row)[0])
            if dir_up == 1 and pred < last_close:
                pred = last_close + abs(pred - last_close)
            elif dir_up == 0 and pred > last_close:
                pred = last_close - abs(pred - last_close)
        return pred, last_close

    def predict_tabular_batch(self, df_features):
        """Vectorized tabular predictions with holdout calibration."""
        X_df = self._sanitize_feature_frame(df_features[self.features].copy())
        X = X_df.values.astype(float)
        base = df_features['Close'].values.astype(float)
        pred_delta_hgb = self.tabular_model.predict(X).astype(float)
        pred_delta_rf = self.rf_120_model.predict(X).astype(float) if self.rf_120_model is not None else pred_delta_hgb
        pred_delta = self._blend_delta(pred_delta_hgb, pred_delta_rf)
        pred = base + pred_delta
        if self.reconstruction_model is not None:
            recon_X = np.column_stack([base, pred_delta_hgb, pred_delta_rf])
            pred = self.reconstruction_model.predict(recon_X).astype(float)
        blend = float(np.clip(getattr(self, "price_blend_weight", 1.0), 0.0, 1.0))
        pred = blend * pred + (1.0 - blend) * base
        cap_pct = self._volatility_cap_pct(df_features)
        lower = base * (1.0 - cap_pct / 100.0)
        upper = base * (1.0 + cap_pct / 100.0)
        pred = np.clip(pred, lower, upper)

        if self.direction_model is not None and getattr(self, "use_direction_adjustment", True):
            if hasattr(self.direction_model, "predict_proba"):
                dir_up = (self.direction_model.predict_proba(X)[:, 1] >= float(getattr(self, "direction_threshold", 0.5))).astype(int)
            else:
                dir_up = self.direction_model.predict(X).astype(int)
            pred = np.where((dir_up == 1) & (pred < base), base + np.abs(pred - base), pred)
            pred = np.where((dir_up == 0) & (pred > base), base - np.abs(pred - base), pred)

        return pred

    def add_sentiment_features(self, df, sentiment_features):
        """
        Add sentiment features to the dataframe
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with stock data and technical features
        sentiment_features : dict
            Dictionary containing sentiment features
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with sentiment features added
        """
        df = df.copy()
        
        # Add sentiment features to all rows (they will be the same for recent predictions)
        for feature, value in sentiment_features.items():
            # Handle the mapping from sentiment_features dict to column names
            if feature == 'sentiment_score':
                df['Sentiment_Score'] = value
            elif feature == 'sentiment_magnitude':
                df['Sentiment_Magnitude'] = value
            elif feature == 'sentiment_volume':
                df['Sentiment_Volume'] = value
            elif feature == 'sentiment_trend':
                df['Sentiment_Trend'] = value
            elif feature == 'sentiment_volatility':
                df['Sentiment_Volatility'] = value
        
        return df

    def predict(self, df, sentiment_features=None):
        """
        Make a prediction for the next month's closing price.
        Includes fallback mechanism if model is not trained or data is insufficient.
        
        Args:
            df: DataFrame with historical stock data
            sentiment_features: Dictionary containing sentiment features
            
        Returns:
            float: Predicted price for next month
            dict: Additional prediction metadata
        """
        # Force using the main prediction code
        force_main_prediction = True
        
        # Check if model exists
        if not self.model and not force_main_prediction:
            print("Warning: LSTM model not trained, using fallback prediction")
            return self._fallback_prediction(df)
        
        try:
            # Create features and validate data
            df_features = self.create_features(df, for_training=False)
            
            # Add sentiment features if provided
            if sentiment_features:
                df_features = self.add_sentiment_features(df_features, sentiment_features)
            if len(df_features) < self.time_steps and not force_main_prediction:
                print(f"Warning: Insufficient data for LSTM prediction. Need at least {self.time_steps} data points")
                return self._fallback_prediction(df)
            
            # Ensure all features exist
            missing_features = [f for f in self.features if f not in df_features.columns]
            if missing_features and not force_main_prediction:
                print(f"Warning: Missing features in data: {missing_features}")
                return self._fallback_prediction(df)
            
            # Use tabular horizon model when available
            if self.tabular_model is not None:
                predicted_price, last_close = self._predict_with_tabular_model(df_features)
                uncertainty = abs(predicted_price - last_close) * 0.25
                return {
                    'predicted_price': predicted_price,
                    'confidence_interval': (predicted_price - 1.96 * uncertainty, predicted_price + 1.96 * uncertainty),
                    'uncertainty': uncertainty,
                    'method': f'delta_ensemble_{self.horizon_steps}m'
                }

            # Fallback to legacy LSTM path
            df_features = df_features.iloc[-self.time_steps:].copy()
            
            # Check if we have enough data after feature creation and dropping NAs
            if len(df_features) < self.time_steps and not force_main_prediction:
                print(f"Warning: After feature creation, insufficient data remains. Have {len(df_features)} rows, need {self.time_steps}")
                return self._fallback_prediction(df)
            
            # Scale the features
            scaled_data = self.scaler.transform(df_features[self.features].values)
            
            # Reshape for LSTM input [samples, time steps, features]
            X_pred = np.expand_dims(scaled_data, axis=0)
            
            # Make prediction with uncertainty estimation
            predictions = []
            n_iterations = 5  # Monte Carlo iterations with dropout enabled
            
            # Enable dropout during inference for uncertainty estimation
            for _ in range(n_iterations):
                # Make prediction
                scaled_pred = self.model(X_pred, training=True)  # Enable dropout during inference
                
                # Create a dummy array to inverse transform the prediction
                dummy = np.zeros((1, len(self.features)))
                dummy[0, self.target_index] = scaled_pred[0, 0]
                
                # Inverse transform to get the actual predicted price
                predicted_price = self.scaler.inverse_transform(dummy)[0, self.target_index]
                predictions.append(predicted_price)
            
            # Calculate mean and standard deviation for uncertainty
            mean_prediction = np.mean(predictions)
            std_prediction = np.std(predictions)
            
            # Calculate confidence interval (95%)
            lower_bound = mean_prediction - 1.96 * std_prediction
            upper_bound = mean_prediction + 1.96 * std_prediction
            
            # Return prediction with metadata
            prediction_info = {
                'predicted_price': mean_prediction,
                'confidence_interval': (lower_bound, upper_bound),
                'uncertainty': std_prediction,
                'method': 'lstm_model',
                'predictions': predictions
            }
            
            return prediction_info
            
        except Exception as e:
            if force_main_prediction:
                # If forcing main prediction but it fails, create a minimal prediction
                # based on the available data without using the fallback
                try:
                    # Get the last price
                    last_price = df['Close'].iloc[-1]
                    # Use a simple trend-based prediction (last 30 days trend)
                    if len(df) >= 30:
                        trend = df['Close'].iloc[-1] / df['Close'].iloc[-30] - 1
                    else:
                        trend = 0.02  # Default 2% growth
                    
                    # Calculate predicted price
                    predicted_price = last_price * (1 + trend)
                    
                    # Add some uncertainty
                    uncertainty = predicted_price * 0.05  # 5% uncertainty
                    
                    return {
                        'predicted_price': predicted_price,
                        'confidence_interval': (predicted_price * 0.95, predicted_price * 1.05),
                        'uncertainty': uncertainty,
                        'method': 'lstm_model_minimal',
                        'error': str(e)
                    }
                except Exception as inner_e:
                    print(f"Error in minimal prediction: {str(inner_e)}")
                    # Last resort - return a simple prediction
                    return {
                        'predicted_price': 100.0,
                        'confidence_interval': (95.0, 105.0),
                        'uncertainty': 5.0,
                        'method': 'lstm_constant',
                        'error': f"{str(e)} -> {str(inner_e)}"
                    }
            else:
                print(f"Error in LSTM prediction: {str(e)}")
                return self._fallback_prediction(df)
    
    def _fallback_prediction(self, df):
        """
        Fallback prediction method when the LSTM model cannot be used.
        Uses a simple moving average and trend-based approach.
        
        Args:
            df: DataFrame with historical stock data
            
        Returns:
            dict: Prediction with metadata
        """
        try:
            # Ensure we have at least some data
            if len(df) < 30:
                raise ValueError("Insufficient data for fallback prediction")
            
            # Get the last known price
            last_price = df['Close'].iloc[-1]
            
            # Calculate simple moving averages
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            
            # Calculate recent trend (last 30 days)
            recent_trend = df['Close'].pct_change(30).iloc[-1]
            
            # Calculate volatility
            volatility = df['Close'].pct_change().std() * np.sqrt(30)  # Monthly volatility
            
            # Determine if market is trending up or down based on moving averages
            if df['MA20'].iloc[-1] > df['MA50'].iloc[-1]:
                # Uptrend - more optimistic prediction
                predicted_change = max(0.01, recent_trend)  # At least 1% increase
            else:
                # Downtrend or sideways - more conservative prediction
                predicted_change = recent_trend
            
            # Apply some dampening to avoid extreme predictions
            if abs(predicted_change) > 0.15:  # Cap at 15%
                predicted_change = 0.15 * (1 if predicted_change > 0 else -1)
            
            # Calculate predicted price
            predicted_price = last_price * (1 + predicted_change)
            
            # Calculate confidence interval based on historical volatility
            lower_bound = predicted_price * (1 - 1.96 * volatility)
            upper_bound = predicted_price * (1 + 1.96 * volatility)
            
            # Return prediction with metadata
            prediction_info = {
                'predicted_price': predicted_price,
                'confidence_interval': (lower_bound, upper_bound),
                'uncertainty': volatility * predicted_price,
                'method': 'fallback_model',
                'last_price': last_price,
                'predicted_change': predicted_change
            }
            
            return prediction_info
            
        except Exception as e:
            # Ultimate fallback - just predict a 2% increase
            last_price = df['Close'].iloc[-1] if len(df) > 0 else 100
            predicted_price = last_price * 1.02
            
            return {
                'predicted_price': predicted_price,
                'confidence_interval': (predicted_price * 0.9, predicted_price * 1.1),
                'uncertainty': predicted_price * 0.1,
                'method': 'simple_fallback',
                'last_price': last_price,
                'error': str(e)
            }

    def _augment_data(self, X, y, augmentation_factor=1.5):
        """
        Perform time series data augmentation to increase training data.
        
        Args:
            X: Input sequences [samples, time_steps, features]
            y: Target values
            augmentation_factor: How much to increase the dataset size
            
        Returns:
            Augmented X and y arrays
        """
        n_samples = len(X)
        target_samples = int(n_samples * augmentation_factor)
        additional_samples = target_samples - n_samples
        
        if additional_samples <= 0:
            return X, y
        
        X_aug = np.copy(X)
        y_aug = np.copy(y)
        
        # List to store augmented samples
        X_additional = []
        y_additional = []
        
        # 1. Jittering: Add random noise
        for i in range(min(additional_samples // 3, n_samples)):
            idx = np.random.randint(0, n_samples)
            noise = np.random.normal(0, 0.01, X[idx].shape)
            X_additional.append(X[idx] + noise)
            y_additional.append(y[idx])
        
        # 2. Scaling: Multiply by random factor close to 1
        for i in range(min(additional_samples // 3, n_samples)):
            idx = np.random.randint(0, n_samples)
            scale_factor = np.random.normal(1.0, 0.02)  # Random scaling factor around 1
            X_additional.append(X[idx] * scale_factor)
            y_additional.append(y[idx] * scale_factor)
        
        # 3. Time warping: Stretch or compress time steps slightly
        remaining = additional_samples - len(X_additional)
        for i in range(min(remaining, n_samples)):
            idx = np.random.randint(0, n_samples)
            orig_seq = X[idx]
            time_steps, features = orig_seq.shape
            
            # Create warped sequence
            warp_factor = np.random.uniform(0.9, 1.1)
            warped_seq = np.zeros_like(orig_seq)
            
            for t in range(time_steps):
                src_t = int(t * warp_factor)
                if 0 <= src_t < time_steps:
                    warped_seq[t] = orig_seq[src_t]
                else:
                    warped_seq[t] = orig_seq[t]
            
            X_additional.append(warped_seq)
            y_additional.append(y[idx])
        
        # Combine original and augmented data
        if X_additional:
            X_aug = np.vstack([X_aug, np.array(X_additional)])
            y_aug = np.concatenate([y_aug, np.array(y_additional)])
        
        # Shuffle the data
        indices = np.arange(len(X_aug))
        np.random.shuffle(indices)
        
        return X_aug[indices], y_aug[indices]
    
    def evaluate(self, test_df):
        """
        Evaluate the model on test data.
        
        Args:
            test_df: DataFrame with test data
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        # Preferred path: tabular horizon model used in app inference.
        if self.tabular_model is not None:
            df_feat = self.create_features(test_df, for_training=False)
            eval_df = df_feat.copy()
            eval_df['Target_Close'] = eval_df['Close'].shift(-self.horizon_steps)
            eval_df = eval_df.dropna().reset_index(drop=True)
            if eval_df.empty:
                raise ValueError("No evaluation samples available after feature/target alignment.")

            y_true = eval_df['Target_Close'].values.astype(float)
            y_pred = self.predict_tabular_batch(eval_df)
            base = eval_df['Close'].values.astype(float)

            mse = float(mean_squared_error(y_true, y_pred))
            rmse = float(np.sqrt(mse))
            mae = float(mean_absolute_error(y_true, y_pred))
            denom = np.maximum(np.abs(y_true), 1e-8)
            mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
            r2 = float(r2_score(y_true, y_pred))
            directional_accuracy = float(np.mean(np.sign(y_true - base) == np.sign(y_pred - base)) * 100.0)
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'r2': r2,
                'directional_accuracy': directional_accuracy
            }

        if not self.model:
            raise ValueError("Model not trained")

        df_features = self.create_features(test_df)

        # Prepare test data
        test_data = self.scaler.transform(df_features[self.features].values)
        X_test, y_test = self.create_sequences(test_data)

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Inverse transform predictions and actual values
        dummy = np.zeros((len(y_pred), len(self.features)))
        dummy[:, self.target_index] = y_pred.flatten()
        y_pred_inv = self.scaler.inverse_transform(dummy)[:, self.target_index]

        dummy = np.zeros((len(y_test), len(self.features)))
        dummy[:, self.target_index] = y_test
        y_test_inv = self.scaler.inverse_transform(dummy)[:, self.target_index]

        # Calculate metrics
        mse = float(np.mean((y_pred_inv - y_test_inv) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(y_pred_inv - y_test_inv)))
        denom = np.maximum(np.abs(y_test_inv), 1e-8)
        mape = float(np.mean(np.abs((y_test_inv - y_pred_inv) / denom)) * 100.0)
        r2 = float(r2_score(y_test_inv, y_pred_inv))

        # Calculate directional accuracy (up/down prediction)
        direction_actual = np.sign(y_test_inv[1:] - y_test_inv[:-1])
        direction_pred = np.sign(y_pred_inv[1:] - y_pred_inv[:-1])
        directional_accuracy = float(np.mean(direction_actual == direction_pred) * 100.0)

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'directional_accuracy': directional_accuracy
        }
    
    def visualize_model(self, output_path='model_visualization.png'):
        """
        Visualize the model architecture and save it to a file.
        
        Args:
            output_path: Path to save the visualization
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.model:
            print("Model not trained, cannot visualize")
            return False
        
        try:
            # Create the model plot
            plot_model(
                self.model,
                to_file=output_path,
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',  # Top to bottom layout
                expand_nested=True,
                dpi=96
            )
            return True
        except Exception as e:
            print(f"Error visualizing model: {str(e)}")
            return False
    
    def feature_importance(self, df, n_samples=1000):
        """
        Estimate feature importance using permutation importance method.
        
        Args:
            df: DataFrame with historical data
            n_samples: Number of permutation samples
            
        Returns:
            dict: Feature importance scores
        """
        if not self.model:
            print("Model not trained, cannot calculate feature importance")
            return {}
        
        try:
            # Prepare data
            df_features = self.create_features(df)
            data = self.scaler.transform(df_features[self.features].values)
            X, y = self.create_sequences(data)
            
            # Get baseline performance
            baseline_pred = self.model.predict(X)
            baseline_mse = np.mean((baseline_pred.flatten() - y) ** 2)
            
            # Calculate importance for each feature
            importance = {}
            for i, feature in enumerate(self.features):
                # Skip target feature
                if i == self.target_index:
                    continue
                    
                feature_importance = 0
                for _ in range(n_samples):
                    # Create a copy of the data
                    X_permuted = X.copy()
                    
                    # Permute the feature
                    permutation_idx = np.random.permutation(len(X))
                    X_permuted[:, :, i] = X_permuted[permutation_idx, :, i]
                    
                    # Predict with permuted feature
                    permuted_pred = self.model.predict(X_permuted)
                    permuted_mse = np.mean((permuted_pred.flatten() - y) ** 2)
                    
                    # Importance is the increase in error
                    feature_importance += (permuted_mse - baseline_mse) / baseline_mse
                
                # Average importance over samples
                importance[feature] = feature_importance / n_samples
            
            # Normalize importance scores
            max_importance = max(importance.values()) if importance else 1
            normalized_importance = {k: v / max_importance for k, v in importance.items()}
            
            # Sort by importance
            sorted_importance = dict(sorted(normalized_importance.items(), key=lambda x: x[1], reverse=True))
            return sorted_importance
            
        except Exception as e:
            print(f"Error calculating feature importance: {str(e)}")
            return {}
    
    def plot_prediction_vs_actual(self, test_df, output_path='lstm_prediction.png'):
        """
        Plot predicted vs actual values and save to file.
        
        Args:
            test_df: DataFrame with test data
            output_path: Path to save the plot
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get predictions
            df_features = self.create_features(test_df)
            test_data = self.scaler.transform(df_features[self.features].values)
            X_test, y_test = self.create_sequences(test_data)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Inverse transform
            dummy = np.zeros((len(y_pred), len(self.features)))
            dummy[:, self.target_index] = y_pred.flatten()
            y_pred_inv = self.scaler.inverse_transform(dummy)[:, self.target_index]
            
            dummy = np.zeros((len(y_test), len(self.features)))
            dummy[:, self.target_index] = y_test
            y_test_inv = self.scaler.inverse_transform(dummy)[:, self.target_index]
            
            # Create plot
            plt.figure(figsize=(12, 6))
            plt.plot(y_test_inv, label='Actual', color='blue', alpha=0.7)
            plt.plot(y_pred_inv, label='Predicted', color='red', alpha=0.7)
            plt.fill_between(
                range(len(y_pred_inv)),
                y_pred_inv * 0.9,  # Lower bound
                y_pred_inv * 1.1,  # Upper bound
                color='red',
                alpha=0.2,
                label='Prediction Interval'
            )
            plt.title('LSTM Model: Predicted vs Actual Stock Prices')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            return True
        except Exception as e:
            print(f"Error plotting predictions: {str(e)}")
            return False    

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Change file extension from .h5 to .keras
        keras_path = path.replace('.h5', '.keras')
        
        # Make sure the directory exists
        os.makedirs(os.path.dirname(keras_path), exist_ok=True)
        
        # Save the deep model only when it exists; tabular-only training is supported.
        if self.model is not None:
            try:
                self.model.save(keras_path)
                print(f"Model saved successfully to {keras_path}")
            except Exception as e:
                print(f"Error saving model: {str(e)}")
                # Try alternative saving method
                try:
                    self.model.save_weights(keras_path + '_weights')
                    print(f"Model weights saved to {keras_path}_weights")
                except Exception as e2:
                    print(f"Error saving model weights: {str(e2)}")
        else:
            print("No deep model instance to save; saving tabular metadata only.")
        
        # Save metadata
        try:
            metadata = {
                'scaler': self.scaler,
                'features': self.features,
                'config': {
                    'time_steps': self.time_steps,
                    'horizon_steps': self.horizon_steps,
                    'target_index': self.target_index
                },
                'tabular_model': self.tabular_model,
                'rf_120_model': self.rf_120_model,
                'reconstruction_model': self.reconstruction_model,
                'direction_model': self.direction_model,
                'ensemble_weights': self.ensemble_weights,
                'direction_threshold': self.direction_threshold,
                'price_blend_weight': self.price_blend_weight,
                'use_direction_adjustment': self.use_direction_adjustment,
            }
            joblib.dump(metadata, keras_path + '_meta.pkl')
            print(f"Model metadata saved to {keras_path}_meta.pkl")
        except Exception as e:
            print(f"Error saving metadata: {str(e)}")
        
        return keras_path

    def predict_next_month(self, df):
        """
        Specifically predict the closing price 30 days in the future.
        
        Args:
            df: DataFrame with historical stock data
            
        Returns:
            float: Predicted closing price 30 days in the future
        """
        current_price = df['Close'].iloc[-1]
        prediction_info = self.predict(df)
        
        # Extract predicted price from dictionary or use directly if it's a float
        if isinstance(prediction_info, dict):
            predicted_price = prediction_info['predicted_price']
            method = prediction_info.get('method', 'unknown')
            print(f"Prediction method: {method}")
            
            # If confidence interval is available, print it
            if 'confidence_interval' in prediction_info:
                lower, upper = prediction_info['confidence_interval']
                print(f"95% Confidence interval: ${lower:.2f} to ${upper:.2f}")
            
            # If uncertainty is available, print it
            if 'uncertainty' in prediction_info:
                print(f"Uncertainty: ±${prediction_info['uncertainty']:.2f}")
        else:
            predicted_price = prediction_info
        
        print(f"Current price: ${current_price:.2f}")
        print(f"Predicted price in 30 days: ${predicted_price:.2f}")
        print(f"Predicted change: {((predicted_price/current_price)-1)*100:.2f}%")
        
        return predicted_price

    @classmethod
    def load(cls, path):
        # Handle both .h5 and .keras extensions
        if path.endswith('.h5'):
            keras_path = path.replace('.h5', '.keras')
        else:
            keras_path = path
            
        meta_path = keras_path + '_meta.pkl'
        has_model_file = os.path.exists(keras_path) or os.path.exists(keras_path + '_weights.index')
        has_metadata = os.path.exists(meta_path)
        if not has_model_file and not has_metadata:
            print(f"Warning: Neither model artifact nor metadata found for {keras_path}")
            return cls()
        
        try:
            metadata = None
            if has_metadata:
                metadata = joblib.load(meta_path)

            if metadata and 'config' in metadata:
                cfg = metadata['config']
                model = cls(
                    time_steps=cfg.get('time_steps', 60),
                    features=metadata.get('features'),
                    horizon_steps=cfg.get('horizon_steps', 120),
                )
            else:
                model = cls()

            # Load deep model if available (optional for tabular-only training)
            if os.path.exists(keras_path):
                try:
                    model.model = tf.keras.models.load_model(keras_path)
                    print(f"Model loaded successfully from {keras_path}")
                except Exception as model_load_error:
                    model.model = None
                    print(f"Warning: Could not load deep model at {keras_path}: {model_load_error}")
            elif os.path.exists(keras_path + '_weights.index'):
                # If only weights exist, create model architecture and load weights
                # This requires building the model first
                try:
                    weights_features = model.features if model.features else ['Close', 'Volume', 'High', 'Low', 'Open']
                    model.features = weights_features
                    dummy_data = np.random.random((1, model.time_steps, len(weights_features)))
                    model.build_model((model.time_steps, len(weights_features)))
                    model.model.predict(dummy_data)  # Initialize weights
                    model.model.load_weights(keras_path + '_weights')
                    print(f"Model weights loaded from {keras_path}_weights")
                except Exception as weights_load_error:
                    model.model = None
                    print(f"Warning: Could not load weights at {keras_path}_weights: {weights_load_error}")
            
            # Load metadata if available
            if metadata:
                model.scaler = metadata.get('scaler')
                model.features = metadata.get('features', model.features)
                config = metadata.get('config', {})
                model.time_steps = config.get('time_steps', model.time_steps)
                model.horizon_steps = config.get('horizon_steps', model.horizon_steps)
                model.target_index = config.get('target_index', model.target_index)
                model.tabular_model = metadata.get('tabular_model')
                model.rf_120_model = metadata.get('rf_120_model')
                model.reconstruction_model = metadata.get('reconstruction_model')
                model.direction_model = metadata.get('direction_model')
                model.ensemble_weights = metadata.get('ensemble_weights', model.ensemble_weights)
                model.direction_threshold = metadata.get('direction_threshold', model.direction_threshold)
                model.price_blend_weight = metadata.get('price_blend_weight', model.price_blend_weight)
                model.use_direction_adjustment = metadata.get('use_direction_adjustment', model.use_direction_adjustment)
                print(f"Metadata loaded from {meta_path}")
            else:
                print(f"Warning: Metadata file {meta_path} not found.")
            
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            # Return an untrained model instance
            model = cls()
            return model

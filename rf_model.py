import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import joblib
import os

class EnhancedRandomForestModel:
    def __init__(self, feature_selection_threshold=0.01, random_state=42, horizon_steps=5):
        """
        Improved Random Forest model with feature selection and hyperparameter tuning
        
        Parameters:
        -----------
        feature_selection_threshold : float
            Threshold for feature selection based on importance
        random_state : int
            Random seed for reproducibility
        """
        self.feature_selection_threshold = feature_selection_threshold
        self.random_state = random_state
        self.horizon_steps = horizon_steps
        self.pipeline = None
        self.feature_columns = None
        self.selected_features = None
        self.best_params_ = None
        self.directional_bias = 0.0
        self.direction_model = None
        self.direction_threshold = 0.5
        self.price_blend_weight = 1.0
        self.use_direction_adjustment = True
        self.holdout_r2_ = 0.0

    def create_features(self, df):
        """
        Enhanced feature engineering with time-series characteristics
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Dataframe with stock data
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with enhanced features
        """
        df = df.copy()
        
        if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        
        df = df.sort_values('Date')
        
        # Initialize all features with zeros
        initial_features = {
            'Return': 0,
            'Volume_Change': 0,
            'Volume_MA5': 0,
            'Volume_MA20': 0,
            'Volume_Ratio': 0,
            'RSI': 50,  # RSI starts at neutral 50
            'MACD': 0,
            'MACD_Signal': 0,
            'MACD_Hist': 0,
            'BB_middle': df['Close'].iloc[0],
            'BB_std': 0,
            'BB_upper': df['Close'].iloc[0],
            'BB_lower': df['Close'].iloc[0],
            'BB_width': 0,
            'BB_position': 0.5,
            'High_Low_Spread': 0,
            'Close_Open_Spread': 0,
            'Close_to_High': 0.5,
            'Day_of_Week': 0,
            'Month': 1,
            'Quarter': 1,
            'Year': df['Date'].iloc[0].year,
            'Day_of_Month': 1,
            'Week_of_Year': 1,
            'Trend_20_50': 0,
            'Volatility_Regime': 0,
            'Sentiment_Score': 0,
            'Sentiment_Magnitude': 0,
            'Sentiment_Volume': 0,
            'Sentiment_Trend': 0,
            'Sentiment_Volatility': 0
        }
        
        # Add lag features
        for lag in [1, 2, 3, 5, 8, 13, 21]:
            initial_features[f'Return_Lag_{lag}'] = 0
            initial_features[f'Close_Lag_{lag}'] = df['Close'].iloc[0]
        
        # Add moving average features
        for window in [5, 10, 20, 50, 100, 200]:
            initial_features[f'MA{window}'] = df['Close'].iloc[0]
            if window <= 50:
                initial_features[f'Volatility{window}'] = 0
        
        # Add price to MA ratios
        initial_features['Price_to_MA50'] = 1
        initial_features['Price_to_MA200'] = 1
        
        # Add moving average crossovers
        for cross in ['MA_Cross_5_20', 'MA_Cross_20_50', 'MA_Cross_50_200']:
            initial_features[cross] = 0
        
        # Add momentum features
        for period in [5, 10, 20, 30]:
            initial_features[f'Momentum{period}'] = 0
        
        # Create features
        df['Return'] = df['Close'].pct_change()
        for lag in [1, 2, 3, 5, 8, 13, 21]:
            df[f'Return_Lag_{lag}'] = df['Return'].shift(lag)
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        
        # Volume Features
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA5'] = df['Volume'].rolling(5).mean()
        df['Volume_MA20'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
        
        # Technical Indicators
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'MA{window}'] = df['Close'].rolling(window).mean()
            if window <= 50:
                df[f'Volatility{window}'] = df['Return'].rolling(window).std()
        
        # Price relative to moving averages
        df['Price_to_MA50'] = df['Close'] / df['MA50']
        df['Price_to_MA200'] = df['Close'] / df['MA200']
        
        # Moving Average Crossovers
        df['MA_Cross_5_20'] = (df['MA5'] > df['MA20']).astype(int)
        df['MA_Cross_20_50'] = (df['MA20'] > df['MA50']).astype(int)
        df['MA_Cross_50_200'] = (df['MA50'] > df['MA200']).astype(int)
        
        # Momentum Indicators
        for period in [5, 10, 20, 30]:
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
        
        # MACD (Moving Average Convergence Divergence)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(20).mean()
        df['BB_std'] = df['Close'].rolling(20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Price Relationships
        df['High_Low_Spread'] = (df['High'] - df['Low']) / df['Low']
        df['Close_Open_Spread'] = (df['Close'] - df['Open']) / df['Open']
        df['Close_to_High'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Date Features
        df['Day_of_Week'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['Year'] = df['Date'].dt.year
        df['Day_of_Month'] = df['Date'].dt.day
        df['Week_of_Year'] = df['Date'].dt.isocalendar().week
        
        # Market Regime Features
        df['Trend_20_50'] = np.where(df['MA20'] > df['MA50'], 1, -1)
        df['Volatility_Regime'] = np.where(df['Volatility20'] > df['Volatility20'].rolling(50).mean(), 1, 0)

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
        
        # Target Variable (for intraday, horizon_steps=5 means +5 minutes with 1-minute bars)
        df['Next_Day_Close'] = df['Close'].shift(-self.horizon_steps)
        
        # Fill initial NaN values with reasonable defaults
        df.fillna(initial_features, inplace=True)
        
        # Drop initial rows with missing values
        df = df.dropna().reset_index(drop=True)
        
        return df
        
    def use_pretrained_pipeline(self, best_params):
        """Reuse parameters from initial training"""
        self.pipeline = Pipeline([
            ('feature_selection', SelectFromModel(
                RandomForestRegressor(n_estimators=100, random_state=self.random_state),
                threshold=self.feature_selection_threshold)),
            ('model', RandomForestRegressor(**best_params))
        ])

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

    def prepare_data(self, df):
        """Prepare feature matrix and target vector"""
        # Create target variable
        df['Next_Day_Close'] = df['Close'].shift(-self.horizon_steps)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Separate features and target
        X = df.drop(['Date', 'Close', 'Next_Day_Close'], axis=1)
        y = df['Next_Day_Close']
        base_close = df['Close']
        
        return X.values, y.values, base_close.values

    def _select_features(self, X, y):
        """
        Select important features using a Random Forest model
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            list: Indices of selected features
        """
        # Train a simple RF model to get feature importances
        selector = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        selector.fit(X, y)
        
        # Get feature importances
        importances = selector.feature_importances_
        
        # Select features above threshold
        selected_indices = np.where(importances > self.feature_selection_threshold)[0]
        
        # If no features selected, take top 5
        if len(selected_indices) == 0:
            selected_indices = np.argsort(importances)[-5:]  # Take top 5 features
        
        return selected_indices.tolist()

    def _align_feature_frame(self, feature_frame):
        """Align incoming feature frame to training-time column order and selection."""
        aligned = feature_frame.reindex(columns=self.feature_columns, fill_value=0)
        if hasattr(self, 'selected_features') and self.selected_features is not None:
            aligned = aligned[self.selected_feature_names]
        aligned = aligned.apply(pd.to_numeric, errors='coerce')
        aligned = aligned.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        return aligned

    def _apply_prediction_calibration(self, X, y_pred, base_prices, apply_direction=True):
        """Apply holdout-calibrated blending and optional direction consistency."""
        pred = np.asarray(y_pred, dtype=float)
        base = np.asarray(base_prices, dtype=float)

        blend = float(np.clip(getattr(self, "price_blend_weight", 1.0), 0.0, 1.0))
        pred = blend * pred + (1.0 - blend) * base
        pred = pred * (1.0 + float(getattr(self, "directional_bias", 0.0)))

        if (
            apply_direction
            and getattr(self, "use_direction_adjustment", True)
            and self.direction_model is not None
            and len(pred) > 0
        ):
            if hasattr(self.direction_model, "predict_proba"):
                dir_up = (
                    self.direction_model.predict_proba(X)[:, 1] >= float(getattr(self, "direction_threshold", 0.5))
                ).astype(int)
            else:
                dir_up = self.direction_model.predict(X).astype(int)

            pred = np.where((dir_up == 1) & (pred < base), base + np.abs(pred - base), pred)
            pred = np.where((dir_up == 0) & (pred > base), base - np.abs(pred - base), pred)
        return pred

    def predict_from_features(self, feature_frame, base_prices, apply_direction=True):
        """Predict from a feature frame already engineered via create_features."""
        X_df = self._align_feature_frame(feature_frame)
        X = X_df.values.astype(float)
        raw_pred = self.pipeline.predict(X)
        return self._apply_prediction_calibration(X, raw_pred, base_prices, apply_direction=apply_direction)

    def predict_future(self, df, window_size=200, sentiment_features=None):
        """Predict the future closing price at the configured horizon
        
        Args:
            df: DataFrame with historical data
            window_size: Number of days to use for prediction
            sentiment_features: Dictionary containing sentiment features
            
        Returns:
            float: Predicted closing price
        """
        # Ensure we have enough historical data
        if len(df) < window_size:
            raise ValueError(f"Insufficient historical data for prediction. Need at least {window_size} days")

        # Create features
        df_engineered = self.create_features(df)
        
        # Add sentiment features if provided
        if sentiment_features:
            df_engineered = self.add_sentiment_features(df_engineered, sentiment_features)

        # Drop unnecessary columns
        columns_to_drop = ['Date', 'Next_Day_Close', 'Next_Month_Close']
        
        # Get the last row for prediction
        last_row = df_engineered.iloc[-1:].copy()
        
        # Handle NaN values by filling with forward fill, then backward fill
        last_row = last_row.ffill().bfill()
        
        # Drop unnecessary columns
        last_row = last_row.drop(columns_to_drop, axis=1, errors='ignore')
        
        # Get feature columns from the original training data
        feature_columns = self.feature_columns
        
        # Ensure we have all required feature columns
        missing_columns = set(feature_columns) - set(last_row.columns)
        if missing_columns:
            # Add missing columns with default values (0)
            for col in missing_columns:
                last_row[col] = 0

        # Reorder/align features and extract matrix
        last_row = self._align_feature_frame(last_row)
        X = last_row.values

        if X.size == 0:
            raise ValueError("No valid features found for prediction")

        last_close = float(df_engineered["Close"].iloc[-1])
        pred = float(
            self._apply_prediction_calibration(
                X,
                self.pipeline.predict(X),
                np.array([last_close]),
                apply_direction=True,
            )[0]
        )
        return pred

    def predict_next_5min(self, df, window_size=200, sentiment_features=None):
        """Convenience method: predict 5 minutes ahead (requires 1-minute bars)."""
        return self.predict_future(df, window_size=window_size, sentiment_features=sentiment_features)

    def predict_next_day(self, df, window_size=200, sentiment_features=None):
        """Backward-compatible alias (now uses configured horizon_steps)."""
        return self.predict_future(df, window_size=window_size, sentiment_features=sentiment_features)

    def predict_next_30_days(self, df):
        """Predict closing prices for the next 30 trading days

        Args:
            df: DataFrame with historical data

        Returns:
            DataFrame: Predictions for the next 30 days
        """
        if len(df) < 200:  # Ensure sufficient history
            raise ValueError("Insufficient historical data for prediction")

        # Create a copy of the dataframe to avoid modifying the original
        df_copy = df.copy()

        # Get the last date in the dataframe
        last_date = df_copy['Date'].iloc[-1]

        # Create a dataframe to store predictions
        predictions = []
        dates = []

        # Predict for the next 30 days
        for i in range(1, 31):
            # Create features for the current state
            df_engineered = self.create_features(df_copy)

            # Get the last row for prediction
            last_data = df_engineered.iloc[-1:].drop(['Date', 'Next_Day_Close'], axis=1, errors='ignore')

            # Extract features in the correct format
            X = last_data.values

            # If we have selected features, use only those
            if hasattr(self, 'selected_features') and self.selected_features is not None:
                X = X[:, self.selected_features]

            # Make prediction
            prediction = self.pipeline.predict(X)[0] * (1.0 + self.directional_bias)
            predictions.append(prediction)

            # Update the dataframe with the new prediction
            next_date = last_date + pd.Timedelta(days=1)
            dates.append(next_date)
            df_copy = pd.concat([df_copy, pd.DataFrame({
                'Date': [next_date],
                'Close': [prediction],
                'Volume': [df_copy['Volume'].iloc[-1]],  # Use last volume as placeholder
                'Open': [df_copy['Close'].iloc[-1]],    # Use last close as placeholder
                'High': [df_copy['Close'].iloc[-1]],    # Use last close as placeholder
                'Low': [df_copy['Close'].iloc[-1]]      # Use last close as placeholder
            })], ignore_index=True)
            last_date = next_date

        # Create a DataFrame for the predictions
        prediction_df = pd.DataFrame({
            'Date': dates,
            'Predicted_Close': predictions
        })

        return prediction_df

    def _tune_hyperparameters(self, X, y, cv=5):
        """Pick RF params with a time-aware holdout to protect 30-min R²."""
        split = int(len(X) * 0.8)
        if split < 100:
            best_params = {
                'n_estimators': 400,
                'max_depth': 20,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 0.5,
                'bootstrap': True,
            }
            print(f"Using default RF parameters: {best_params}")
            return best_params

        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        candidates = [
            {'n_estimators': 500, 'max_depth': 20, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 0.5, 'bootstrap': True},
            {'n_estimators': 400, 'max_depth': None, 'min_samples_split': 3, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'bootstrap': True},
            {'n_estimators': 600, 'max_depth': 28, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 0.65, 'bootstrap': True},
        ]

        best_params, best_score = candidates[0], -np.inf
        for params in candidates:
            model = RandomForestRegressor(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                max_features=params['max_features'],
                random_state=self.random_state,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            score = r2_score(y_val, model.predict(X_val))
            if score > best_score:
                best_score = score
                best_params = params

        self.holdout_r2_ = float(best_score)
        print(f"Selected RF parameters with holdout R2={best_score:.4f}: {best_params}")
        return best_params

    def _create_pipeline(self, params):
        """
        Create a pipeline with the given parameters
        
        Args:
            params: Dictionary of parameters for the RandomForestRegressor
            
        Returns:
            Pipeline: Configured pipeline
        """
        return RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', None),
            min_samples_split=params.get('min_samples_split', 2),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            max_features=params.get('max_features', 'sqrt'),
            random_state=self.random_state,
            n_jobs=-1
        )
        
    def train(self, df, cv=5):
        """
        Train the model with feature selection and hyperparameter tuning

        Args:
            df: DataFrame with historical data
            cv: Number of cross-validation folds

        Returns:
            dict: Training metrics and selected features
        """
        # Create features
        df_features = self.create_features(df)

        # Prepare data
        X, y, base_close = self.prepare_data(df_features)

        # Set feature columns
        self.feature_columns = df_features.drop(['Date', 'Close', 'Next_Month_Close', 'Next_Day_Close'], axis=1, errors='ignore').columns.tolist()

        # Feature selection
        if self.feature_selection_threshold > 0:
            # Create a selector model
            selector = RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)
            selector.fit(X, y)
            importances = selector.feature_importances_

            # Get indices of selected features
            self.selected_features = np.where(importances > self.feature_selection_threshold)[0].tolist()

            # Get the names of selected features
            self.selected_feature_names = [self.feature_columns[i] for i in self.selected_features]

            # Select features using indices
            X_selected = X[:, self.selected_features]
            print(f"Selected {len(self.selected_features)} out of {X.shape[1]} features")
        else:
            self.selected_features = list(range(X.shape[1]))
            self.selected_feature_names = self.feature_columns
            X_selected = X

        # Hyperparameter tuning
        best_params = self._tune_hyperparameters(X_selected, y, cv)

        # Train final model with best parameters
        self.pipeline = self._create_pipeline(best_params)
        print("Training final model with best parameters...")
        self.pipeline.fit(X_selected, y)

        # Train a dedicated direction classifier (up/down) for horizon movement
        y_dir = (y > base_close).astype(int)
        self.direction_model = ExtraTreesClassifier(
            n_estimators=120,
            max_depth=14,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.direction_model.fit(X_selected, y_dir)

        # Holdout calibration on tail split.
        calib_start = int(len(y) * 0.8)
        self.direction_threshold = 0.5
        self.price_blend_weight = 1.0
        self.use_direction_adjustment = True
        if len(y) - calib_start > 20:
            calib_X = X_selected[calib_start:]
            calib_pred = self.pipeline.predict(calib_X)
            calib_true = y[calib_start:]
            calib_base = base_close[calib_start:]
            self.directional_bias = 0.0

            # Blend with persistence baseline to stabilize horizon regression.
            if self.holdout_r2_ < 0:
                # In unstable holdout regimes, a partial persistence anchor is more robust.
                self.price_blend_weight = 0.3
                best_rmse = np.sqrt(mean_squared_error(
                    calib_true,
                    (self.price_blend_weight * calib_pred) + ((1.0 - self.price_blend_weight) * calib_base),
                ))
            else:
                best_blend = 1.0
                best_rmse = np.inf
                best_obj = np.inf
                for blend in np.linspace(0.0, 1.0, 21):
                    blended = blend * calib_pred + (1.0 - blend) * calib_base
                    rmse = np.sqrt(mean_squared_error(calib_true, blended))
                    dir_acc = np.mean(np.sign(blended - calib_base) == np.sign(calib_true - calib_base))
                    obj = rmse * (1.0 + 0.3 * (1.0 - dir_acc))
                    if obj < best_obj:
                        best_obj = obj
                        best_rmse = rmse
                        best_blend = float(blend)
                self.price_blend_weight = best_blend

            blended_calib = self.price_blend_weight * calib_pred + (1.0 - self.price_blend_weight) * calib_base

            if self.use_direction_adjustment:
                dir_proba = self.direction_model.predict_proba(calib_X)[:, 1]
                calib_true_dir = (calib_true > calib_base).astype(int)
                best_thr, best_acc = 0.5, -1.0
                for thr in np.linspace(0.45, 0.60, 16):
                    pred_dir = (dir_proba >= thr).astype(int)
                    acc = np.mean(pred_dir == calib_true_dir)
                    if acc > best_acc:
                        best_acc = acc
                        best_thr = float(thr)
                self.direction_threshold = best_thr

                # Keep direction adjustment only if it does not meaningfully hurt RMSE.
                dir_up = (dir_proba >= self.direction_threshold).astype(int)
                adjusted = np.where((dir_up == 1) & (blended_calib < calib_base), calib_base + np.abs(blended_calib - calib_base), blended_calib)
                adjusted = np.where((dir_up == 0) & (adjusted > calib_base), calib_base - np.abs(adjusted - calib_base), adjusted)
                adjusted_rmse = np.sqrt(mean_squared_error(calib_true, adjusted))
                self.use_direction_adjustment = adjusted_rmse <= (best_rmse * 1.01)
        else:
            self.directional_bias = 0.0
            self.price_blend_weight = 1.0
            self.use_direction_adjustment = True

        # In-sample diagnostics
        y_pred = self._apply_prediction_calibration(X_selected, self.pipeline.predict(X_selected), base_close, apply_direction=False)
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        mae = float(mean_absolute_error(y, y_pred))
        r2 = float(r2_score(y, y_pred))

        # Calculate feature importances
        importances = self.pipeline.feature_importances_
        indices = np.argsort(importances)[::-1]

        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'best_params': best_params,
            'selected_features': self.selected_feature_names,
            'feature_importances': {
                'features': [self.selected_feature_names[i] for i in indices],
                'importance': importances[indices].tolist()
            },
            'training_score': float(self.pipeline.score(X_selected, y))
        }

        return metrics

    def save_model(self, filepath):
        """Save trained model to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
    
    @classmethod
    def load_model(cls, filepath):
        """Load trained model from disk"""
        return joblib.load(filepath)

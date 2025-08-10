"""
Sentiment Predictor Module
ML pipeline for predicting asset returns based on sentiment data.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class SentimentPredictor:
    """
    Machine learning pipeline for sentiment-driven return prediction.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize SentimentPredictor.
        
        Args:
            config: Configuration dictionary from data ingestion
        """
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.trained_assets = []
        
    def create_features(self, 
                       prices: pd.DataFrame, 
                       sentiment: pd.DataFrame, 
                       returns: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for ML model training.
        
        Args:
            prices: Price data
            sentiment: Sentiment data
            returns: Returns data
            
        Returns:
            DataFrame with engineered features
        """
        features_list = []
        
        # Get configuration
        sentiment_lags = self.config.get('feature_engineering', {}).get('sentiment_lags', [1, 2, 3, 5])
        price_features = self.config.get('feature_engineering', {}).get('price_features', [])
        
        for asset in prices.columns:
            asset_features = pd.DataFrame(index=prices.index)
            
            # Current and lagged sentiment features
            if f'{asset}_sentiment' in sentiment.columns:
                # Asset-specific sentiment
                asset_features[f'{asset}_sentiment'] = sentiment[f'{asset}_sentiment']
                for lag in sentiment_lags:
                    asset_features[f'{asset}_sentiment_lag_{lag}'] = sentiment[f'{asset}_sentiment'].shift(lag)
            
            # Market sentiment
            if 'market_sentiment' in sentiment.columns:
                asset_features['market_sentiment'] = sentiment['market_sentiment']
                for lag in sentiment_lags:
                    asset_features[f'market_sentiment_lag_{lag}'] = sentiment['market_sentiment'].shift(lag)
            
            # Price-based features
            if 'returns_1d' in price_features:
                asset_features[f'{asset}_returns_1d'] = returns[asset].shift(1)
            
            if 'returns_5d' in price_features:
                asset_features[f'{asset}_returns_5d'] = returns[asset].rolling(5).mean().shift(1)
            
            if 'volatility_10d' in price_features:
                asset_features[f'{asset}_volatility_10d'] = returns[asset].rolling(10).std().shift(1)
            
            if 'rsi_14d' in price_features:
                asset_features[f'{asset}_rsi_14d'] = self.calculate_rsi(prices[asset], 14).shift(1)
            
            # Target variable (next-day return)
            asset_features[f'{asset}_target'] = returns[asset].shift(-1)
            
            # Add asset identifier
            asset_features['asset'] = asset
            
            features_list.append(asset_features)
        
        # Combine all asset features
        all_features = pd.concat(features_list, axis=0, ignore_index=False)
        all_features = all_features.dropna()
        
        return all_features
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Price series
            window: RSI window period
            
        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def prepare_training_data(self, features_df: pd.DataFrame) -> Dict[str, Tuple]:
        """
        Prepare training data for each asset.
        
        Args:
            features_df: DataFrame with engineered features
            
        Returns:
            Dictionary mapping asset to (X, y) tuples
        """
        training_data = {}
        
        for asset in features_df['asset'].unique():
            asset_data = features_df[features_df['asset'] == asset].copy()
            
            # Features (exclude target and asset identifier)
            feature_cols = [col for col in asset_data.columns 
                           if not col.endswith('_target') and col != 'asset']
            X = asset_data[feature_cols]
            
            # Target
            y = asset_data[f'{asset}_target']
            
            # Remove any remaining NaN values
            valid_idx = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_idx]
            y = y[valid_idx]
            
            if len(X) > 50:  # Minimum samples for training
                training_data[asset] = (X, y)
                
        return training_data
    
    def train_models(self, training_data: Dict[str, Tuple]) -> Dict:
        """
        Train ML models for each asset.
        
        Args:
            training_data: Dictionary of training data for each asset
            
        Returns:
            Dictionary with training results and metrics
        """
        model_params = self.config.get('model_params', {}).get('lightgbm', {})
        results = {}
        
        for asset, (X, y) in training_data.items():
            print(f"Training model for {asset}...")
            
            # Store feature names
            if not self.feature_names:
                self.feature_names = list(X.columns)
            
            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Train LightGBM model
                train_data = lgb.Dataset(X_train_scaled, label=y_train)
                val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)
                
                model = lgb.train(
                    model_params,
                    train_data,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
                )
                
                # Validate
                y_pred = model.predict(X_val_scaled)
                score = mean_squared_error(y_val, y_pred)
                cv_scores.append(score)
            
            # Train final model on all data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            train_data = lgb.Dataset(X_scaled, label=y)
            final_model = lgb.train(
                model_params,
                train_data,
                callbacks=[lgb.log_evaluation(0)]
            )
            
            # Store model and scaler
            self.models[asset] = final_model
            self.scalers[asset] = scaler
            self.trained_assets.append(asset)
            
            # Calculate final metrics
            y_pred_final = final_model.predict(X_scaled)
            final_metrics = {
                'mse': mean_squared_error(y, y_pred_final),
                'mae': mean_absolute_error(y, y_pred_final),
                'r2': r2_score(y, y_pred_final),
                'cv_mse_mean': np.mean(cv_scores),
                'cv_mse_std': np.std(cv_scores),
                'num_samples': len(X)
            }
            
            results[asset] = final_metrics
            print(f"  MSE: {final_metrics['mse']:.6f}, MAE: {final_metrics['mae']:.6f}, R²: {final_metrics['r2']:.4f}")
        
        return results
    
    def predict_returns(self, sentiment_features: Dict[str, float]) -> Dict[str, Dict]:
        """
        Predict returns for all assets based on current sentiment.
        
        Args:
            sentiment_features: Dictionary with current sentiment values
            
        Returns:
            Dictionary with predicted returns and uncertainty for each asset
        """
        if not self.models:
            raise ValueError("Models not trained yet")
        
        predictions = {}
        
        for asset in self.trained_assets:
            # Create feature vector for prediction
            feature_vector = []
            
            for feature_name in self.feature_names:
                if feature_name in sentiment_features:
                    feature_vector.append(sentiment_features[feature_name])
                else:
                    # Use default values for missing features
                    feature_vector.append(0.0)
            
            # Scale features
            feature_vector = np.array(feature_vector).reshape(1, -1)
            feature_vector_scaled = self.scalers[asset].transform(feature_vector)
            
            # Make prediction
            pred_mean = self.models[asset].predict(feature_vector_scaled)[0]
            
            # Estimate uncertainty (simplified approach)
            # In practice, you might use quantile regression or ensemble methods
            pred_std = abs(pred_mean) * 0.3  # Simple heuristic
            
            predictions[asset] = {
                'predicted_return': pred_mean,
                'prediction_std': pred_std,
                'confidence_interval_95': [
                    pred_mean - 1.96 * pred_std,
                    pred_mean + 1.96 * pred_std
                ]
            }
        
        return predictions
    
    def get_feature_importance(self, asset: str, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance for a specific asset.
        
        Args:
            asset: Asset symbol
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if asset not in self.models:
            raise ValueError(f"No model found for asset {asset}")
        
        model = self.models[asset]
        importance = model.feature_importance()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def model_summary(self) -> Dict:
        """
        Get summary of trained models.
        
        Returns:
            Dictionary with model summary
        """
        return {
            'num_models': len(self.models),
            'trained_assets': self.trained_assets,
            'num_features': len(self.feature_names),
            'feature_names': self.feature_names
        }
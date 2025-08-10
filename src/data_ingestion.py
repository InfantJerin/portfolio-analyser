"""
Data Ingestion Module
Handles loading and preprocessing of portfolio, price, and sentiment data
with PyPortfolioOpt integration.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DataIngestion:
    """
    Data ingestion and preprocessing for portfolio prediction system.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize DataIngestion with configuration.
        
        Args:
            config_path: Path to configuration JSON file
        """
        self.config = self.load_config(config_path)
        self.portfolio_data = None
        self.price_data = None
        self.sentiment_data = None
        self.current_state = None
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def load_portfolio(self, portfolio_path: str) -> pd.DataFrame:
        """
        Load portfolio configuration.
        
        Args:
            portfolio_path: Path to portfolio CSV file
            
        Returns:
            DataFrame with portfolio data
        """
        self.portfolio_data = pd.read_csv(portfolio_path)
        
        # Validate portfolio weights sum to 1
        weight_sum = self.portfolio_data['weight'].sum()
        if abs(weight_sum - 1.0) > 0.001:
            print(f"Warning: Portfolio weights sum to {weight_sum:.4f}, not 1.0")
            
        return self.portfolio_data
    
    def load_prices(self, prices_path: str) -> pd.DataFrame:
        """
        Load historical price data in PyPortfolioOpt compatible format.
        
        Args:
            prices_path: Path to prices CSV file
            
        Returns:
            DataFrame with dates as index and tickers as columns
        """
        self.price_data = pd.read_csv(prices_path)
        self.price_data['date'] = pd.to_datetime(self.price_data['date'])
        self.price_data.set_index('date', inplace=True)
        
        # Ensure all portfolio tickers are present
        if self.portfolio_data is not None:
            portfolio_tickers = set(self.portfolio_data['ticker'])
            price_tickers = set(self.price_data.columns)
            missing_tickers = portfolio_tickers - price_tickers
            if missing_tickers:
                raise ValueError(f"Missing price data for tickers: {missing_tickers}")
        
        return self.price_data
    
    def load_sentiment(self, sentiment_path: str) -> pd.DataFrame:
        """
        Load sentiment data.
        
        Args:
            sentiment_path: Path to sentiment CSV file
            
        Returns:
            DataFrame with sentiment data
        """
        self.sentiment_data = pd.read_csv(sentiment_path)
        self.sentiment_data['date'] = pd.to_datetime(self.sentiment_data['date'])
        self.sentiment_data.set_index('date', inplace=True)
        
        return self.sentiment_data
    
    def load_current_state(self, current_state_path: str) -> Dict:
        """
        Load current market state.
        
        Args:
            current_state_path: Path to current state JSON file
            
        Returns:
            Dictionary with current state data
        """
        with open(current_state_path, 'r') as f:
            self.current_state = json.load(f)
            
        return self.current_state
    
    def calculate_returns(self, price_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate daily returns from price data.
        
        Args:
            price_data: Price DataFrame (uses self.price_data if None)
            
        Returns:
            DataFrame with daily returns
        """
        if price_data is None:
            price_data = self.price_data
            
        if price_data is None:
            raise ValueError("No price data loaded")
            
        returns = price_data.pct_change().dropna()
        return returns
    
    def align_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Align price, sentiment, and returns data by date.
        
        Returns:
            Tuple of (prices, sentiment, returns) DataFrames with aligned dates
        """
        if any(data is None for data in [self.price_data, self.sentiment_data]):
            raise ValueError("Must load price and sentiment data first")
        
        # Calculate returns
        returns = self.calculate_returns()
        
        # Find common date range
        common_dates = self.price_data.index.intersection(self.sentiment_data.index)
        common_dates = common_dates.intersection(returns.index)
        
        if len(common_dates) == 0:
            raise ValueError("No common dates found between price and sentiment data")
        
        # Align all data to common dates
        aligned_prices = self.price_data.loc[common_dates]
        aligned_sentiment = self.sentiment_data.loc[common_dates]
        aligned_returns = returns.loc[common_dates]
        
        return aligned_prices, aligned_sentiment, aligned_returns
    
    def get_portfolio_weights(self) -> Dict[str, float]:
        """
        Get portfolio weights as dictionary.
        
        Returns:
            Dictionary mapping ticker to weight
        """
        if self.portfolio_data is None:
            raise ValueError("No portfolio data loaded")
            
        return dict(zip(self.portfolio_data['ticker'], self.portfolio_data['weight']))
    
    def get_portfolio_tickers(self) -> list:
        """
        Get list of portfolio tickers.
        
        Returns:
            List of ticker symbols
        """
        if self.portfolio_data is None:
            raise ValueError("No portfolio data loaded")
            
        return list(self.portfolio_data['ticker'])
    
    def create_pypfopt_data(self) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Prepare data in PyPortfolioOpt format.
        
        Returns:
            Tuple of (price_data, weights_dict) for PyPortfolioOpt
        """
        if self.price_data is None or self.portfolio_data is None:
            raise ValueError("Must load price and portfolio data first")
        
        # Get portfolio tickers only
        portfolio_tickers = self.get_portfolio_tickers()
        price_data_filtered = self.price_data[portfolio_tickers].copy()
        
        # Get weights dictionary
        weights_dict = self.get_portfolio_weights()
        
        return price_data_filtered, weights_dict
    
    def summary(self) -> Dict:
        """
        Get summary of loaded data.
        
        Returns:
            Dictionary with data summary statistics
        """
        summary = {}
        
        if self.portfolio_data is not None:
            summary['portfolio'] = {
                'num_assets': len(self.portfolio_data),
                'tickers': list(self.portfolio_data['ticker']),
                'total_weight': self.portfolio_data['weight'].sum()
            }
        
        if self.price_data is not None:
            summary['prices'] = {
                'date_range': f"{self.price_data.index.min()} to {self.price_data.index.max()}",
                'num_days': len(self.price_data),
                'num_assets': len(self.price_data.columns)
            }
        
        if self.sentiment_data is not None:
            summary['sentiment'] = {
                'date_range': f"{self.sentiment_data.index.min()} to {self.sentiment_data.index.max()}",
                'num_days': len(self.sentiment_data),
                'sentiment_range': f"{self.sentiment_data.select_dtypes(include=[np.number]).min().min():.3f} to {self.sentiment_data.select_dtypes(include=[np.number]).max().max():.3f}"
            }
        
        return summary
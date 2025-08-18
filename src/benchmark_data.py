import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List
import logging

# Data source imports
try:
    from nselib import capital_market
    NSELIB_AVAILABLE = True
except ImportError:
    NSELIB_AVAILABLE = False
    
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



class BenchmarkDownloader:
    """Download benchmark indices using nselib or yfinance"""
    
    # Index mappings for different sources
    INDEX_MAPPINGS = {
        'nselib': {
            'Nifty 50': 'Nifty 50',
            'Nifty IT': 'Nifty IT',
            'Nifty Bank': 'Nifty Bank', 
            'Nifty Auto': 'Nifty Auto',
            'Nifty Pharma': 'Nifty Pharma',
            'Nifty FMCG': 'Nifty FMCG',
            'Nifty Metal': 'Nifty Metal',
            'Nifty Energy': 'Nifty Energy',
            'Nifty Financial Services': 'Nifty Financial Services',
            'Nifty Media': 'Nifty Media',
            'Nifty Realty': 'Nifty Realty'
        },
        'yfinance': {
            'Nifty 50': '^NSEI',
            'Nifty IT': '^CNXIT',
            'Nifty Bank': '^NSEBANK',
            'Nifty Auto': '^CNXAUTO',
            'Nifty Pharma': '^CNXPHARMA',
            'Nifty FMCG': '^CNXFMCG',
            'Nifty Metal': '^CNXMETAL',
            'Nifty Energy': '^CNXENERGY',
            'Nifty Financial Services': '^CNXFINANCE',
            'Nifty Media': '^CNXMEDIA',
            'Nifty Realty': '^CNXREALTY'
        }
    }
    
    def __init__(self, prefer_source: str = 'nselib'):
        """
        Initialize downloader
        
        Args:
            prefer_source: 'nselib' or 'yfinance'
        """
        self.available_sources = []
        
        if NSELIB_AVAILABLE:
            self.available_sources.append('nselib')
        if YFINANCE_AVAILABLE:
            self.available_sources.append('yfinance')
            
        if not self.available_sources:
            raise ImportError("No supported data sources available. Install nselib or yfinance")
        
        if prefer_source in self.available_sources:
            self.primary_source = prefer_source
        else:
            self.primary_source = self.available_sources[0]
            
        logger.info(f"Primary source: {self.primary_source}, Available: {self.available_sources}")
    
    def get_available_indices(self) -> List[str]:
        """Get list of available indices"""
        return list(self.INDEX_MAPPINGS.get(self.primary_source, {}).keys())
    
    def _download_nselib(self, index: str, from_date: str, to_date: str) -> pd.DataFrame:
        """Download data using nselib"""
        mapped_index = self.INDEX_MAPPINGS['nselib'].get(index, index)
        return capital_market.index_data(
            index=mapped_index,
            from_date=from_date,
            to_date=to_date
        )
    
    def _download_yfinance(self, index: str, from_date: str, to_date: str) -> pd.DataFrame:
        """Download data using yfinance"""
        mapped_index = self.INDEX_MAPPINGS['yfinance'].get(index, index)
        
        # Convert date format for yfinance (YYYY-MM-DD)
        start_date = datetime.strptime(from_date, '%d-%m-%Y').strftime('%Y-%m-%d')
        end_date = datetime.strptime(to_date, '%d-%m-%Y').strftime('%Y-%m-%d')
        
        ticker = yf.Ticker(mapped_index)
        data = ticker.history(start=start_date, end=end_date)
        
        # Reset index to get Date as column and standardize column names
        data.reset_index(inplace=True)
        data.rename(columns={
            'Date': 'Date',
            'Open': 'Open',
            'High': 'High', 
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        }, inplace=True)
        
        return data
    
    def download_index_data(self, index: str, from_date: str, to_date: str, 
                          fallback: bool = True) -> pd.DataFrame:
        """
        Download index data with fallback support
        
        Args:
            index: Index name (e.g., 'Nifty 50', 'Nifty IT')
            from_date: Start date in DD-MM-YYYY format
            to_date: End date in DD-MM-YYYY format
            fallback: Use fallback source if primary fails
            
        Returns:
            DataFrame with index data
        """
        # Try primary source first
        try:
            if self.primary_source == 'nselib' and NSELIB_AVAILABLE:
                return self._download_nselib(index, from_date, to_date)
            elif self.primary_source == 'yfinance' and YFINANCE_AVAILABLE:
                return self._download_yfinance(index, from_date, to_date)
        except Exception as e:
            logger.warning(f"Primary source {self.primary_source} failed for {index}: {str(e)}")
            
            if not fallback:
                raise
        
        # Try fallback source
        if fallback:
            for source in self.available_sources:
                if source != self.primary_source:
                    try:
                        logger.info(f"Trying fallback source: {source}")
                        if source == 'nselib' and NSELIB_AVAILABLE:
                            return self._download_nselib(index, from_date, to_date)
                        elif source == 'yfinance' and YFINANCE_AVAILABLE:
                            return self._download_yfinance(index, from_date, to_date)
                    except Exception as e:
                        logger.warning(f"Fallback source {source} failed for {index}: {str(e)}")
        
        raise RuntimeError(f"All data sources failed for index: {index}")
    
    def download_multiple_indices(self, 
                                indices: List[str], 
                                from_date: str, 
                                to_date: str,
                                save_directory: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Download multiple indices data
        
        Args:
            indices: List of index names
            from_date: Start date in DD-MM-YYYY format
            to_date: End date in DD-MM-YYYY format
            save_directory: Optional directory to save CSV files
            
        Returns:
            Dictionary mapping index names to DataFrames
        """
        results = {}
        
        if save_directory:
            os.makedirs(save_directory, exist_ok=True)
        
        for index in indices:
            try:
                logger.info(f"Downloading {index} data...")
                data = self.download_index_data(index, from_date, to_date)
                results[index] = data
                
                if save_directory:
                    filename = f"{index.replace(' ', '_').lower()}_{from_date.replace('-', '')}_to_{to_date.replace('-', '')}.csv"
                    filepath = os.path.join(save_directory, filename)
                    data.to_csv(filepath, index=False)
                    logger.info(f"Saved {index} data to {filepath}")
                    
            except Exception as e:
                logger.error(f"Failed to download {index}: {str(e)}")
                
        return results

def download_and_save_nifty_data():
    """Legacy function - maintained for backward compatibility"""
    project_root = get_project_root()
    portfolio_csv_path = os.path.join(project_root, 'data', 'nifty_prices_new.csv')

    downloader = BenchmarkDownloader()
    data = downloader.download_index_data('Nifty 50', '01-01-2024', '15-08-2025')
    data.to_csv(portfolio_csv_path, index=False)

def download_benchmark_indices(indices: List[str] = None, 
                             from_date: str = '01-01-2024',
                             to_date: str = '15-08-2025',
                             prefer_source: str = 'nselib') -> Dict[str, pd.DataFrame]:
    """
    Download multiple benchmark indices
    
    Args:
        indices: List of index names. If None, downloads common indices
        from_date: Start date in DD-MM-YYYY format
        to_date: End date in DD-MM-YYYY format
        prefer_source: Preferred data source ('nselib' or 'yfinance')
        
    Returns:
        Dictionary mapping index names to DataFrames
    """
    if indices is None:
        indices = ['Nifty 50', 'Nifty IT', 'Nifty Bank', 'Nifty Auto', 'Nifty Pharma']
    
    project_root = get_project_root()
    data_dir = os.path.join(project_root, 'data', 'benchmarks')
    
    downloader = BenchmarkDownloader(prefer_source)
    
    logger.info(f"Available indices: {downloader.get_available_indices()}")
    
    return downloader.download_multiple_indices(
        indices=indices,
        from_date=from_date,
        to_date=to_date,
        save_directory=data_dir
    )

class BenchMarkData:
    """Class for processing benchmark index data and calculating returns"""
    
    def __init__(self, index_csv_path: str):
        """
        Initialize with path to index CSV file
        
        Args:
            index_csv_path: Path to CSV file containing index data
        """
        self.index_csv_path = index_csv_path
        self.data = None
        self.returns = None
    
    def load_data(self) -> pd.DataFrame:
        """Load index data from CSV file"""
        if self.data is None:
            self.data = pd.read_csv(self.index_csv_path)
            logger.info(f"Loaded data from {self.index_csv_path} - Shape: {self.data.shape}")
        return self.data
    
    def calculate_index_returns(self, date_column: str = 'TIMESTAMP', 
                              price_column: str = 'CLOSE_INDEX_VAL') -> pd.Series:
        """
        Calculate daily returns for index from CSV file
        
        Args:
            date_column: Name of the date column (default: 'TIMESTAMP')
            price_column: Name of the closing price column (default: 'CLOSE_INDEX_VAL')
            
        Returns:
            pandas.Series with daily returns for the index
        """
        # Load data if not already loaded
        df = self.load_data().copy()
        
        # Convert date column to datetime
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column], format='%d-%m-%Y')
        else:
            raise ValueError(f"Date column '{date_column}' not found in data")
        
        # Check if price column exists
        if price_column not in df.columns:
            raise ValueError(f"Price column '{price_column}' not found in data. Available columns: {list(df.columns)}")
        
        # Remove duplicate dates by keeping the first occurrence
        df = df.drop_duplicates(subset=[date_column], keep='first')
        
        # Set date as index and sort
        df.set_index(date_column, inplace=True)
        df.sort_index(inplace=True)
        
        # Calculate daily returns using percentage change
        index_returns = df[price_column].pct_change().dropna()
        
        # Store returns for future use
        self.returns = index_returns
        
        logger.info(f"Calculated {len(index_returns)} daily returns")
        logger.info(f"Mean daily return: {index_returns.mean():.6f}")
        logger.info(f"Daily volatility: {index_returns.std():.6f}")
        
        return index_returns
    
    def get_returns_summary(self) -> Dict:
        """
        Get summary statistics of calculated returns
        
        Returns:
            Dictionary containing return statistics
        """
        if self.returns is None:
            raise ValueError("Returns not calculated yet. Call calculate_index_returns() first.")
        
        return {
            'count': len(self.returns),
            'mean_daily_return': self.returns.mean(),
            'daily_volatility': self.returns.std(),
            'annualized_return': self.returns.mean() * 252,
            'annualized_volatility': self.returns.std() * np.sqrt(252),
            'min_return': self.returns.min(),
            'max_return': self.returns.max(),
            'skewness': self.returns.skew(),
            'kurtosis': self.returns.kurtosis()
        }
    
    @staticmethod
    def calculate_returns_from_csv(csv_path: str, date_column: str = 'TIMESTAMP', 
                                 price_column: str = 'CLOSE_INDEX_VAL') -> pd.Series:
        """
        Static method to calculate returns directly from CSV path
        
        Args:
            csv_path: Path to CSV file
            date_column: Name of the date column
            price_column: Name of the closing price column
            
        Returns:
            pandas.Series with daily returns
        """
        benchmark = BenchMarkData(csv_path)
        return benchmark.calculate_index_returns(date_column, price_column)


if __name__ == "__main__":
    # Example usage - Download benchmark indices
    indices_to_download = ['Nifty 50', 'Nifty IT', 'Nifty Bank']
    results = download_benchmark_indices(
        indices=indices_to_download,
        from_date='01-01-2024',
        to_date='15-08-2025'
    )
    
    for index_name, data in results.items():
        print(f"\n{index_name} - Shape: {data.shape}")
        print(data.head())
    
    print("\n" + "="*60)
    print("TESTING INDEX RETURNS CALCULATION")
    print("="*60)
    
    # Test returns calculation using downloaded data
    project_root = get_project_root()
    data_dir = os.path.join(project_root, 'data', 'benchmarks')
    
    # Test with Nifty 50 data
    nifty_file = os.path.join(data_dir, 'nifty_50_01012024_to_15082025.csv')
    
    if os.path.exists(nifty_file):
        print(f"\nTesting with: {nifty_file}")
        
        # Method 1: Using class instance
        benchmark = BenchMarkData(nifty_file)
        returns = benchmark.calculate_index_returns()
        
        print(f"\nNifty 50 Returns (first 10):")
        print(returns.head(10))
        
        # Get summary statistics
        summary = benchmark.get_returns_summary()
        print(f"\nReturns Summary:")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")
        
        # Method 2: Using static method
        print(f"\n" + "-"*40)
        print("Testing static method:")
        static_returns = BenchMarkData.calculate_returns_from_csv(nifty_file)
        print(f"Static method returns count: {len(static_returns)}")
        
    else:
        print(f"File not found: {nifty_file}")
        print("Run the download section first to generate test data.")


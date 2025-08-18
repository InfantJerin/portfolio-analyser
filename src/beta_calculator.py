import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
import logging

# Import our modules
from portfolio import Portfolio
from benchmark_data import BenchMarkData, get_project_root

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_beta(portfolio_returns, market_returns):
    """
    Calculate the beta for a portfolio.
    
    Beta = Covariance(Portfolio Returns, Market Returns) / Variance(Market Returns)
    
    Args:
        portfolio_returns: pandas.Series of portfolio daily returns
        market_returns: pandas.Series of market (Nifty) daily returns
    
    Returns:
        float: Beta value of the portfolio
    """
    # Align the returns by date (ensure same dates)
    aligned_data = pd.concat([portfolio_returns, market_returns], axis=1, join='inner')
    aligned_data.columns = ['portfolio', 'market']
    
    # Calculate covariance between portfolio and market returns
    covariance = np.cov(aligned_data['portfolio'], aligned_data['market'])[0, 1]
    
    # Calculate variance of market returns
    market_variance = np.var(aligned_data['market'])
    
    # Beta = Covariance / Market Variance
    beta = covariance / market_variance
    
    return beta


class BetaCalculator:
    """
    Calculate portfolio sensitivity (beta) to different benchmark indices
    """
    
    def __init__(self, portfolio: Portfolio, benchmark_data_dir: Optional[str] = None):
        """
        Initialize Beta Calculator
        
        Args:
            portfolio: Portfolio object with daily returns calculation capability
            benchmark_data_dir: Directory containing benchmark CSV files (auto-detected if None)
        """
        self.portfolio = portfolio
        self.portfolio_returns = None
        
        # Set benchmark data directory
        if benchmark_data_dir is None:
            project_root = get_project_root()
            self.benchmark_data_dir = os.path.join(project_root, 'data', 'benchmarks')
        else:
            self.benchmark_data_dir = benchmark_data_dir
            
        # Available indices mapping (filename pattern to display name)
        self.available_indices = {
            'nifty_50': 'Nifty 50',
            'nifty_it': 'Nifty IT', 
            'nifty_bank': 'Nifty Bank',
            'nifty_auto': 'Nifty Auto',
            'nifty_pharma': 'Nifty Pharma',
            'nifty_fmcg': 'Nifty FMCG',
            'nifty_metal': 'Nifty Metal',
            'nifty_energy': 'Nifty Energy',
            'nifty_financial_services': 'Nifty Financial Services',
            'nifty_media': 'Nifty Media',
            'nifty_realty': 'Nifty Realty'
        }
        
        logger.info(f"Beta Calculator initialized with benchmark directory: {self.benchmark_data_dir}")
    
    def _get_portfolio_returns(self) -> pd.Series:
        """Get portfolio daily returns (cached)"""
        if self.portfolio_returns is None:
            self.portfolio_returns = self.portfolio.calculate_daily_returns()
            logger.info(f"Calculated portfolio returns: {len(self.portfolio_returns)} days")
        return self.portfolio_returns
    
    def _find_benchmark_file(self, index_key: str) -> Optional[str]:
        """
        Find benchmark CSV file for given index
        
        Args:
            index_key: Index key (e.g., 'nifty_50', 'nifty_it')
            
        Returns:
            Full path to CSV file if found, None otherwise
        """
        # Look for files matching pattern: {index_key}_*.csv
        import glob
        pattern = os.path.join(self.benchmark_data_dir, f"{index_key}_*.csv")
        files = glob.glob(pattern)
        
        if files:
            # Return the first matching file (or most recent if multiple)
            return sorted(files)[-1]
        return None
    
    def calculate_single_beta(self, index_name: str) -> Optional[Tuple[float, Dict]]:
        """
        Calculate beta for portfolio against a single index
        
        Args:
            index_name: Index name (e.g., 'Nifty 50', 'Nifty IT') or key (e.g., 'nifty_50')
            
        Returns:
            Tuple of (beta_value, metadata) or None if calculation fails
        """
        # Convert display name to key if needed
        index_key = None
        if index_name in self.available_indices.values():
            # Find key by value
            index_key = next(k for k, v in self.available_indices.items() if v == index_name)
        elif index_name.lower().replace(' ', '_') in self.available_indices:
            index_key = index_name.lower().replace(' ', '_')
        else:
            logger.error(f"Unknown index: {index_name}")
            return None
        
        # Find benchmark file
        benchmark_file = self._find_benchmark_file(index_key)
        if not benchmark_file:
            logger.error(f"Benchmark file not found for {index_name} (key: {index_key})")
            return None
        
        try:
            # Get portfolio returns
            portfolio_returns = self._get_portfolio_returns()
            
            # Get benchmark returns
            benchmark_returns = BenchMarkData.calculate_returns_from_csv(benchmark_file)
            
            # Calculate beta
            beta = calculate_beta(portfolio_returns, benchmark_returns)
            
            # Create metadata
            metadata = {
                'index_name': self.available_indices[index_key],
                'index_key': index_key,
                'benchmark_file': benchmark_file,
                'portfolio_periods': len(portfolio_returns),
                'benchmark_periods': len(benchmark_returns),
                'overlapping_periods': len(pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner')),
                'portfolio_mean_return': portfolio_returns.mean(),
                'portfolio_volatility': portfolio_returns.std(),
                'benchmark_mean_return': benchmark_returns.mean(),
                'benchmark_volatility': benchmark_returns.std()
            }
            
            logger.info(f"Calculated beta for {index_name}: {beta:.4f}")
            return beta, metadata
            
        except Exception as e:
            logger.error(f"Error calculating beta for {index_name}: {str(e)}")
            return None
    
    def calculate_multiple_betas(self, indices: Optional[List[str]] = None) -> Dict[str, Tuple[float, Dict]]:
        """
        Calculate beta for portfolio against multiple indices
        
        Args:
            indices: List of index names/keys. If None, calculates for all available indices
            
        Returns:
            Dictionary mapping index names to (beta, metadata) tuples
        """
        if indices is None:
            indices = list(self.available_indices.values())
        
        results = {}
        
        logger.info(f"Calculating betas for {len(indices)} indices...")
        
        for index_name in indices:
            result = self.calculate_single_beta(index_name)
            if result is not None:
                beta, metadata = result
                results[metadata['index_name']] = (beta, metadata)
        
        return results
    
    def get_sensitivity_analysis(self, indices: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get comprehensive sensitivity analysis as DataFrame
        
        Args:
            indices: List of index names. If None, analyzes all available indices
            
        Returns:
            DataFrame with beta analysis results
        """
        results = self.calculate_multiple_betas(indices)
        
        if not results:
            logger.warning("No beta calculations successful")
            return pd.DataFrame()
        
        # Create summary DataFrame
        data = []
        for index_name, (beta, metadata) in results.items():
            data.append({
                'Index': index_name,
                'Beta': beta,
                'Sensitivity': self._interpret_beta(beta),
                'Overlapping_Periods': metadata['overlapping_periods'],
                'Portfolio_Mean_Return': metadata['portfolio_mean_return'],
                'Portfolio_Volatility': metadata['portfolio_volatility'],
                'Benchmark_Mean_Return': metadata['benchmark_mean_return'],
                'Benchmark_Volatility': metadata['benchmark_volatility'],
                'Correlation': self._calculate_correlation(metadata)
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Beta', ascending=False)
        
        return df
    
    def _interpret_beta(self, beta: float) -> str:
        """Interpret beta value"""
        if beta > 1.2:
            return "Highly Aggressive"
        elif beta > 1.0:
            return "Aggressive"
        elif beta > 0.8:
            return "Moderately Aggressive"
        elif beta > 0.5:
            return "Defensive"
        elif beta > 0:
            return "Highly Defensive"
        else:
            return "Negative Sensitivity"
    
    def _calculate_correlation(self, metadata: Dict) -> float:
        """Calculate correlation coefficient for additional insight"""
        try:
            # Re-align the data to calculate correlation
            portfolio_returns = self._get_portfolio_returns()
            benchmark_file = metadata['benchmark_file']
            benchmark_returns = BenchMarkData.calculate_returns_from_csv(benchmark_file)
            
            aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner')
            correlation = aligned_data.corr().iloc[0, 1]
            return correlation
            
        except Exception:
            return np.nan
    
    def print_sensitivity_report(self, indices: Optional[List[str]] = None):
        """
        Print a formatted sensitivity analysis report
        
        Args:
            indices: List of index names to analyze. If None, analyzes all available
        """
        df = self.get_sensitivity_analysis(indices)
        
        if df.empty:
            print("No sensitivity analysis data available")
            return
        
        print("\n" + "="*80)
        print("PORTFOLIO SENSITIVITY ANALYSIS (BETA CALCULATION)")
        print("="*80)
        
        print(f"\nPortfolio Data:")
        print(f"- Data file: {self.portfolio.portfolio_csv_path}")
        print(f"- Portfolio periods: {len(self._get_portfolio_returns())}")
        print(f"- Portfolio mean daily return: {self._get_portfolio_returns().mean():.6f}")
        print(f"- Portfolio daily volatility: {self._get_portfolio_returns().std():.6f}")
        
        print(f"\nBeta Analysis Results:")
        print("-" * 80)
        
        for _, row in df.iterrows():
            print(f"\n{row['Index']}:")
            print(f"  Beta: {row['Beta']:.4f} ({row['Sensitivity']})")
            print(f"  Correlation: {row['Correlation']:.4f}")
            print(f"  Overlapping periods: {row['Overlapping_Periods']}")
            print(f"  Index mean return: {row['Benchmark_Mean_Return']:.6f}")
            print(f"  Index volatility: {row['Benchmark_Volatility']:.6f}")
        
        print(f"\n" + "-" * 80)
        print(f"Beta Interpretation:")
        print(f"- Beta > 1.0: Portfolio is more volatile than the index (aggressive)")
        print(f"- Beta < 1.0: Portfolio is less volatile than the index (defensive)")
        print(f"- Beta ≈ 1.0: Portfolio moves in line with the index")
        print(f"- Negative Beta: Portfolio moves opposite to the index")
        
        # Summary statistics
        print(f"\nSummary:")
        print(f"- Highest Beta: {df.iloc[0]['Index']} ({df.iloc[0]['Beta']:.4f})")
        print(f"- Lowest Beta: {df.iloc[-1]['Index']} ({df.iloc[-1]['Beta']:.4f})")
        print(f"- Average Beta: {df['Beta'].mean():.4f}")
        print(f"- Most Correlated: {df.loc[df['Correlation'].idxmax(), 'Index']} ({df['Correlation'].max():.4f})")
    
    def get_available_indices(self) -> Dict[str, str]:
        """Get dictionary of available indices"""
        return self.available_indices.copy()


def test_beta_calculator():
    """Test the beta calculator with sample data"""
    print("TESTING BETA CALCULATOR")
    print("="*50)
    
    # Check if we have test portfolio data
    project_root = get_project_root()
    test_portfolio_path = os.path.join(project_root, 'data', 'test_portfolio.csv')
    
    if not os.path.exists(test_portfolio_path):
        print(f"Test portfolio not found at {test_portfolio_path}")
        print("Run the market_scenario.py script first to generate test portfolio data")
        return
    
    # Create portfolio object
    portfolio = Portfolio(test_portfolio_path)
    
    # Create beta calculator
    beta_calc = BetaCalculator(portfolio)
    
    # Test with specific indices first
    test_indices = ['Nifty 50', 'Nifty IT', 'Nifty Bank']
    
    print(f"\nTesting with indices: {test_indices}")
    print("-" * 50)
    
    # Calculate single beta
    result = beta_calc.calculate_single_beta('Nifty 50')
    if result:
        beta, metadata = result
        print(f"\nSingle Beta Calculation (Nifty 50):")
        print(f"Beta: {beta:.4f}")
        print(f"Overlapping periods: {metadata['overlapping_periods']}")
    
    # Calculate multiple betas
    results = beta_calc.calculate_multiple_betas(test_indices)
    print(f"\nMultiple Beta Calculations:")
    for index_name, (beta, metadata) in results.items():
        print(f"{index_name}: {beta:.4f}")
    
    # Get sensitivity analysis DataFrame
    df = beta_calc.get_sensitivity_analysis(test_indices)
    if not df.empty:
        print(f"\nSensitivity Analysis DataFrame:")
        print(df[['Index', 'Beta', 'Sensitivity', 'Correlation', 'Overlapping_Periods']].to_string(index=False))
    
    # Print full report
    beta_calc.print_sensitivity_report(test_indices)


def demo_beta_calculator_usage():
   
    beta_calc = BetaCalculator(Portfolio("dummy"))  # Just for getting available indices
    for key, name in beta_calc.get_available_indices().items():
        print(f"- {name} (key: {key})")


if __name__ == "__main__":
    test_beta_calculator()
    #demo_beta_calculator_usage() 

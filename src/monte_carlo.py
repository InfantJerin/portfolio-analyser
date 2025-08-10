"""
Monte Carlo Simulation Module
Monte Carlo and bootstrapping simulation for portfolio forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class MonteCarloSimulation:
    """
    Monte Carlo simulation engine for portfolio forecasting and risk analysis.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize MonteCarloSimulation.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.n_simulations = config.get('simulation_params', {}).get('monte_carlo_iterations', 10000)
        self.bootstrap_blocks = config.get('simulation_params', {}).get('bootstrap_blocks', 1000)
        self.random_seed = config.get('simulation_params', {}).get('random_seed', 42)
        self.forecast_horizon = config.get('forecast_horizon', 20)
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        
    def monte_carlo_portfolio_simulation(self,
                                       expected_returns: pd.Series,
                                       covariance_matrix: pd.DataFrame,
                                       weights: pd.Series,
                                       initial_portfolio_value: float = 100000) -> Dict:
        """
        Run Monte Carlo simulation for portfolio performance.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Asset covariance matrix
            weights: Portfolio weights
            initial_portfolio_value: Starting portfolio value
            
        Returns:
            Dictionary with simulation results
        """
        print(f"Running Monte Carlo simulation with {self.n_simulations} iterations...")
        
        # Convert expected returns to daily from annual
        daily_expected_returns = expected_returns / 252
        daily_covariance = covariance_matrix / 252
        
        # Storage for simulation results
        portfolio_paths = np.zeros((self.n_simulations, self.forecast_horizon + 1))
        portfolio_paths[:, 0] = initial_portfolio_value
        
        final_values = []
        max_drawdowns = []
        
        for sim in range(self.n_simulations):
            # Generate correlated random returns
            random_returns = np.random.multivariate_normal(
                daily_expected_returns.values,
                daily_covariance.values,
                self.forecast_horizon
            )
            
            # Calculate portfolio returns for each day
            portfolio_returns = np.dot(random_returns, weights.values)
            
            # Calculate portfolio value path
            portfolio_values = [initial_portfolio_value]
            for day_return in portfolio_returns:
                new_value = portfolio_values[-1] * (1 + day_return)
                portfolio_values.append(new_value)
            
            portfolio_paths[sim, :] = portfolio_values
            final_values.append(portfolio_values[-1])
            
            # Calculate maximum drawdown for this path
            running_max = np.maximum.accumulate(portfolio_values)
            drawdowns = (np.array(portfolio_values) - running_max) / running_max
            max_drawdowns.append(drawdowns.min())
        
        # Calculate statistics
        final_values = np.array(final_values)
        returns = (final_values - initial_portfolio_value) / initial_portfolio_value
        
        # Percentiles for confidence intervals
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        value_percentiles = np.percentile(final_values, percentiles)
        return_percentiles = np.percentile(returns, percentiles)
        
        results = {
            'simulation_params': {
                'n_simulations': self.n_simulations,
                'forecast_horizon': self.forecast_horizon,
                'initial_value': initial_portfolio_value
            },
            'final_values': {
                'mean': np.mean(final_values),
                'std': np.std(final_values),
                'min': np.min(final_values),
                'max': np.max(final_values),
                'percentiles': dict(zip(percentiles, value_percentiles))
            },
            'returns': {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'min': np.min(returns),
                'max': np.max(returns),
                'percentiles': dict(zip(percentiles, return_percentiles))
            },
            'risk_metrics': {
                'probability_of_loss': (returns < 0).mean(),
                'expected_shortfall_5%': np.mean(returns[returns <= np.percentile(returns, 5)]),
                'expected_shortfall_1%': np.mean(returns[returns <= np.percentile(returns, 1)]),
                'max_drawdown_mean': np.mean(max_drawdowns),
                'max_drawdown_worst': np.min(max_drawdowns)
            },
            'portfolio_paths': portfolio_paths,
            'final_values_array': final_values,
            'returns_array': returns
        }
        
        return results
    
    def bootstrap_simulation(self,
                           historical_returns: pd.DataFrame,
                           weights: pd.Series,
                           initial_portfolio_value: float = 100000,
                           block_size: int = 5) -> Dict:
        """
        Run bootstrap simulation using historical return blocks.
        
        Args:
            historical_returns: Historical return data
            weights: Portfolio weights
            initial_portfolio_value: Starting portfolio value
            block_size: Size of blocks to resample
            
        Returns:
            Dictionary with bootstrap simulation results
        """
        print(f"Running bootstrap simulation with {self.bootstrap_blocks} blocks...")
        
        # Calculate historical portfolio returns
        historical_portfolio_returns = (historical_returns * weights.values).sum(axis=1)
        
        # Create overlapping blocks
        returns_array = historical_portfolio_returns.values
        n_returns = len(returns_array)
        
        if n_returns < self.forecast_horizon:
            raise ValueError(f"Not enough historical data. Need at least {self.forecast_horizon} days.")
        
        # Storage for bootstrap results
        bootstrap_paths = []
        final_values = []
        max_drawdowns = []
        
        for _ in range(self.bootstrap_blocks):
            # Randomly select starting point for block sampling
            simulated_returns = []
            
            while len(simulated_returns) < self.forecast_horizon:
                # Random starting point for block
                start_idx = np.random.randint(0, max(1, n_returns - block_size + 1))
                end_idx = min(start_idx + block_size, n_returns)
                block_returns = returns_array[start_idx:end_idx]
                
                # Add block to simulated returns
                remaining_needed = self.forecast_horizon - len(simulated_returns)
                simulated_returns.extend(block_returns[:remaining_needed])
            
            # Calculate portfolio value path
            portfolio_values = [initial_portfolio_value]
            for day_return in simulated_returns[:self.forecast_horizon]:
                new_value = portfolio_values[-1] * (1 + day_return)
                portfolio_values.append(new_value)
            
            bootstrap_paths.append(portfolio_values)
            final_values.append(portfolio_values[-1])
            
            # Calculate maximum drawdown
            running_max = np.maximum.accumulate(portfolio_values)
            drawdowns = (np.array(portfolio_values) - running_max) / running_max
            max_drawdowns.append(drawdowns.min())
        
        # Convert to arrays
        bootstrap_paths = np.array(bootstrap_paths)
        final_values = np.array(final_values)
        returns = (final_values - initial_portfolio_value) / initial_portfolio_value
        
        # Calculate statistics
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        value_percentiles = np.percentile(final_values, percentiles)
        return_percentiles = np.percentile(returns, percentiles)
        
        results = {
            'simulation_params': {
                'n_blocks': self.bootstrap_blocks,
                'block_size': block_size,
                'forecast_horizon': self.forecast_horizon,
                'initial_value': initial_portfolio_value
            },
            'final_values': {
                'mean': np.mean(final_values),
                'std': np.std(final_values),
                'min': np.min(final_values),
                'max': np.max(final_values),
                'percentiles': dict(zip(percentiles, value_percentiles))
            },
            'returns': {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'min': np.min(returns),
                'max': np.max(returns),
                'percentiles': dict(zip(percentiles, return_percentiles))
            },
            'risk_metrics': {
                'probability_of_loss': (returns < 0).mean(),
                'expected_shortfall_5%': np.mean(returns[returns <= np.percentile(returns, 5)]),
                'expected_shortfall_1%': np.mean(returns[returns <= np.percentile(returns, 1)]),
                'max_drawdown_mean': np.mean(max_drawdowns),
                'max_drawdown_worst': np.min(max_drawdowns)
            },
            'portfolio_paths': bootstrap_paths,
            'final_values_array': final_values,
            'returns_array': returns
        }
        
        return results
    
    def sentiment_adjusted_simulation(self,
                                    base_expected_returns: pd.Series,
                                    covariance_matrix: pd.DataFrame,
                                    weights: pd.Series,
                                    sentiment_adjustment: float,
                                    initial_portfolio_value: float = 100000) -> Dict:
        """
        Run Monte Carlo simulation with sentiment adjustment.
        
        Args:
            base_expected_returns: Base expected returns
            covariance_matrix: Asset covariance matrix
            weights: Portfolio weights
            sentiment_adjustment: Adjustment factor for returns (-1 to 1)
            initial_portfolio_value: Starting portfolio value
            
        Returns:
            Dictionary with sentiment-adjusted simulation results
        """
        # Adjust expected returns based on sentiment
        adjusted_returns = base_expected_returns * (1 + sentiment_adjustment)
        
        # Run standard Monte Carlo with adjusted returns
        results = self.monte_carlo_portfolio_simulation(
            adjusted_returns,
            covariance_matrix,
            weights,
            initial_portfolio_value
        )
        
        # Add sentiment information
        results['sentiment_adjustment'] = sentiment_adjustment
        results['adjusted_returns'] = adjusted_returns.to_dict()
        
        return results
    
    def compare_simulations(self, simulation_results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare results from multiple simulations.
        
        Args:
            simulation_results: Dictionary mapping simulation names to results
            
        Returns:
            DataFrame comparing simulation statistics
        """
        comparison_data = []
        
        for sim_name, results in simulation_results.items():
            row = {
                'simulation': sim_name,
                'mean_return': results['returns']['mean'],
                'return_std': results['returns']['std'],
                'return_5th_percentile': results['returns']['percentiles'][5],
                'return_95th_percentile': results['returns']['percentiles'][95],
                'probability_of_loss': results['risk_metrics']['probability_of_loss'],
                'expected_shortfall_5%': results['risk_metrics']['expected_shortfall_5%'],
                'max_drawdown_worst': results['risk_metrics']['max_drawdown_worst']
            }
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def calculate_risk_metrics(self, returns_array: np.ndarray, confidence_levels: List[float] = None) -> Dict:
        """
        Calculate comprehensive risk metrics from simulation results.
        
        Args:
            returns_array: Array of simulated returns
            confidence_levels: List of confidence levels for VaR/CVaR
            
        Returns:
            Dictionary with risk metrics
        """
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]
        
        risk_metrics = {}
        
        # Basic statistics
        risk_metrics['mean_return'] = np.mean(returns_array)
        risk_metrics['volatility'] = np.std(returns_array)
        risk_metrics['skewness'] = self._calculate_skewness(returns_array)
        risk_metrics['kurtosis'] = self._calculate_kurtosis(returns_array)
        
        # VaR and CVaR for each confidence level
        for confidence_level in confidence_levels:
            alpha = 1 - confidence_level
            var_value = np.percentile(returns_array, alpha * 100)
            cvar_value = np.mean(returns_array[returns_array <= var_value])
            
            risk_metrics[f'VaR_{confidence_level}'] = var_value
            risk_metrics[f'CVaR_{confidence_level}'] = cvar_value
        
        # Additional risk metrics
        risk_metrics['probability_of_loss'] = (returns_array < 0).mean()
        risk_metrics['maximum_loss'] = np.min(returns_array)
        risk_metrics['maximum_gain'] = np.max(returns_array)
        
        return risk_metrics
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis
    
    def export_simulation_paths(self, simulation_results: Dict, filename: str = None) -> pd.DataFrame:
        """
        Export simulation paths to DataFrame.
        
        Args:
            simulation_results: Results from simulation
            filename: Optional filename to save CSV
            
        Returns:
            DataFrame with simulation paths
        """
        paths_df = pd.DataFrame(
            simulation_results['portfolio_paths'].T,
            columns=[f'simulation_{i}' for i in range(simulation_results['portfolio_paths'].shape[0])]
        )
        paths_df.index.name = 'day'
        
        if filename:
            paths_df.to_csv(filename)
        
        return paths_df
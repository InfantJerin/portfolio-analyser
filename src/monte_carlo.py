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
    
    def calculate_goal_probabilities(self, final_values: np.ndarray, targets: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate probability of reaching specific financial targets.
        
        Args:
            final_values: Array of simulated final portfolio values
            targets: Dictionary of target names and values (e.g., {'retirement_goal': 5000000})
            
        Returns:
            Dictionary with probability of reaching each target
        """
        probabilities = {}
        
        for target_name, target_value in targets.items():
            prob_success = (final_values >= target_value).mean()
            probabilities[target_name] = prob_success
            
        return probabilities
    
    def retirement_analysis(self, simulation_results: Dict, retirement_target: float, 
                          shortfall_threshold: float = None, upside_threshold: float = None) -> Dict:
        """
        Comprehensive retirement planning analysis.
        
        Args:
            simulation_results: Results from Monte Carlo simulation
            retirement_target: Target retirement amount (e.g., 50000000 for ₹5 crore)
            shortfall_threshold: Threshold for shortfall analysis (default: 80% of target)
            upside_threshold: Threshold for upside analysis (default: 160% of target)
            
        Returns:
            Dictionary with retirement planning insights
        """
        final_values = simulation_results['final_values_array']
        initial_value = simulation_results['simulation_params']['initial_value']
        
        # Set default thresholds if not provided
        if shortfall_threshold is None:
            shortfall_threshold = retirement_target * 0.8  # 80% of target
        if upside_threshold is None:
            upside_threshold = retirement_target * 1.6    # 160% of target
            
        # Calculate key probabilities
        prob_success = (final_values >= retirement_target).mean()
        prob_shortfall = (final_values < shortfall_threshold).mean()
        prob_upside = (final_values >= upside_threshold).mean()
        
        # Calculate expected outcomes in different scenarios
        successful_outcomes = final_values[final_values >= retirement_target]
        shortfall_outcomes = final_values[final_values < shortfall_threshold]
        upside_outcomes = final_values[final_values >= upside_threshold]
        
        # Additional analysis
        median_outcome = np.median(final_values)
        mean_outcome = np.mean(final_values)
        worst_case_5pct = np.percentile(final_values, 5)
        best_case_95pct = np.percentile(final_values, 95)
        
        analysis = {
            'retirement_target': retirement_target,
            'shortfall_threshold': shortfall_threshold,
            'upside_threshold': upside_threshold,
            'initial_investment': initial_value,
            'forecast_horizon_years': simulation_results['simulation_params']['forecast_horizon'] / 252,
            
            # Core probabilities
            'probability_success': prob_success,
            'probability_shortfall': prob_shortfall, 
            'probability_upside': prob_upside,
            
            # Expected outcomes
            'median_outcome': median_outcome,
            'mean_outcome': mean_outcome,
            'worst_case_5pct': worst_case_5pct,
            'best_case_95pct': best_case_95pct,
            
            # Conditional analysis
            'mean_if_successful': np.mean(successful_outcomes) if len(successful_outcomes) > 0 else None,
            'mean_if_shortfall': np.mean(shortfall_outcomes) if len(shortfall_outcomes) > 0 else None,
            'mean_if_upside': np.mean(upside_outcomes) if len(upside_outcomes) > 0 else None,
            
            # Risk metrics
            'shortfall_amount_mean': max(0, retirement_target - np.mean(final_values[final_values < retirement_target])) if any(final_values < retirement_target) else 0,
            'probability_of_loss': (final_values < initial_value).mean(),
        }
        
        return analysis
    
    def goal_based_summary(self, retirement_analysis: Dict, currency_symbol: str = "₹") -> str:
        """
        Generate user-friendly retirement planning summary.
        
        Args:
            retirement_analysis: Results from retirement_analysis method
            currency_symbol: Currency symbol for formatting
            
        Returns:
            Formatted string with retirement planning insights
        """
        target = retirement_analysis['retirement_target']
        shortfall_threshold = retirement_analysis['shortfall_threshold'] 
        upside_threshold = retirement_analysis['upside_threshold']
        
        prob_success = retirement_analysis['probability_success']
        prob_shortfall = retirement_analysis['probability_shortfall']
        prob_upside = retirement_analysis['probability_upside']
        
        median_outcome = retirement_analysis['median_outcome']
        
        # Format amounts in crores for Indian context
        target_crores = target / 10_000_000
        shortfall_crores = shortfall_threshold / 10_000_000
        upside_crores = upside_threshold / 10_000_000
        median_crores = median_outcome / 10_000_000
        
        summary = f"""
🎯 RETIREMENT PLANNING ANALYSIS
{'='*50}

Target: {currency_symbol}{target_crores:.1f} crore

📊 PROBABILITY BREAKDOWN:
• {prob_success:.1%} probability of reaching {currency_symbol}{target_crores:.1f} crore target
• {prob_shortfall:.1%} probability of shortfall (less than {currency_symbol}{shortfall_crores:.1f} crore)
• {prob_upside:.1%} probability of upside success (more than {currency_symbol}{upside_crores:.1f} crore)

💰 EXPECTED OUTCOME:
• Median portfolio value: {currency_symbol}{median_crores:.2f} crore
• Most likely range: {currency_symbol}{retirement_analysis['worst_case_5pct']/10_000_000:.2f} - {currency_symbol}{retirement_analysis['best_case_95pct']/10_000_000:.2f} crore (90% confidence)

⚠️ RISK ANALYSIS:
• Probability of any loss: {retirement_analysis['probability_of_loss']:.1%}
• Years to goal: {retirement_analysis['forecast_horizon_years']:.1f} years
"""
        
        # Add recommendation based on success probability
        if prob_success >= 0.8:
            summary += "\n✅ RECOMMENDATION: Strong likelihood of meeting retirement goal!"
        elif prob_success >= 0.6:
            summary += "\n⚠️ RECOMMENDATION: Moderate success probability. Consider increasing contributions."
        else:
            summary += "\n🚨 RECOMMENDATION: Low success probability. Reassess strategy or extend timeline."
            
        return summary
    
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
"""
Risk Management Module

This module provides comprehensive risk analysis capabilities including
VaR, CVaR, stress testing, correlation analysis, and risk attribution.
"""

import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')


class RiskMetrics:
    """Container for risk metrics."""
    
    def __init__(self):
        self.var_95 = None
        self.var_99 = None
        self.cvar_95 = None
        self.cvar_99 = None
        self.volatility = None
        self.max_drawdown = None
        self.beta = None
        self.tracking_error = None
        self.information_ratio = None
        self.downside_deviation = None
        self.sortino_ratio = None


class RiskManager:
    """
    Comprehensive Risk Management System.
    
    This class provides:
    - Value at Risk (VaR) and Conditional VaR (CVaR) calculations
    - Stress testing and scenario analysis
    - Correlation and covariance analysis
    - Risk attribution and decomposition
    - Monte Carlo simulations
    - Factor risk modeling
    - Risk budgeting and allocation
    """
    
    def __init__(self, 
                 confidence_levels: List[float] = [0.95, 0.99],
                 risk_free_rate: float = 0.02,
                 trading_days_per_year: int = 252):
        """
        Initialize Risk Manager.
        
        Args:
            confidence_levels: Confidence levels for VaR/CVaR calculations
            risk_free_rate: Risk-free rate for Sharpe ratio calculations
            trading_days_per_year: Number of trading days per year
        """
        self.confidence_levels = confidence_levels
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days_per_year
        
        # Risk models and data
        self.returns_data: Optional[pd.DataFrame] = None
        self.covariance_matrix: Optional[pd.DataFrame] = None
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.factor_loadings: Optional[pd.DataFrame] = None
        
        # Historical data for analysis
        self.historical_portfolios: List[Dict] = []
        self.benchmark_returns: Optional[pd.Series] = None
    
    def load_returns_data(self, returns: Union[pd.DataFrame, Dict[str, List[float]]]):
        """
        Load historical returns data for risk analysis.
        
        Args:
            returns: Returns data as DataFrame or dictionary
        """
        if isinstance(returns, dict):
            self.returns_data = pd.DataFrame(returns)
        else:
            self.returns_data = returns.copy()
        
        # Calculate risk matrices
        self.covariance_matrix = self.returns_data.cov()
        self.correlation_matrix = self.returns_data.corr()
    
    def set_benchmark_returns(self, benchmark_returns: Union[pd.Series, List[float]]):
        """Set benchmark returns for beta and tracking error calculations."""
        if isinstance(benchmark_returns, list):
            self.benchmark_returns = pd.Series(benchmark_returns)
        else:
            self.benchmark_returns = benchmark_returns.copy()
    
    def calculate_portfolio_var(self, 
                              weights: Dict[str, float],
                              method: str = "historical",
                              time_horizon: int = 1) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) for a portfolio.
        
        Args:
            weights: Portfolio weights
            method: VaR calculation method ("historical", "parametric", "monte_carlo")
            time_horizon: Time horizon in days
            
        Returns:
            Dictionary with VaR values for different confidence levels
        """
        if self.returns_data is None:
            raise ValueError("Returns data not loaded. Call load_returns_data() first.")
        
        # Calculate portfolio returns
        weights_series = pd.Series(weights)
        portfolio_returns = (self.returns_data * weights_series).sum(axis=1)
        
        var_results = {}
        
        if method == "historical":
            var_results = self._calculate_historical_var(portfolio_returns, time_horizon)
        elif method == "parametric":
            var_results = self._calculate_parametric_var(portfolio_returns, time_horizon)
        elif method == "monte_carlo":
            var_results = self._calculate_monte_carlo_var(weights, time_horizon)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        return var_results
    
    def _calculate_historical_var(self, 
                                portfolio_returns: pd.Series, 
                                time_horizon: int) -> Dict[str, float]:
        """Calculate VaR using historical simulation method."""
        # Scale returns for time horizon
        scaled_returns = portfolio_returns * np.sqrt(time_horizon)
        
        var_results = {}
        for confidence_level in self.confidence_levels:
            alpha = 1 - confidence_level
            var_value = np.percentile(scaled_returns, alpha * 100)
            var_results[f'VaR_{confidence_level}'] = var_value
        
        return var_results
    
    def _calculate_parametric_var(self, 
                                portfolio_returns: pd.Series, 
                                time_horizon: int) -> Dict[str, float]:
        """Calculate VaR using parametric (normal distribution) method."""
        mean_return = portfolio_returns.mean()
        volatility = portfolio_returns.std()
        
        # Scale for time horizon
        scaled_mean = mean_return * time_horizon
        scaled_volatility = volatility * np.sqrt(time_horizon)
        
        var_results = {}
        for confidence_level in self.confidence_levels:
            alpha = 1 - confidence_level
            z_score = stats.norm.ppf(alpha)
            var_value = scaled_mean + z_score * scaled_volatility
            var_results[f'VaR_{confidence_level}'] = var_value
        
        return var_results
    
    def _calculate_monte_carlo_var(self, 
                                 weights: Dict[str, float], 
                                 time_horizon: int,
                                 num_simulations: int = 10000) -> Dict[str, float]:
        """Calculate VaR using Monte Carlo simulation."""
        weights_array = np.array([weights[col] for col in self.returns_data.columns])
        
        # Generate random returns based on multivariate normal distribution
        mean_returns = self.returns_data.mean().values
        cov_matrix = self.covariance_matrix.values
        
        # Scale for time horizon
        scaled_mean = mean_returns * time_horizon
        scaled_cov = cov_matrix * time_horizon
        
        # Run Monte Carlo simulation
        random_returns = np.random.multivariate_normal(
            scaled_mean, scaled_cov, num_simulations
        )
        
        # Calculate portfolio returns for each simulation
        portfolio_returns = np.dot(random_returns, weights_array)
        
        var_results = {}
        for confidence_level in self.confidence_levels:
            alpha = 1 - confidence_level
            var_value = np.percentile(portfolio_returns, alpha * 100)
            var_results[f'VaR_{confidence_level}'] = var_value
        
        return var_results
    
    def calculate_cvar(self, 
                      weights: Dict[str, float],
                      method: str = "historical",
                      time_horizon: int = 1) -> Dict[str, float]:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).
        
        Args:
            weights: Portfolio weights
            method: Calculation method
            time_horizon: Time horizon in days
            
        Returns:
            Dictionary with CVaR values
        """
        if self.returns_data is None:
            raise ValueError("Returns data not loaded.")
        
        weights_series = pd.Series(weights)
        portfolio_returns = (self.returns_data * weights_series).sum(axis=1)
        
        # Scale for time horizon
        scaled_returns = portfolio_returns * np.sqrt(time_horizon)
        
        cvar_results = {}
        for confidence_level in self.confidence_levels:
            alpha = 1 - confidence_level
            var_threshold = np.percentile(scaled_returns, alpha * 100)
            
            # CVaR is the expected value of returns below VaR threshold
            tail_returns = scaled_returns[scaled_returns <= var_threshold]
            cvar_value = tail_returns.mean() if len(tail_returns) > 0 else var_threshold
            
            cvar_results[f'CVaR_{confidence_level}'] = cvar_value
        
        return cvar_results
    
    def calculate_portfolio_volatility(self, weights: Dict[str, float]) -> float:
        """Calculate portfolio volatility using covariance matrix."""
        if self.covariance_matrix is None:
            raise ValueError("Covariance matrix not available.")
        
        weights_series = pd.Series(weights)
        portfolio_variance = np.dot(weights_series.values, 
                                  np.dot(self.covariance_matrix.values, weights_series.values))
        return np.sqrt(portfolio_variance)
    
    def calculate_risk_contribution(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate risk contribution of each asset to portfolio risk.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Dictionary with risk contributions
        """
        if self.covariance_matrix is None:
            raise ValueError("Covariance matrix not available.")
        
        weights_series = pd.Series(weights)
        
        # Calculate portfolio variance
        portfolio_variance = np.dot(weights_series.values, 
                                  np.dot(self.covariance_matrix.values, weights_series.values))
        
        # Calculate marginal risk contributions
        marginal_contributions = np.dot(self.covariance_matrix.values, weights_series.values)
        
        # Calculate component risk contributions
        risk_contributions = {}
        for i, asset in enumerate(weights_series.index):
            contribution = weights_series.iloc[i] * marginal_contributions[i] / portfolio_variance
            risk_contributions[asset] = contribution
        
        return risk_contributions
    
    def calculate_correlation_analysis(self) -> Dict[str, Any]:
        """
        Perform comprehensive correlation analysis.
        
        Returns:
            Dictionary with correlation statistics
        """
        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix not available.")
        
        # Extract upper triangle of correlation matrix (excluding diagonal)
        upper_triangle = np.triu(self.correlation_matrix.values, k=1)
        correlations = upper_triangle[upper_triangle != 0]
        
        return {
            'average_correlation': float(np.mean(correlations)),
            'max_correlation': float(np.max(correlations)),
            'min_correlation': float(np.min(correlations)),
            'correlation_std': float(np.std(correlations)),
            'correlation_matrix': self.correlation_matrix.to_dict(),
            'highly_correlated_pairs': self._find_highly_correlated_pairs(threshold=0.8),
            'diversification_ratio': self._calculate_diversification_ratio()
        }
    
    def _find_highly_correlated_pairs(self, threshold: float = 0.8) -> List[Dict]:
        """Find pairs of assets with high correlation."""
        highly_correlated = []
        
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i + 1, len(self.correlation_matrix.columns)):
                correlation = self.correlation_matrix.iloc[i, j]
                if abs(correlation) >= threshold:
                    highly_correlated.append({
                        'asset1': self.correlation_matrix.columns[i],
                        'asset2': self.correlation_matrix.columns[j],
                        'correlation': float(correlation)
                    })
        
        return highly_correlated
    
    def _calculate_diversification_ratio(self) -> float:
        """Calculate portfolio diversification ratio."""
        if self.covariance_matrix is None:
            return 0.0
        
        # Diversification ratio = weighted average volatility / portfolio volatility
        individual_volatilities = np.sqrt(np.diag(self.covariance_matrix.values))
        equal_weights = np.ones(len(individual_volatilities)) / len(individual_volatilities)
        
        weighted_avg_volatility = np.dot(equal_weights, individual_volatilities)
        portfolio_volatility = self.calculate_portfolio_volatility(
            dict(zip(self.covariance_matrix.columns, equal_weights))
        )
        
        return weighted_avg_volatility / portfolio_volatility if portfolio_volatility > 0 else 1.0
    
    def stress_test_portfolio(self, 
                            weights: Dict[str, float],
                            scenarios: Dict[str, Dict[str, float]]) -> Dict[str, Dict]:
        """
        Perform stress testing with custom scenarios.
        
        Args:
            weights: Portfolio weights
            scenarios: Dictionary of scenario_name -> {asset: shock_percentage}
            
        Returns:
            Dictionary with stress test results
        """
        if self.returns_data is None:
            raise ValueError("Returns data not loaded.")
        
        stress_results = {}
        base_value = 100.0  # Assume $100 portfolio value
        
        for scenario_name, shocks in scenarios.items():
            scenario_return = 0.0
            
            for asset, weight in weights.items():
                shock = shocks.get(asset, 0.0)
                scenario_return += weight * shock
            
            new_value = base_value * (1 + scenario_return)
            loss = base_value - new_value
            
            stress_results[scenario_name] = {
                'scenario_return': scenario_return,
                'portfolio_value': new_value,
                'absolute_loss': loss,
                'percentage_loss': (loss / base_value) * 100,
                'shocks_applied': shocks
            }
        
        return stress_results
    
    def calculate_maximum_drawdown(self, 
                                 portfolio_values: List[float],
                                 dates: List[date] = None) -> Dict[str, Any]:
        """
        Calculate maximum drawdown and related statistics.
        
        Args:
            portfolio_values: Series of portfolio values
            dates: Corresponding dates (optional)
            
        Returns:
            Dictionary with drawdown statistics
        """
        if len(portfolio_values) < 2:
            return {'max_drawdown': 0.0}
        
        values = np.array(portfolio_values)
        peak = values[0]
        max_drawdown = 0.0
        drawdown_start_idx = 0
        drawdown_end_idx = 0
        max_drawdown_start_idx = 0
        max_drawdown_end_idx = 0
        
        current_drawdown = 0.0
        in_drawdown = False
        
        for i, value in enumerate(values):
            if value > peak:
                peak = value
                if in_drawdown:
                    # End of drawdown period
                    in_drawdown = False
                    if current_drawdown > max_drawdown:
                        max_drawdown = current_drawdown
                        max_drawdown_start_idx = drawdown_start_idx
                        max_drawdown_end_idx = i - 1
                    current_drawdown = 0.0
            else:
                if not in_drawdown:
                    # Start of new drawdown
                    in_drawdown = True
                    drawdown_start_idx = i
                
                current_drawdown = (peak - value) / peak
        
        # Handle case where drawdown continues to end
        if in_drawdown and current_drawdown > max_drawdown:
            max_drawdown = current_drawdown
            max_drawdown_start_idx = drawdown_start_idx
            max_drawdown_end_idx = len(values) - 1
        
        result = {
            'max_drawdown': max_drawdown,
            'max_drawdown_percentage': max_drawdown * 100,
            'drawdown_start_index': max_drawdown_start_idx,
            'drawdown_end_index': max_drawdown_end_idx
        }
        
        if dates:
            result.update({
                'drawdown_start_date': dates[max_drawdown_start_idx].isoformat(),
                'drawdown_end_date': dates[max_drawdown_end_idx].isoformat(),
                'drawdown_duration_days': (dates[max_drawdown_end_idx] - 
                                         dates[max_drawdown_start_idx]).days
            })
        
        return result
    
    def calculate_beta(self, 
                      portfolio_returns: pd.Series,
                      benchmark_returns: pd.Series = None) -> float:
        """Calculate portfolio beta relative to benchmark."""
        if benchmark_returns is None:
            benchmark_returns = self.benchmark_returns
        
        if benchmark_returns is None:
            raise ValueError("Benchmark returns not available.")
        
        # Align the series
        aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        if len(aligned_data) < 2:
            return 1.0  # Default beta
        
        port_returns = aligned_data.iloc[:, 0]
        bench_returns = aligned_data.iloc[:, 1]
        
        covariance = np.cov(port_returns, bench_returns)[0, 1]
        benchmark_variance = np.var(bench_returns)
        
        return covariance / benchmark_variance if benchmark_variance > 0 else 1.0
    
    def calculate_tracking_error(self, 
                               portfolio_returns: pd.Series,
                               benchmark_returns: pd.Series = None) -> float:
        """Calculate tracking error (standard deviation of excess returns)."""
        if benchmark_returns is None:
            benchmark_returns = self.benchmark_returns
        
        if benchmark_returns is None:
            raise ValueError("Benchmark returns not available.")
        
        # Calculate excess returns
        excess_returns = portfolio_returns - benchmark_returns
        return float(excess_returns.std())
    
    def calculate_information_ratio(self, 
                                  portfolio_returns: pd.Series,
                                  benchmark_returns: pd.Series = None) -> float:
        """Calculate information ratio (excess return / tracking error)."""
        if benchmark_returns is None:
            benchmark_returns = self.benchmark_returns
        
        if benchmark_returns is None:
            raise ValueError("Benchmark returns not available.")
        
        excess_returns = portfolio_returns - benchmark_returns
        mean_excess = excess_returns.mean()
        tracking_error = excess_returns.std()
        
        return float(mean_excess / tracking_error) if tracking_error > 0 else 0.0
    
    def calculate_downside_metrics(self, 
                                 portfolio_returns: pd.Series,
                                 target_return: float = 0.0) -> Dict[str, float]:
        """
        Calculate downside risk metrics.
        
        Args:
            portfolio_returns: Portfolio return series
            target_return: Target return for downside calculation
            
        Returns:
            Dictionary with downside metrics
        """
        # Downside returns (returns below target)
        downside_returns = portfolio_returns[portfolio_returns < target_return]
        
        if len(downside_returns) == 0:
            return {
                'downside_deviation': 0.0,
                'sortino_ratio': float('inf') if portfolio_returns.mean() > target_return else 0.0,
                'downside_frequency': 0.0
            }
        
        # Downside deviation
        downside_deviation = np.sqrt(np.mean((downside_returns - target_return) ** 2))
        
        # Sortino ratio
        excess_return = portfolio_returns.mean() - target_return
        sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0.0
        
        # Downside frequency
        downside_frequency = len(downside_returns) / len(portfolio_returns)
        
        return {
            'downside_deviation': float(downside_deviation),
            'sortino_ratio': float(sortino_ratio),
            'downside_frequency': float(downside_frequency)
        }
    
    def generate_risk_report(self, 
                           weights: Dict[str, float],
                           portfolio_returns: pd.Series = None) -> Dict[str, Any]:
        """
        Generate comprehensive risk report.
        
        Args:
            weights: Portfolio weights
            portfolio_returns: Portfolio return series (optional)
            
        Returns:
            Comprehensive risk report
        """
        if self.returns_data is None:
            raise ValueError("Returns data not loaded.")
        
        # Calculate portfolio returns if not provided
        if portfolio_returns is None:
            weights_series = pd.Series(weights)
            portfolio_returns = (self.returns_data * weights_series).sum(axis=1)
        
        # Calculate all risk metrics
        var_metrics = self.calculate_portfolio_var(weights, method="historical")
        cvar_metrics = self.calculate_cvar(weights)
        volatility = self.calculate_portfolio_volatility(weights)
        risk_contributions = self.calculate_risk_contribution(weights)
        correlation_analysis = self.calculate_correlation_analysis()
        downside_metrics = self.calculate_downside_metrics(portfolio_returns)
        
        # Calculate relative metrics if benchmark is available
        relative_metrics = {}
        if self.benchmark_returns is not None:
            relative_metrics = {
                'beta': self.calculate_beta(portfolio_returns),
                'tracking_error': self.calculate_tracking_error(portfolio_returns),
                'information_ratio': self.calculate_information_ratio(portfolio_returns)
            }
        
        # Stress test scenarios
        stress_scenarios = {
            'market_crash': {asset: -0.20 for asset in weights.keys()},
            'sector_rotation': {asset: -0.10 if i % 2 == 0 else 0.05 
                              for i, asset in enumerate(weights.keys())},
            'interest_rate_shock': {asset: -0.15 for asset in weights.keys()}
        }
        stress_results = self.stress_test_portfolio(weights, stress_scenarios)
        
        return {
            'portfolio_weights': weights,
            'risk_metrics': {
                'volatility': float(volatility),
                'annualized_volatility': float(volatility * np.sqrt(self.trading_days)),
                **var_metrics,
                **cvar_metrics,
                **downside_metrics
            },
            'relative_metrics': relative_metrics,
            'risk_attribution': {
                'risk_contributions': risk_contributions,
                'largest_risk_contributor': max(risk_contributions.items(), key=lambda x: x[1]),
                'risk_concentration': max(risk_contributions.values())
            },
            'correlation_analysis': correlation_analysis,
            'stress_test_results': stress_results,
            'risk_summary': {
                'overall_risk_level': self._assess_risk_level(volatility),
                'diversification_score': correlation_analysis.get('diversification_ratio', 1.0),
                'concentration_risk': max(list(weights.values())) if weights else 0.0
            }
        }
    
    def _assess_risk_level(self, volatility: float) -> str:
        """Assess overall risk level based on volatility."""
        annualized_vol = volatility * np.sqrt(self.trading_days)
        
        if annualized_vol < 0.10:
            return "LOW"
        elif annualized_vol < 0.20:
            return "MODERATE"
        elif annualized_vol < 0.30:
            return "HIGH"
        else:
            return "VERY_HIGH"
    
    def monte_carlo_simulation(self, 
                             weights: Dict[str, float],
                             initial_value: float = 100000,
                             time_horizon_days: int = 252,
                             num_simulations: int = 10000) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for portfolio projections.
        
        Args:
            weights: Portfolio weights
            initial_value: Initial portfolio value
            time_horizon_days: Simulation horizon in days
            num_simulations: Number of simulation paths
            
        Returns:
            Dictionary with simulation results
        """
        if self.returns_data is None:
            raise ValueError("Returns data not loaded.")
        
        weights_array = np.array([weights[col] for col in self.returns_data.columns])
        
        # Calculate expected returns and covariance
        mean_returns = self.returns_data.mean().values
        cov_matrix = self.covariance_matrix.values
        
        # Storage for simulation results
        final_values = []
        paths = []
        
        for _ in range(num_simulations):
            # Generate random returns for each day
            daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, time_horizon_days)
            
            # Calculate portfolio returns
            portfolio_daily_returns = np.dot(daily_returns, weights_array)
            
            # Calculate portfolio value path
            value_path = [initial_value]
            current_value = initial_value
            
            for daily_return in portfolio_daily_returns:
                current_value *= (1 + daily_return)
                value_path.append(current_value)
            
            final_values.append(current_value)
            if len(paths) < 100:  # Store only first 100 paths to save memory
                paths.append(value_path)
        
        final_values = np.array(final_values)
        
        # Calculate statistics
        return {
            'simulation_parameters': {
                'initial_value': initial_value,
                'time_horizon_days': time_horizon_days,
                'num_simulations': num_simulations
            },
            'final_value_statistics': {
                'mean': float(np.mean(final_values)),
                'median': float(np.median(final_values)),
                'std': float(np.std(final_values)),
                'min': float(np.min(final_values)),
                'max': float(np.max(final_values)),
                'percentile_5': float(np.percentile(final_values, 5)),
                'percentile_25': float(np.percentile(final_values, 25)),
                'percentile_75': float(np.percentile(final_values, 75)),
                'percentile_95': float(np.percentile(final_values, 95))
            },
            'probability_analysis': {
                'prob_positive_return': float(np.mean(final_values > initial_value)),
                'prob_loss_greater_10pct': float(np.mean(final_values < initial_value * 0.9)),
                'prob_loss_greater_20pct': float(np.mean(final_values < initial_value * 0.8)),
                'expected_return': float((np.mean(final_values) - initial_value) / initial_value),
                'expected_shortfall_5pct': float(np.mean(final_values[final_values <= np.percentile(final_values, 5)]))
            },
            'sample_paths': paths[:10] if paths else []  # Return first 10 paths as samples
        }
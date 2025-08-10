"""
Portfolio Analyzer Module
Portfolio risk analysis and optimization using PyPortfolioOpt.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# PyPortfolioOpt imports
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions
# Discrete allocation will be imported conditionally
from pypfopt import plotting


class PortfolioAnalyzer:
    """
    Portfolio analysis and optimization using PyPortfolioOpt with sentiment adjustments.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize PortfolioAnalyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.risk_free_rate = config.get('risk_params', {}).get('risk_free_rate', 0.02)
        self.confidence_levels = config.get('risk_params', {}).get('confidence_levels', [0.95, 0.99])
        
        # Store calculated data
        self.base_expected_returns = None
        self.sentiment_adjusted_returns = None
        self.covariance_matrix = None
        self.current_weights = None
        
    def calculate_base_metrics(self, 
                             price_data: pd.DataFrame, 
                             weights: Dict[str, float]) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Calculate base expected returns and covariance matrix using PyPortfolioOpt.
        
        Args:
            price_data: Historical price data
            weights: Portfolio weights
            
        Returns:
            Tuple of (expected_returns, covariance_matrix)
        """
        # Calculate base expected returns using multiple methods
        mu_mean = expected_returns.mean_historical_return(price_data, frequency=252)
        mu_ema = expected_returns.ema_historical_return(price_data, frequency=252)
        
        # Use EMA as primary, fallback to mean
        self.base_expected_returns = mu_ema.fillna(mu_mean)
        
        # Calculate covariance matrix
        self.covariance_matrix = risk_models.sample_cov(price_data, frequency=252)
        
        # Store current weights
        self.current_weights = pd.Series(weights)
        
        return self.base_expected_returns, self.covariance_matrix
    
    def adjust_returns_for_sentiment(self, 
                                   sentiment_predictions: Dict[str, Dict],
                                   adjustment_factor: float = 1.0) -> pd.Series:
        """
        Adjust expected returns based on sentiment predictions.
        
        Args:
            sentiment_predictions: Predictions from sentiment model
            adjustment_factor: Factor to scale sentiment impact
            
        Returns:
            Sentiment-adjusted expected returns
        """
        if self.base_expected_returns is None:
            raise ValueError("Must calculate base metrics first")
        
        adjusted_returns = self.base_expected_returns.copy()
        
        for asset, prediction in sentiment_predictions.items():
            if asset in adjusted_returns.index:
                # Adjust return based on predicted return from sentiment
                predicted_return = prediction['predicted_return']
                
                # Weighted combination of base return and sentiment prediction
                base_return = self.base_expected_returns[asset]
                adjusted_return = (0.7 * base_return + 0.3 * predicted_return * adjustment_factor)
                adjusted_returns[asset] = adjusted_return
        
        self.sentiment_adjusted_returns = adjusted_returns
        return adjusted_returns
    
    def calculate_portfolio_metrics(self, 
                                  weights: Optional[Dict[str, float]] = None,
                                  use_sentiment_adjusted: bool = True) -> Dict:
        """
        Calculate portfolio performance metrics.
        
        Args:
            weights: Portfolio weights (uses current if None)
            use_sentiment_adjusted: Whether to use sentiment-adjusted returns
            
        Returns:
            Dictionary with portfolio metrics
        """
        if weights is None:
            weights = self.current_weights
        else:
            weights = pd.Series(weights)
        
        if use_sentiment_adjusted and self.sentiment_adjusted_returns is not None:
            expected_returns = self.sentiment_adjusted_returns
        else:
            expected_returns = self.base_expected_returns
        
        if expected_returns is None or self.covariance_matrix is None:
            raise ValueError("Must calculate base metrics first")
        
        # Portfolio return
        portfolio_return = (weights * expected_returns).sum()
        
        # Portfolio volatility
        portfolio_variance = np.dot(weights.values, np.dot(self.covariance_matrix.values, weights.values))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'risk_free_rate': self.risk_free_rate
        }
    
    def calculate_var_cvar(self, 
                          returns_series: pd.Series,
                          weights: Optional[Dict[str, float]] = None) -> Dict:
        """
        Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR).
        
        Args:
            returns_series: Historical returns data
            weights: Portfolio weights
            
        Returns:
            Dictionary with VaR and CVaR values
        """
        if weights is None:
            weights = self.current_weights
        else:
            weights = pd.Series(weights)
        
        # Calculate portfolio returns
        portfolio_returns = (returns_series * weights.values).sum(axis=1)
        
        var_cvar_results = {}
        
        for confidence_level in self.confidence_levels:
            alpha = 1 - confidence_level
            
            # Value at Risk (VaR)
            var_value = np.percentile(portfolio_returns, alpha * 100)
            
            # Conditional Value at Risk (CVaR)
            cvar_value = portfolio_returns[portfolio_returns <= var_value].mean()
            
            var_cvar_results[f'VaR_{confidence_level}'] = var_value
            var_cvar_results[f'CVaR_{confidence_level}'] = cvar_value
        
        return var_cvar_results
    
    def optimize_portfolio(self, 
                          objective: str = 'max_sharpe',
                          use_sentiment_adjusted: bool = True) -> Dict:
        """
        Optimize portfolio using PyPortfolioOpt.
        
        Args:
            objective: Optimization objective ('max_sharpe', 'min_volatility', 'max_return')
            use_sentiment_adjusted: Whether to use sentiment-adjusted returns
            
        Returns:
            Dictionary with optimized weights and performance
        """
        if use_sentiment_adjusted and self.sentiment_adjusted_returns is not None:
            expected_returns = self.sentiment_adjusted_returns
        else:
            expected_returns = self.base_expected_returns
        
        if expected_returns is None or self.covariance_matrix is None:
            raise ValueError("Must calculate base metrics first")
        
        # Create EfficientFrontier object
        ef = EfficientFrontier(expected_returns, self.covariance_matrix)
        
        # Optimize based on objective
        if objective == 'max_sharpe':
            weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        elif objective == 'min_volatility':
            weights = ef.min_volatility()
        elif objective == 'max_return':
            weights = ef.max_return()
        else:
            raise ValueError(f"Unknown objective: {objective}")
        
        # Clean weights (remove tiny positions)
        cleaned_weights = ef.clean_weights()
        
        # Calculate performance
        performance = ef.portfolio_performance(
            verbose=False,
            risk_free_rate=self.risk_free_rate
        )
        
        return {
            'weights': cleaned_weights,
            'expected_return': performance[0],
            'volatility': performance[1],
            'sharpe_ratio': performance[2]
        }
    
    def efficient_frontier_analysis(self, 
                                  num_portfolios: int = 100,
                                  use_sentiment_adjusted: bool = True) -> pd.DataFrame:
        """
        Generate efficient frontier analysis.
        
        Args:
            num_portfolios: Number of portfolios to generate
            use_sentiment_adjusted: Whether to use sentiment-adjusted returns
            
        Returns:
            DataFrame with efficient frontier data
        """
        if use_sentiment_adjusted and self.sentiment_adjusted_returns is not None:
            expected_returns = self.sentiment_adjusted_returns
        else:
            expected_returns = self.base_expected_returns
        
        if expected_returns is None or self.covariance_matrix is None:
            raise ValueError("Must calculate base metrics first")
        
        # Generate efficient frontier
        ef = EfficientFrontier(expected_returns, self.covariance_matrix)
        
        # Get return range
        min_return = expected_returns.min()
        max_return = expected_returns.max()
        target_returns = np.linspace(min_return, max_return, num_portfolios)
        
        frontier_data = []
        
        for target_return in target_returns:
            try:
                ef_copy = EfficientFrontier(expected_returns, self.covariance_matrix)
                weights = ef_copy.efficient_return(target_return)
                performance = ef_copy.portfolio_performance(
                    verbose=False,
                    risk_free_rate=self.risk_free_rate
                )
                
                frontier_data.append({
                    'return': performance[0],
                    'volatility': performance[1],
                    'sharpe_ratio': performance[2],
                    'weights': weights
                })
            except:
                continue
        
        return pd.DataFrame(frontier_data)
    
    def scenario_analysis(self, 
                         scenarios: Dict[str, Dict[str, float]],
                         base_weights: Optional[Dict[str, float]] = None) -> Dict:
        """
        Analyze portfolio performance under different scenarios.
        
        Args:
            scenarios: Dictionary of scenarios with asset return adjustments
            base_weights: Base portfolio weights
            
        Returns:
            Dictionary with scenario analysis results
        """
        if base_weights is None:
            base_weights = self.current_weights
        
        results = {}
        
        for scenario_name, return_adjustments in scenarios.items():
            # Create scenario-adjusted returns
            scenario_returns = self.base_expected_returns.copy()
            
            for asset, adjustment in return_adjustments.items():
                if asset in scenario_returns.index:
                    scenario_returns[asset] += adjustment
            
            # Calculate portfolio metrics for this scenario
            portfolio_return = (base_weights * scenario_returns).sum()
            portfolio_variance = np.dot(base_weights.values, 
                                      np.dot(self.covariance_matrix.values, base_weights.values))
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            results[scenario_name] = {
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'scenario_returns': scenario_returns.to_dict()
            }
        
        return results
    
    def get_asset_contributions(self, weights: Optional[Dict[str, float]] = None) -> Dict:
        """
        Calculate individual asset contributions to portfolio risk and return.
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Dictionary with asset contributions
        """
        if weights is None:
            weights = self.current_weights
        else:
            weights = pd.Series(weights)
        
        if self.base_expected_returns is None or self.covariance_matrix is None:
            raise ValueError("Must calculate base metrics first")
        
        # Return contributions
        return_contributions = weights * self.base_expected_returns
        
        # Risk contributions (marginal contribution to portfolio variance)
        portfolio_variance = np.dot(weights.values, np.dot(self.covariance_matrix.values, weights.values))
        marginal_contributions = np.dot(self.covariance_matrix.values, weights.values)
        risk_contributions = weights.values * marginal_contributions / portfolio_variance
        
        return {
            'return_contributions': return_contributions.to_dict(),
            'risk_contributions': dict(zip(weights.index, risk_contributions)),
            'total_return_contribution': return_contributions.sum(),
            'total_risk_contribution': risk_contributions.sum()
        }
    
    def calculate_current_portfolio_value(self, holdings_data: Dict) -> Dict:
        """
        Calculate current portfolio metrics based on actual holdings.
        
        Args:
            holdings_data: Dictionary with current holdings information
            
        Returns:
            Dictionary with current portfolio value analysis
        """
        total_value = holdings_data['total_portfolio_value']
        analysis = {
            'total_value': total_value,
            'holdings_breakdown': {},
            'weight_analysis': {}
        }
        
        for ticker, holding in holdings_data.items():
            if ticker == 'total_portfolio_value':
                continue
                
            analysis['holdings_breakdown'][ticker] = {
                'shares': holding['shares_held'],
                'price': holding['current_price'],
                'market_value': holding['market_value'],
                'percentage': holding['market_value'] / total_value
            }
            
            analysis['weight_analysis'][ticker] = {
                'target_weight': holding['target_weight'],
                'actual_weight': holding['actual_weight'],
                'difference': holding['actual_weight'] - holding['target_weight']
            }
        
        return analysis
    
    def calculate_rebalancing_trades(self, rebalancing_data: Dict, 
                                   transaction_cost_pct: float = 0.001) -> Dict:
        """
        Calculate specific trades needed for rebalancing with transaction costs.
        
        Args:
            rebalancing_data: Rebalancing requirements from DataIngestion
            transaction_cost_pct: Transaction cost as percentage of trade value
            
        Returns:
            Dictionary with trade recommendations
        """
        trades = {}
        total_cost = 0
        
        for ticker, data in rebalancing_data.items():
            if ticker == 'summary':
                continue
                
            difference = data['difference']
            shares_to_trade = data['shares_to_trade']
            
            if abs(shares_to_trade) < 0.1:  # Skip tiny trades
                continue
                
            trade_value = abs(difference)
            transaction_cost = trade_value * transaction_cost_pct
            total_cost += transaction_cost
            
            trades[ticker] = {
                'action': 'buy' if shares_to_trade > 0 else 'sell',
                'shares': abs(shares_to_trade),
                'trade_value': trade_value,
                'transaction_cost': transaction_cost,
                'current_weight': data['current_weight'],
                'target_weight': data['target_weight']
            }
        
        # Calculate net benefit of rebalancing
        rebalancing_benefit = self._estimate_rebalancing_benefit(rebalancing_data)
        
        trades['summary'] = {
            'total_trades': len([t for t in trades.values() if isinstance(t, dict) and 'action' in t]),
            'total_transaction_cost': total_cost,
            'estimated_benefit': rebalancing_benefit,
            'net_benefit': rebalancing_benefit - total_cost,
            'recommended': rebalancing_benefit > total_cost * 2  # Only if benefit > 2x cost
        }
        
        return trades
    
    def _estimate_rebalancing_benefit(self, rebalancing_data: Dict) -> float:
        """
        Estimate the benefit of rebalancing based on weight deviations.
        
        Args:
            rebalancing_data: Rebalancing data
            
        Returns:
            Estimated annual benefit in dollar terms
        """
        if 'summary' not in rebalancing_data:
            return 0
        
        total_value = sum(data['current_value'] for ticker, data in rebalancing_data.items() 
                         if ticker != 'summary')
        rebalancing_pct = rebalancing_data['summary']['rebalancing_percentage']
        
        # Rough estimate: benefit = portfolio_value * rebalancing_percentage * risk_reduction_factor
        # This is a simplified model; in practice, you'd use more sophisticated calculations
        risk_reduction_factor = 0.15  # Assume 15% of misallocation translates to risk reduction
        estimated_benefit = total_value * rebalancing_pct * risk_reduction_factor
        
        return estimated_benefit
    
    def discrete_allocation_recommendation(self, 
                                         optimized_weights: Dict[str, float],
                                         total_portfolio_value: float,
                                         current_prices: Dict[str, float]) -> Dict:
        """
        Calculate discrete share allocation for a given portfolio value.
        
        Args:
            optimized_weights: Target weights from optimization
            total_portfolio_value: Total value to allocate
            current_prices: Current prices for each asset
            
        Returns:
            Dictionary with discrete allocation recommendation
        """
        try:
            from pypfopt.discrete_allocation import DiscreteAllocation
            
            # Create price series for DiscreteAllocation
            prices_series = pd.Series(current_prices)
            
            da = DiscreteAllocation(optimized_weights, prices_series, 
                                  total_portfolio_value=total_portfolio_value)
            allocation, leftover = da.greedy_portfolio()
            
            # Calculate actual values and weights
            actual_allocation = {}
            total_allocated = 0
            
            for ticker, shares in allocation.items():
                value = shares * current_prices[ticker]
                actual_allocation[ticker] = {
                    'shares': shares,
                    'value': value,
                    'target_weight': optimized_weights[ticker],
                    'actual_weight': value / total_portfolio_value
                }
                total_allocated += value
            
            return {
                'allocation': actual_allocation,
                'leftover_cash': leftover,
                'total_allocated': total_allocated,
                'allocation_efficiency': total_allocated / total_portfolio_value
            }
            
        except ImportError:
            print("Warning: DiscreteAllocation requires pypfopt. Returning approximate allocation.")
            
            # Fallback: simple allocation
            allocation = {}
            leftover = total_portfolio_value
            
            for ticker, weight in optimized_weights.items():
                target_value = total_portfolio_value * weight
                shares = int(target_value / current_prices[ticker])
                actual_value = shares * current_prices[ticker]
                
                allocation[ticker] = {
                    'shares': shares,
                    'value': actual_value,
                    'target_weight': weight,
                    'actual_weight': actual_value / total_portfolio_value
                }
                leftover -= actual_value
            
            return {
                'allocation': allocation,
                'leftover_cash': leftover,
                'total_allocated': total_portfolio_value - leftover,
                'allocation_efficiency': (total_portfolio_value - leftover) / total_portfolio_value
            }
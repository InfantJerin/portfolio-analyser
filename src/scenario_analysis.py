"""
Scenario Analysis Module
Implements Bullish/Neutral/Bearish scenario analysis for portfolio forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class ScenarioAnalysis:
    """
    Scenario analysis for portfolio performance under different market conditions.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize ScenarioAnalysis.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.scenario_thresholds = config.get('scenario_thresholds', {})
        self.bullish_threshold = self.scenario_thresholds.get('bullish', 0.2)
        self.bearish_threshold = self.scenario_thresholds.get('bearish', -0.2)
        self.sentiment_shifts = self.scenario_thresholds.get('sentiment_shift', {
            'bullish': 0.3,
            'neutral': 0.0,
            'bearish': -0.3
        })
        
    def classify_market_regime(self, sentiment_score: float) -> str:
        """
        Classify market regime based on sentiment score.
        
        Args:
            sentiment_score: Current sentiment score
            
        Returns:
            Market regime ('bullish', 'neutral', 'bearish')
        """
        if sentiment_score >= self.bullish_threshold:
            return 'bullish'
        elif sentiment_score <= self.bearish_threshold:
            return 'bearish'
        else:
            return 'neutral'
    
    def generate_scenario_returns(self, 
                                base_returns: pd.Series,
                                scenarios: List[str] = None) -> Dict[str, pd.Series]:
        """
        Generate expected returns for different scenarios.
        
        Args:
            base_returns: Base expected returns
            scenarios: List of scenarios to generate
            
        Returns:
            Dictionary mapping scenario names to adjusted returns
        """
        if scenarios is None:
            scenarios = ['bullish', 'neutral', 'bearish']
        
        scenario_returns = {}
        
        for scenario in scenarios:
            adjustment_factor = self.sentiment_shifts.get(scenario, 0.0)
            adjusted_returns = base_returns * (1 + adjustment_factor)
            scenario_returns[scenario] = adjusted_returns
        
        return scenario_returns
    
    def portfolio_scenario_analysis(self,
                                  base_returns: pd.Series,
                                  covariance_matrix: pd.DataFrame,
                                  weights: pd.Series,
                                  scenarios: List[str] = None) -> Dict:
        """
        Analyze portfolio performance under different scenarios.
        
        Args:
            base_returns: Base expected returns
            covariance_matrix: Asset covariance matrix
            weights: Portfolio weights
            scenarios: List of scenarios to analyze
            
        Returns:
            Dictionary with scenario analysis results
        """
        if scenarios is None:
            scenarios = ['bullish', 'neutral', 'bearish']
        
        scenario_returns = self.generate_scenario_returns(base_returns, scenarios)
        results = {}
        
        for scenario, expected_returns in scenario_returns.items():
            # Portfolio return
            portfolio_return = (weights * expected_returns).sum()
            
            # Portfolio volatility (same covariance matrix for all scenarios)
            portfolio_variance = np.dot(weights.values, np.dot(covariance_matrix.values, weights.values))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Sharpe ratio
            risk_free_rate = self.config.get('risk_params', {}).get('risk_free_rate', 0.02)
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            
            # Asset-level analysis
            asset_contributions = weights * expected_returns
            top_contributors = asset_contributions.nlargest(3)
            bottom_contributors = asset_contributions.nsmallest(3)
            
            results[scenario] = {
                'portfolio_metrics': {
                    'expected_return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'risk_free_rate': risk_free_rate
                },
                'asset_returns': expected_returns.to_dict(),
                'asset_contributions': asset_contributions.to_dict(),
                'top_contributors': top_contributors.to_dict(),
                'bottom_contributors': bottom_contributors.to_dict(),
                'scenario_adjustment': self.sentiment_shifts.get(scenario, 0.0)
            }
        
        return results
    
    def monte_carlo_scenario_comparison(self,
                                      base_returns: pd.Series,
                                      covariance_matrix: pd.DataFrame,
                                      weights: pd.Series,
                                      monte_carlo_engine,
                                      initial_value: float = 100000) -> Dict:
        """
        Run Monte Carlo simulations for each scenario.
        
        Args:
            base_returns: Base expected returns
            covariance_matrix: Asset covariance matrix
            weights: Portfolio weights
            monte_carlo_engine: MonteCarloSimulation instance
            initial_value: Initial portfolio value
            
        Returns:
            Dictionary with Monte Carlo results for each scenario
        """
        scenarios = ['bullish', 'neutral', 'bearish']
        scenario_results = {}
        
        for scenario in scenarios:
            adjustment_factor = self.sentiment_shifts.get(scenario, 0.0)
            
            # Run Monte Carlo with sentiment adjustment
            mc_results = monte_carlo_engine.sentiment_adjusted_simulation(
                base_returns,
                covariance_matrix,
                weights,
                adjustment_factor,
                initial_value
            )
            
            scenario_results[scenario] = mc_results
        
        return scenario_results
    
    def scenario_stress_testing(self,
                               base_returns: pd.Series,
                               covariance_matrix: pd.DataFrame,
                               weights: pd.Series) -> Dict:
        """
        Perform stress testing under extreme scenarios.
        
        Args:
            base_returns: Base expected returns
            covariance_matrix: Asset covariance matrix
            weights: Portfolio weights
            
        Returns:
            Dictionary with stress test results
        """
        stress_scenarios = {
            'market_crash': -0.5,      # 50% drop in all returns
            'tech_bubble_burst': {'AAPL': -0.6, 'GOOGL': -0.6, 'MSFT': -0.6, 'AMZN': -0.5, 'TSLA': -0.7},
            'financial_crisis': {'JPM': -0.4, 'AAPL': -0.3, 'GOOGL': -0.3},
            'inflation_spike': -0.2,   # 20% drop due to inflation
            'extreme_volatility': None  # Double the volatility
        }
        
        stress_results = {}
        
        for scenario_name, adjustment in stress_scenarios.items():
            if scenario_name == 'extreme_volatility':
                # Double the covariance matrix
                stressed_cov = covariance_matrix * 2
                stressed_returns = base_returns
            elif isinstance(adjustment, dict):
                # Asset-specific adjustments
                stressed_returns = base_returns.copy()
                for asset, adj in adjustment.items():
                    if asset in stressed_returns.index:
                        stressed_returns[asset] *= (1 + adj)
                stressed_cov = covariance_matrix
            else:
                # Market-wide adjustment
                stressed_returns = base_returns * (1 + adjustment)
                stressed_cov = covariance_matrix
            
            # Calculate stressed portfolio metrics
            portfolio_return = (weights * stressed_returns).sum()
            portfolio_variance = np.dot(weights.values, np.dot(stressed_cov.values, weights.values))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            risk_free_rate = self.config.get('risk_params', {}).get('risk_free_rate', 0.02)
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            
            stress_results[scenario_name] = {
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'return_impact': portfolio_return - (weights * base_returns).sum(),
                'volatility_impact': portfolio_volatility - np.sqrt(np.dot(weights.values, np.dot(covariance_matrix.values, weights.values)))
            }
        
        return stress_results
    
    def historical_scenario_analysis(self,
                                   historical_returns: pd.DataFrame,
                                   historical_sentiment: pd.DataFrame,
                                   weights: pd.Series) -> Dict:
        """
        Analyze historical performance during different sentiment regimes.
        
        Args:
            historical_returns: Historical return data
            historical_sentiment: Historical sentiment data
            weights: Portfolio weights
            
        Returns:
            Dictionary with historical regime analysis
        """
        # Calculate historical portfolio returns
        portfolio_returns = (historical_returns * weights.values).sum(axis=1)
        
        # Classify each period by sentiment regime
        market_sentiment = historical_sentiment.get('market_sentiment', 
                                                   historical_sentiment.iloc[:, 0])
        
        regimes = []
        for sentiment in market_sentiment:
            regimes.append(self.classify_market_regime(sentiment))
        
        regime_series = pd.Series(regimes, index=market_sentiment.index)
        
        # Analyze performance by regime
        regime_results = {}
        
        for regime in ['bullish', 'neutral', 'bearish']:
            regime_mask = regime_series == regime
            
            if regime_mask.sum() == 0:
                continue
            
            regime_returns = portfolio_returns[regime_mask]
            regime_sentiment = market_sentiment[regime_mask]
            
            regime_results[regime] = {
                'num_periods': len(regime_returns),
                'avg_return': regime_returns.mean(),
                'volatility': regime_returns.std(),
                'avg_sentiment': regime_sentiment.mean(),
                'best_return': regime_returns.max(),
                'worst_return': regime_returns.min(),
                'positive_periods': (regime_returns > 0).sum(),
                'negative_periods': (regime_returns < 0).sum(),
                'sharpe_ratio': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0
            }
        
        return regime_results
    
    def regime_transition_analysis(self,
                                 historical_sentiment: pd.Series) -> Dict:
        """
        Analyze transitions between sentiment regimes.
        
        Args:
            historical_sentiment: Historical sentiment time series
            
        Returns:
            Dictionary with regime transition probabilities
        """
        # Classify regimes
        regimes = [self.classify_market_regime(sentiment) for sentiment in historical_sentiment]
        regime_series = pd.Series(regimes, index=historical_sentiment.index)
        
        # Calculate transition matrix
        transitions = {}
        unique_regimes = ['bullish', 'neutral', 'bearish']
        
        for from_regime in unique_regimes:
            transitions[from_regime] = {}
            for to_regime in unique_regimes:
                transitions[from_regime][to_regime] = 0
        
        # Count transitions
        for i in range(len(regime_series) - 1):
            from_regime = regime_series.iloc[i]
            to_regime = regime_series.iloc[i + 1]
            transitions[from_regime][to_regime] += 1
        
        # Convert to probabilities
        for from_regime in unique_regimes:
            total_transitions = sum(transitions[from_regime].values())
            if total_transitions > 0:
                for to_regime in unique_regimes:
                    transitions[from_regime][to_regime] /= total_transitions
        
        # Calculate regime persistence
        regime_lengths = {}
        current_regime = None
        current_length = 0
        
        for regime in regime_series:
            if regime == current_regime:
                current_length += 1
            else:
                if current_regime is not None:
                    if current_regime not in regime_lengths:
                        regime_lengths[current_regime] = []
                    regime_lengths[current_regime].append(current_length)
                current_regime = regime
                current_length = 1
        
        # Add final regime
        if current_regime is not None:
            if current_regime not in regime_lengths:
                regime_lengths[current_regime] = []
            regime_lengths[current_regime].append(current_length)
        
        # Calculate average regime duration
        avg_regime_duration = {}
        for regime, lengths in regime_lengths.items():
            avg_regime_duration[regime] = np.mean(lengths) if lengths else 0
        
        return {
            'transition_matrix': transitions,
            'average_regime_duration': avg_regime_duration,
            'regime_distribution': regime_series.value_counts(normalize=True).to_dict()
        }
    
    def scenario_summary_report(self, scenario_results: Dict) -> pd.DataFrame:
        """
        Create a summary report of scenario analysis results.
        
        Args:
            scenario_results: Results from portfolio scenario analysis
            
        Returns:
            DataFrame with summary statistics
        """
        summary_data = []
        
        for scenario, results in scenario_results.items():
            metrics = results['portfolio_metrics']
            row = {
                'scenario': scenario,
                'expected_return': metrics['expected_return'],
                'volatility': metrics['volatility'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'return_vs_neutral': 0,  # Will be calculated below
                'scenario_adjustment': results['scenario_adjustment']
            }
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Calculate return difference vs neutral scenario
        if 'neutral' in scenario_results:
            neutral_return = scenario_results['neutral']['portfolio_metrics']['expected_return']
            summary_df['return_vs_neutral'] = summary_df['expected_return'] - neutral_return
        
        return summary_df
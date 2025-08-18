"""
Portfolio Optimization Module

This module provides comprehensive portfolio optimization capabilities using
Modern Portfolio Theory, factor models, and advanced optimization techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from scipy.optimize import minimize, differential_evolution
from scipy import stats
import warnings
from dataclasses import dataclass, field
from enum import Enum

warnings.filterwarnings('ignore')


class OptimizationObjective(Enum):
    """Portfolio optimization objectives."""
    MAX_SHARPE = "max_sharpe"
    MIN_VOLATILITY = "min_volatility"
    MAX_RETURN = "max_return"
    MIN_CVAR = "min_cvar"
    MAX_UTILITY = "max_utility"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"


@dataclass
class OptimizationConstraints:
    """Container for optimization constraints."""
    min_weights: Dict[str, float] = field(default_factory=dict)
    max_weights: Dict[str, float] = field(default_factory=dict)
    target_return: Optional[float] = None
    target_volatility: Optional[float] = None
    max_concentration: Optional[float] = None
    min_concentration: Optional[float] = None
    sector_constraints: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    turnover_constraint: Optional[float] = None
    long_only: bool = True


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    objective_value: float
    success: bool
    message: str
    iterations: int
    method: str
    constraints_satisfied: bool


class PortfolioOptimizer:
    """
    Advanced Portfolio Optimization System.
    
    This class provides:
    - Modern Portfolio Theory optimization
    - Multi-objective optimization
    - Risk budgeting and parity approaches
    - Factor-based optimization
    - Black-Litterman model integration
    - Transaction cost optimization
    - Robust optimization techniques
    """
    
    def __init__(self, 
                 expected_returns: pd.Series,
                 covariance_matrix: pd.DataFrame,
                 risk_free_rate: float = 0.02):
        """
        Initialize Portfolio Optimizer.
        
        Args:
            expected_returns: Expected returns for assets
            covariance_matrix: Covariance matrix of asset returns
            risk_free_rate: Risk-free rate for Sharpe ratio calculations
        """
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.risk_free_rate = risk_free_rate
        
        # Validate inputs
        self._validate_inputs()
        
        # Optimization settings
        self.max_iterations = 1000
        self.tolerance = 1e-8
        self.method = 'SLSQP'
        
        # Factor model data (optional)
        self.factor_returns: Optional[pd.DataFrame] = None
        self.factor_loadings: Optional[pd.DataFrame] = None
        
        # Black-Litterman data (optional)
        self.market_cap_weights: Optional[pd.Series] = None
        self.investor_views: Optional[Dict] = None
        
        # Transaction cost model (optional)
        self.transaction_costs: Optional[Dict[str, float]] = None
    
    def _validate_inputs(self):
        """Validate input data consistency."""
        if not self.expected_returns.index.equals(self.covariance_matrix.index):
            raise ValueError("Expected returns and covariance matrix indices must match")
        
        if not self.covariance_matrix.index.equals(self.covariance_matrix.columns):
            raise ValueError("Covariance matrix must be square")
        
        # Check for positive semi-definiteness
        eigenvalues = np.linalg.eigvals(self.covariance_matrix.values)
        if np.any(eigenvalues < -1e-8):
            warnings.warn("Covariance matrix is not positive semi-definite")
    
    def optimize(self, 
                objective: OptimizationObjective,
                constraints: OptimizationConstraints = None,
                initial_weights: Dict[str, float] = None) -> OptimizationResult:
        """
        Optimize portfolio based on specified objective and constraints.
        
        Args:
            objective: Optimization objective
            constraints: Portfolio constraints
            initial_weights: Initial weight guess
            
        Returns:
            OptimizationResult object
        """
        if constraints is None:
            constraints = OptimizationConstraints()
        
        # Set up optimization variables
        assets = list(self.expected_returns.index)
        n_assets = len(assets)
        
        # Initial weights
        if initial_weights is None:
            x0 = np.ones(n_assets) / n_assets
        else:
            x0 = np.array([initial_weights.get(asset, 1/n_assets) for asset in assets])
        
        # Set up bounds
        bounds = self._create_bounds(assets, constraints)
        
        # Set up constraints
        constraint_list = self._create_constraints(assets, constraints)
        
        # Define objective function
        objective_func = self._get_objective_function(objective, constraints)
        
        # Run optimization
        try:
            if objective == OptimizationObjective.RISK_PARITY:
                # Use specialized risk parity algorithm
                result = self._optimize_risk_parity(constraints)
            elif objective == OptimizationObjective.EQUAL_WEIGHT:
                # Simple equal weight
                result = self._equal_weight_portfolio()
            else:
                # Standard optimization
                opt_result = minimize(
                    objective_func,
                    x0,
                    method=self.method,
                    bounds=bounds,
                    constraints=constraint_list,
                    options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
                )
                
                result = self._process_optimization_result(opt_result, assets, objective)
        
        except Exception as e:
            return OptimizationResult(
                weights={asset: 1/n_assets for asset in assets},
                expected_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                objective_value=0.0,
                success=False,
                message=f"Optimization failed: {str(e)}",
                iterations=0,
                method=self.method,
                constraints_satisfied=False
            )
        
        return result
    
    def _create_bounds(self, assets: List[str], constraints: OptimizationConstraints) -> List[Tuple]:
        """Create bounds for optimization variables."""
        bounds = []
        
        for asset in assets:
            min_weight = constraints.min_weights.get(asset, 0.0 if constraints.long_only else -1.0)
            max_weight = constraints.max_weights.get(asset, 1.0)
            bounds.append((min_weight, max_weight))
        
        return bounds
    
    def _create_constraints(self, assets: List[str], constraints: OptimizationConstraints) -> List[Dict]:
        """Create constraint list for optimization."""
        constraint_list = []
        
        # Weights sum to 1
        constraint_list.append({
            'type': 'eq',
            'fun': lambda x: np.sum(x) - 1.0
        })
        
        # Target return constraint
        if constraints.target_return is not None:
            constraint_list.append({
                'type': 'eq',
                'fun': lambda x: np.dot(x, self.expected_returns.values) - constraints.target_return
            })
        
        # Target volatility constraint
        if constraints.target_volatility is not None:
            constraint_list.append({
                'type': 'eq',
                'fun': lambda x: np.sqrt(np.dot(x, np.dot(self.covariance_matrix.values, x))) - constraints.target_volatility
            })
        
        # Maximum concentration constraint
        if constraints.max_concentration is not None:
            for i in range(len(assets)):
                constraint_list.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i: constraints.max_concentration - x[i]
                })
        
        # Minimum concentration constraint
        if constraints.min_concentration is not None:
            for i in range(len(assets)):
                constraint_list.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i: x[i] - constraints.min_concentration
                })
        
        return constraint_list
    
    def _get_objective_function(self, 
                               objective: OptimizationObjective, 
                               constraints: OptimizationConstraints) -> Callable:
        """Get objective function based on optimization goal."""
        
        if objective == OptimizationObjective.MAX_SHARPE:
            def sharpe_objective(weights):
                portfolio_return = np.dot(weights, self.expected_returns.values)
                portfolio_volatility = np.sqrt(np.dot(weights, np.dot(self.covariance_matrix.values, weights)))
                if portfolio_volatility == 0:
                    return -np.inf
                return -(portfolio_return - self.risk_free_rate) / portfolio_volatility
            return sharpe_objective
        
        elif objective == OptimizationObjective.MIN_VOLATILITY:
            def volatility_objective(weights):
                return np.sqrt(np.dot(weights, np.dot(self.covariance_matrix.values, weights)))
            return volatility_objective
        
        elif objective == OptimizationObjective.MAX_RETURN:
            def return_objective(weights):
                return -np.dot(weights, self.expected_returns.values)
            return return_objective
        
        elif objective == OptimizationObjective.MIN_CVAR:
            def cvar_objective(weights):
                # Simplified CVaR approximation using normal distribution
                portfolio_return = np.dot(weights, self.expected_returns.values)
                portfolio_volatility = np.sqrt(np.dot(weights, np.dot(self.covariance_matrix.values, weights)))
                z_score = stats.norm.ppf(0.05)  # 5% VaR
                cvar = portfolio_return + z_score * portfolio_volatility
                return -cvar  # Minimize negative CVaR (maximize CVaR)
            return cvar_objective
        
        elif objective == OptimizationObjective.MAX_UTILITY:
            # Default utility function: return - 0.5 * risk_aversion * variance
            risk_aversion = 5.0  # Default risk aversion
            def utility_objective(weights):
                portfolio_return = np.dot(weights, self.expected_returns.values)
                portfolio_variance = np.dot(weights, np.dot(self.covariance_matrix.values, weights))
                utility = portfolio_return - 0.5 * risk_aversion * portfolio_variance
                return -utility  # Minimize negative utility
            return utility_objective
        
        else:
            raise ValueError(f"Unknown objective: {objective}")
    
    def _optimize_risk_parity(self, constraints: OptimizationConstraints) -> OptimizationResult:
        """Optimize for risk parity portfolio."""
        assets = list(self.expected_returns.index)
        n_assets = len(assets)
        
        def risk_parity_objective(weights):
            # Calculate risk contributions
            portfolio_volatility = np.sqrt(np.dot(weights, np.dot(self.covariance_matrix.values, weights)))
            marginal_contrib = np.dot(self.covariance_matrix.values, weights)
            risk_contrib = weights * marginal_contrib / portfolio_volatility
            
            # Target is equal risk contribution (1/n for each asset)
            target_contrib = np.ones(n_assets) / n_assets
            
            # Minimize sum of squared deviations from equal risk contribution
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Bounds and constraints
        bounds = self._create_bounds(assets, constraints)
        constraint_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        
        # Optimize
        opt_result = minimize(
            risk_parity_objective,
            x0,
            method=self.method,
            bounds=bounds,
            constraints=constraint_list,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )
        
        return self._process_optimization_result(opt_result, assets, OptimizationObjective.RISK_PARITY)
    
    def _equal_weight_portfolio(self) -> OptimizationResult:
        """Create equal weight portfolio."""
        assets = list(self.expected_returns.index)
        n_assets = len(assets)
        weights = {asset: 1/n_assets for asset in assets}
        
        # Calculate metrics
        weight_array = np.array(list(weights.values()))
        expected_return = np.dot(weight_array, self.expected_returns.values)
        volatility = np.sqrt(np.dot(weight_array, np.dot(self.covariance_matrix.values, weight_array)))
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            objective_value=0.0,
            success=True,
            message="Equal weight portfolio created",
            iterations=0,
            method="equal_weight",
            constraints_satisfied=True
        )
    
    def _process_optimization_result(self, 
                                   opt_result, 
                                   assets: List[str], 
                                   objective: OptimizationObjective) -> OptimizationResult:
        """Process scipy optimization result into OptimizationResult."""
        if opt_result.success:
            weights = {asset: float(weight) for asset, weight in zip(assets, opt_result.x)}
            
            # Calculate portfolio metrics
            weight_array = opt_result.x
            expected_return = np.dot(weight_array, self.expected_returns.values)
            volatility = np.sqrt(np.dot(weight_array, np.dot(self.covariance_matrix.values, weight_array)))
            sharpe_ratio = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
            
            return OptimizationResult(
                weights=weights,
                expected_return=expected_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                objective_value=opt_result.fun,
                success=True,
                message=opt_result.message,
                iterations=opt_result.nit if hasattr(opt_result, 'nit') else 0,
                method=self.method,
                constraints_satisfied=True
            )
        else:
            # Return equal weight as fallback
            return self._equal_weight_portfolio()
    
    def efficient_frontier(self, 
                          num_portfolios: int = 100,
                          constraints: OptimizationConstraints = None) -> pd.DataFrame:
        """
        Generate efficient frontier.
        
        Args:
            num_portfolios: Number of portfolios on the frontier
            constraints: Portfolio constraints
            
        Returns:
            DataFrame with efficient frontier data
        """
        if constraints is None:
            constraints = OptimizationConstraints()
        
        # Calculate return range
        min_return = self.expected_returns.min()
        max_return = self.expected_returns.max()
        target_returns = np.linspace(min_return, max_return, num_portfolios)
        
        frontier_results = []
        
        for target_return in target_returns:
            # Set target return constraint
            target_constraints = OptimizationConstraints(
                min_weights=constraints.min_weights,
                max_weights=constraints.max_weights,
                target_return=target_return,
                max_concentration=constraints.max_concentration,
                long_only=constraints.long_only
            )
            
            # Optimize for minimum volatility at target return
            try:
                result = self.optimize(OptimizationObjective.MIN_VOLATILITY, target_constraints)
                
                if result.success:
                    frontier_results.append({
                        'return': result.expected_return,
                        'volatility': result.volatility,
                        'sharpe_ratio': result.sharpe_ratio,
                        'weights': result.weights
                    })
            except:
                continue
        
        return pd.DataFrame(frontier_results)
    
    def calculate_maximum_diversification_portfolio(self, 
                                                  constraints: OptimizationConstraints = None) -> OptimizationResult:
        """
        Calculate maximum diversification portfolio.
        
        The diversification ratio is the ratio of the weighted average volatility
        to the portfolio volatility.
        """
        if constraints is None:
            constraints = OptimizationConstraints()
        
        assets = list(self.expected_returns.index)
        individual_volatilities = np.sqrt(np.diag(self.covariance_matrix.values))
        
        def diversification_objective(weights):
            weighted_avg_vol = np.dot(weights, individual_volatilities)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.covariance_matrix.values, weights)))
            if portfolio_vol == 0:
                return -np.inf
            return -weighted_avg_vol / portfolio_vol  # Minimize negative diversification ratio
        
        # Initial guess
        x0 = np.ones(len(assets)) / len(assets)
        
        # Bounds and constraints
        bounds = self._create_bounds(assets, constraints)
        constraint_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
        
        # Optimize
        opt_result = minimize(
            diversification_objective,
            x0,
            method=self.method,
            bounds=bounds,
            constraints=constraint_list,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )
        
        return self._process_optimization_result(opt_result, assets, OptimizationObjective.MAX_SHARPE)
    
    def black_litterman_optimization(self, 
                                   market_cap_weights: Dict[str, float],
                                   investor_views: Dict[str, Dict],
                                   tau: float = 0.025,
                                   confidence_scaling: float = 1.0) -> Dict[str, float]:
        """
        Implement Black-Litterman model for expected returns.
        
        Args:
            market_cap_weights: Market capitalization weights
            investor_views: Dictionary with investor views
            tau: Scaling factor for uncertainty of prior
            confidence_scaling: Scaling factor for confidence in views
            
        Returns:
            Black-Litterman expected returns
        """
        # Convert to arrays
        assets = list(self.expected_returns.index)
        w_market = np.array([market_cap_weights.get(asset, 0) for asset in assets])
        
        # Calculate implied returns (reverse optimization)
        # Assuming risk aversion coefficient of 3
        risk_aversion = 3.0
        pi = risk_aversion * np.dot(self.covariance_matrix.values, w_market)
        
        # Process investor views
        if not investor_views:
            return dict(zip(assets, pi))
        
        # Create P matrix (picking matrix) and Q vector (view returns)
        P = []
        Q = []
        omega_diag = []
        
        for view_name, view_data in investor_views.items():
            if 'assets' in view_data and 'expected_return' in view_data:
                p_row = np.zeros(len(assets))
                for asset, weight in view_data['assets'].items():
                    if asset in assets:
                        asset_idx = assets.index(asset)
                        p_row[asset_idx] = weight
                
                P.append(p_row)
                Q.append(view_data['expected_return'])
                
                # Omega (uncertainty matrix) - diagonal element
                confidence = view_data.get('confidence', 1.0) * confidence_scaling
                view_variance = np.dot(p_row, np.dot(self.covariance_matrix.values, p_row))
                omega_diag.append(view_variance / confidence)
        
        if not P:
            return dict(zip(assets, pi))
        
        P = np.array(P)
        Q = np.array(Q)
        Omega = np.diag(omega_diag)
        
        # Black-Litterman formula
        tau_cov = tau * self.covariance_matrix.values
        
        # Calculate new expected returns
        term1 = np.linalg.inv(tau_cov)
        term2 = np.dot(P.T, np.dot(np.linalg.inv(Omega), P))
        term3 = np.dot(P.T, np.dot(np.linalg.inv(Omega), Q))
        term4 = np.dot(np.linalg.inv(tau_cov), pi)
        
        mu_bl = np.dot(np.linalg.inv(term1 + term2), term4 + term3)
        
        return dict(zip(assets, mu_bl))
    
    def robust_optimization(self, 
                          uncertainty_sets: Dict[str, float],
                          constraints: OptimizationConstraints = None) -> OptimizationResult:
        """
        Robust optimization considering parameter uncertainty.
        
        Args:
            uncertainty_sets: Dictionary with uncertainty bounds for returns
            constraints: Portfolio constraints
            
        Returns:
            Robust optimization result
        """
        if constraints is None:
            constraints = OptimizationConstraints()
        
        assets = list(self.expected_returns.index)
        n_assets = len(assets)
        
        def robust_objective(weights):
            # Worst-case return considering uncertainty
            worst_case_return = 0
            for i, asset in enumerate(assets):
                base_return = self.expected_returns[asset]
                uncertainty = uncertainty_sets.get(asset, 0.0)
                worst_case_asset_return = base_return - uncertainty
                worst_case_return += weights[i] * worst_case_asset_return
            
            # Portfolio volatility (unchanged)
            portfolio_volatility = np.sqrt(np.dot(weights, np.dot(self.covariance_matrix.values, weights)))
            
            # Robust Sharpe ratio (worst-case)
            if portfolio_volatility == 0:
                return -np.inf
            return -(worst_case_return - self.risk_free_rate) / portfolio_volatility
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Bounds and constraints
        bounds = self._create_bounds(assets, constraints)
        constraint_list = self._create_constraints(assets, constraints)
        
        # Optimize
        opt_result = minimize(
            robust_objective,
            x0,
            method=self.method,
            bounds=bounds,
            constraints=constraint_list,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )
        
        return self._process_optimization_result(opt_result, assets, OptimizationObjective.MAX_SHARPE)
    
    def multi_period_optimization(self, 
                                forecast_horizons: List[int],
                                return_forecasts: Dict[int, pd.Series],
                                rebalancing_costs: Dict[str, float] = None,
                                constraints: OptimizationConstraints = None) -> Dict[int, OptimizationResult]:
        """
        Multi-period optimization considering rebalancing costs.
        
        Args:
            forecast_horizons: List of forecast periods
            return_forecasts: Dictionary mapping periods to return forecasts
            rebalancing_costs: Transaction costs for rebalancing
            constraints: Portfolio constraints
            
        Returns:
            Dictionary mapping periods to optimization results
        """
        if constraints is None:
            constraints = OptimizationConstraints()
        
        if rebalancing_costs is None:
            rebalancing_costs = {}
        
        results = {}
        current_weights = None
        
        for period in sorted(forecast_horizons):
            if period not in return_forecasts:
                continue
            
            # Update expected returns for this period
            period_returns = return_forecasts[period]
            
            # Create temporary optimizer for this period
            temp_optimizer = PortfolioOptimizer(
                expected_returns=period_returns,
                covariance_matrix=self.covariance_matrix,
                risk_free_rate=self.risk_free_rate
            )
            
            # If we have previous weights, consider transaction costs
            if current_weights is not None:
                adjusted_constraints = self._adjust_constraints_for_transaction_costs(
                    constraints, current_weights, rebalancing_costs
                )
            else:
                adjusted_constraints = constraints
            
            # Optimize for this period
            result = temp_optimizer.optimize(
                OptimizationObjective.MAX_SHARPE,
                adjusted_constraints
            )
            
            results[period] = result
            current_weights = result.weights
        
        return results
    
    def _adjust_constraints_for_transaction_costs(self, 
                                                constraints: OptimizationConstraints,
                                                current_weights: Dict[str, float],
                                                transaction_costs: Dict[str, float]) -> OptimizationConstraints:
        """Adjust constraints to account for transaction costs."""
        # This is a simplified implementation
        # In practice, you'd want to incorporate transaction costs directly into the objective function
        adjusted_constraints = OptimizationConstraints(
            min_weights=constraints.min_weights.copy(),
            max_weights=constraints.max_weights.copy(),
            target_return=constraints.target_return,
            target_volatility=constraints.target_volatility,
            max_concentration=constraints.max_concentration,
            long_only=constraints.long_only
        )
        
        # Add turnover constraint based on transaction costs
        if transaction_costs:
            max_turnover = 0.1  # 10% maximum turnover
            adjusted_constraints.turnover_constraint = max_turnover
        
        return adjusted_constraints
    
    def factor_model_optimization(self, 
                                factor_returns: pd.DataFrame,
                                factor_loadings: pd.DataFrame,
                                factor_constraints: Dict[str, Tuple[float, float]] = None,
                                constraints: OptimizationConstraints = None) -> OptimizationResult:
        """
        Optimize portfolio using factor model constraints.
        
        Args:
            factor_returns: Historical factor returns
            factor_loadings: Asset loadings on factors
            factor_constraints: Min/max exposure to each factor
            constraints: Portfolio constraints
            
        Returns:
            Factor-aware optimization result
        """
        if constraints is None:
            constraints = OptimizationConstraints()
        
        assets = list(self.expected_returns.index)
        
        # Standard optimization setup
        def objective_func(weights):
            portfolio_return = np.dot(weights, self.expected_returns.values)
            portfolio_volatility = np.sqrt(np.dot(weights, np.dot(self.covariance_matrix.values, weights)))
            if portfolio_volatility == 0:
                return -np.inf
            return -(portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        # Create factor exposure constraints
        constraint_list = self._create_constraints(assets, constraints)
        
        if factor_constraints:
            for factor_name, (min_exp, max_exp) in factor_constraints.items():
                if factor_name in factor_loadings.columns:
                    factor_loadings_vector = factor_loadings[factor_name].values
                    
                    # Minimum exposure constraint
                    constraint_list.append({
                        'type': 'ineq',
                        'fun': lambda x, loadings=factor_loadings_vector: np.dot(x, loadings) - min_exp
                    })
                    
                    # Maximum exposure constraint
                    constraint_list.append({
                        'type': 'ineq',
                        'fun': lambda x, loadings=factor_loadings_vector: max_exp - np.dot(x, loadings)
                    })
        
        # Initial guess and bounds
        x0 = np.ones(len(assets)) / len(assets)
        bounds = self._create_bounds(assets, constraints)
        
        # Optimize
        opt_result = minimize(
            objective_func,
            x0,
            method=self.method,
            bounds=bounds,
            constraints=constraint_list,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )
        
        return self._process_optimization_result(opt_result, assets, OptimizationObjective.MAX_SHARPE)
    
    def generate_optimization_report(self, 
                                   result: OptimizationResult,
                                   benchmark_weights: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Generate comprehensive optimization report.
        
        Args:
            result: Optimization result
            benchmark_weights: Benchmark weights for comparison
            
        Returns:
            Detailed optimization report
        """
        report = {
            'optimization_summary': {
                'success': result.success,
                'method': result.method,
                'iterations': result.iterations,
                'message': result.message,
                'constraints_satisfied': result.constraints_satisfied
            },
            'portfolio_metrics': {
                'expected_return': result.expected_return,
                'expected_return_annualized': result.expected_return * 252,
                'volatility': result.volatility,
                'volatility_annualized': result.volatility * np.sqrt(252),
                'sharpe_ratio': result.sharpe_ratio,
                'sharpe_ratio_annualized': result.sharpe_ratio * np.sqrt(252)
            },
            'weights': result.weights,
            'weight_statistics': {
                'number_of_positions': len([w for w in result.weights.values() if w > 0.001]),
                'largest_position': max(result.weights.values()),
                'smallest_position': min([w for w in result.weights.values() if w > 0]),
                'concentration_top_5': sum(sorted(result.weights.values(), reverse=True)[:5]),
                'effective_number_of_positions': 1 / sum(w**2 for w in result.weights.values() if w > 0)
            }
        }
        
        # Add benchmark comparison if provided
        if benchmark_weights:
            weight_diff = {asset: result.weights.get(asset, 0) - benchmark_weights.get(asset, 0)
                          for asset in set(list(result.weights.keys()) + list(benchmark_weights.keys()))}
            
            total_turnover = sum(abs(diff) for diff in weight_diff.values()) / 2
            
            report['benchmark_comparison'] = {
                'benchmark_weights': benchmark_weights,
                'weight_differences': weight_diff,
                'total_turnover': total_turnover,
                'tracking_error_estimate': np.sqrt(np.sum([diff**2 for diff in weight_diff.values()]))
            }
        
        return report
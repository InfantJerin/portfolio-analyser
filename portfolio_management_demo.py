#!/usr/bin/env python3
"""
Portfolio Management Package Demo

This comprehensive demo showcases all the features of the portfolio management package:
- Portfolio creation and management
- Position tracking with cost basis
- Transaction recording and audit trail
- Risk analysis (VaR, CVaR, stress testing)
- Portfolio optimization using Modern Portfolio Theory
- Performance metrics and reporting

Run this file to see the portfolio management system in action.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from decimal import Decimal
from pprint import pprint

# Add the portfolio_management package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ''))

# Import portfolio management components
from portfolio_management.core.portfolio import Portfolio, PortfolioType
from portfolio_management.core.position import Position, PositionType
from portfolio_management.core.transaction import Transaction, TransactionType, TransactionFees
from portfolio_management.analytics.risk_manager import RiskManager
from portfolio_management.analytics.optimizer import PortfolioOptimizer, OptimizationObjective, OptimizationConstraints


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def generate_sample_data():
    """Generate sample market data for demonstration."""
    # Sample assets
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    # Generate random historical returns (252 trading days)
    np.random.seed(42)  # For reproducible results
    returns_data = {}
    
    base_returns = {
        'AAPL': 0.001,
        'GOOGL': 0.0008,
        'MSFT': 0.0012,
        'AMZN': 0.0006,
        'TSLA': 0.002
    }
    
    volatilities = {
        'AAPL': 0.025,
        'GOOGL': 0.028,
        'MSFT': 0.023,
        'AMZN': 0.030,
        'TSLA': 0.045
    }
    
    for asset in assets:
        returns_data[asset] = np.random.normal(
            base_returns[asset], 
            volatilities[asset], 
            252
        )
    
    returns_df = pd.DataFrame(returns_data)
    
    # Generate sample prices
    initial_prices = {
        'AAPL': 150.0,
        'GOOGL': 2500.0,
        'MSFT': 300.0,
        'AMZN': 3200.0,
        'TSLA': 800.0
    }
    
    current_prices = {}
    for asset in assets:
        # Simulate price movement
        price_change = np.random.normal(0.05, 0.15)  # 5% average return with 15% volatility
        current_prices[asset] = Decimal(str(initial_prices[asset] * (1 + price_change)))
    
    return returns_df, current_prices, initial_prices


def demo_portfolio_creation():
    """Demonstrate portfolio creation and basic operations."""
    print_section("1. Portfolio Creation and Basic Operations")
    
    # Create a new portfolio
    portfolio = Portfolio(
        name="Demo Growth Portfolio",
        portfolio_type=PortfolioType.GROWTH,
        initial_cash=Decimal('100000'),  # $100,000 initial cash
        benchmark_symbol="SPY"
    )
    
    print(f"Created portfolio: {portfolio.name}")
    print(f"Portfolio ID: {portfolio.portfolio_id}")
    print(f"Initial cash: ${portfolio.get_cash_balance():,.2f}")
    print(f"Portfolio type: {portfolio.portfolio_type}")
    
    return portfolio


def demo_transactions_and_positions(portfolio, current_prices):
    """Demonstrate buying/selling shares and position tracking."""
    print_section("2. Transactions and Position Management")
    
    # Buy some shares
    transactions = []
    
    # Buy AAPL
    fees = TransactionFees(commission=Decimal('9.99'))
    txn_id = portfolio.buy_shares(
        symbol='AAPL',
        shares=Decimal('100'),
        price=current_prices['AAPL'],
        fees=fees
    )
    transactions.append(txn_id)
    print(f"Bought 100 shares of AAPL at ${current_prices['AAPL']}")
    
    # Buy GOOGL
    txn_id = portfolio.buy_shares(
        symbol='GOOGL',
        shares=Decimal('20'),
        price=current_prices['GOOGL'],
        fees=TransactionFees(commission=Decimal('9.99'))
    )
    transactions.append(txn_id)
    print(f"Bought 20 shares of GOOGL at ${current_prices['GOOGL']}")
    
    # Buy MSFT
    txn_id = portfolio.buy_shares(
        symbol='MSFT',
        shares=Decimal('50'),
        price=current_prices['MSFT'],
        fees=TransactionFees(commission=Decimal('9.99'))
    )
    transactions.append(txn_id)
    print(f"Bought 50 shares of MSFT at ${current_prices['MSFT']}")
    
    # Update market prices (simulate price changes)
    new_prices = {}
    for symbol, price in current_prices.items():
        change_pct = np.random.normal(0.02, 0.05)  # 2% average gain, 5% volatility
        new_prices[symbol] = price * Decimal(str(1 + change_pct))
    
    portfolio.update_market_prices(new_prices)
    print(f"\nUpdated market prices:")
    for symbol, price in new_prices.items():
        if symbol in portfolio.positions:
            print(f"  {symbol}: ${price:.2f}")
    
    # Show portfolio summary
    print(f"\nPortfolio Summary:")
    summary = portfolio.get_portfolio_summary()
    print(f"  Total Value: ${summary['total_value']:,.2f}")
    print(f"  Market Value: ${summary['market_value']:,.2f}")
    print(f"  Cash Balance: ${summary['cash_balance']:,.2f}")
    print(f"  Total P&L: ${summary['total_pnl']:,.2f}")
    print(f"  Return %: {summary['total_return_pct']:.2f}%")
    
    # Show individual positions
    print(f"\nPositions:")
    for symbol, position in portfolio.get_all_positions().items():
        pos_summary = position.get_position_summary()
        print(f"  {symbol}: {pos_summary['total_shares']} shares, "
              f"Value: ${pos_summary['market_value']:,.2f}, "
              f"P&L: ${pos_summary['total_pnl']:,.2f}")
    
    return new_prices


def demo_advanced_position_features(portfolio):
    """Demonstrate advanced position features like tax lots and P&L tracking."""
    print_section("3. Advanced Position Features")
    
    # Get AAPL position and show tax lot details
    aapl_position = portfolio.get_position('AAPL')
    if aapl_position:
        print("AAPL Tax Lot Details:")
        tax_lots = aapl_position.get_tax_lot_details()
        for lot in tax_lots:
            print(f"  Lot {lot['lot_id'][:8]}...: {lot['shares']} shares, "
                  f"Cost: ${lot['cost_basis_per_share']:.2f}, "
                  f"Current: ${lot['current_price']:.2f}, "
                  f"P&L: ${lot['unrealized_pnl']:.2f}")
    
    # Simulate selling some shares
    if aapl_position and aapl_position.total_shares >= 30:
        sell_txn = portfolio.sell_shares(
            symbol='AAPL',
            shares=Decimal('30'),
            price=aapl_position.current_price * Decimal('1.05'),  # Sell at 5% higher price
            fees=TransactionFees(commission=Decimal('9.99'))
        )
        
        print(f"\nSold 30 shares of AAPL")
        
        # Show updated position
        updated_summary = aapl_position.get_position_summary()
        print(f"Updated AAPL position:")
        print(f"  Remaining shares: {updated_summary['total_shares']}")
        print(f"  Realized P&L: ${updated_summary['realized_pnl']:.2f}")
        print(f"  Unrealized P&L: ${updated_summary['unrealized_pnl']:.2f}")


def demo_dividend_tracking(portfolio):
    """Demonstrate dividend recording and tracking."""
    print_section("4. Dividend Tracking")
    
    # Record dividends for AAPL and MSFT
    dividend_transactions = []
    
    # AAPL dividend
    aapl_position = portfolio.get_position('AAPL')
    if aapl_position:
        dividend_txn = portfolio.record_dividend(
            symbol='AAPL',
            dividend_per_share=Decimal('0.88'),
            ex_date=date.today() - timedelta(days=5),
            pay_date=date.today()
        )
        dividend_transactions.append(dividend_txn)
        total_dividend = aapl_position.total_shares * Decimal('0.88')
        print(f"Recorded AAPL dividend: ${total_dividend:.2f}")
    
    # MSFT dividend
    msft_position = portfolio.get_position('MSFT')
    if msft_position:
        dividend_txn = portfolio.record_dividend(
            symbol='MSFT',
            dividend_per_share=Decimal('0.68'),
            ex_date=date.today() - timedelta(days=3),
            pay_date=date.today()
        )
        dividend_transactions.append(dividend_txn)
        total_dividend = msft_position.total_shares * Decimal('0.68')
        print(f"Recorded MSFT dividend: ${total_dividend:.2f}")
    
    # Show updated cash balance
    print(f"Updated cash balance: ${portfolio.get_cash_balance():,.2f}")
    print(f"Total dividends received: ${portfolio.total_dividends:.2f}")


def demo_risk_analysis(portfolio, returns_data):
    """Demonstrate comprehensive risk analysis."""
    print_section("5. Risk Analysis")
    
    # Create risk manager
    risk_manager = RiskManager(confidence_levels=[0.95, 0.99])
    risk_manager.load_returns_data(returns_data)
    
    # Get current portfolio weights
    weights = portfolio.get_position_weights()
    # Remove CASH from weights for risk analysis and only include assets in returns data
    risk_weights = {k: float(v) for k, v in weights.items() if k != 'CASH' and k in returns_data.columns}
    
    if not risk_weights:
        print("No positions available for risk analysis")
        return
    
    # Normalize weights to sum to 1
    total_weight = sum(risk_weights.values())
    if total_weight > 0:
        risk_weights = {k: v/total_weight for k, v in risk_weights.items()}
    
    # Filter returns data to only include assets we have positions in
    filtered_returns = returns_data[list(risk_weights.keys())]
    risk_manager.load_returns_data(filtered_returns)
    
    print(f"Portfolio weights for risk analysis: {risk_weights}")
    
    # Calculate VaR
    var_results = risk_manager.calculate_portfolio_var(risk_weights, method="historical")
    print(f"\nValue at Risk (VaR):")
    for var_level, var_value in var_results.items():
        print(f"  {var_level}: {var_value:.4f} ({var_value*100:.2f}%)")
    
    # Calculate CVaR
    cvar_results = risk_manager.calculate_cvar(risk_weights)
    print(f"\nConditional Value at Risk (CVaR):")
    for cvar_level, cvar_value in cvar_results.items():
        print(f"  {cvar_level}: {cvar_value:.4f} ({cvar_value*100:.2f}%)")
    
    # Portfolio volatility
    volatility = risk_manager.calculate_portfolio_volatility(risk_weights)
    print(f"\nPortfolio Volatility:")
    print(f"  Daily: {volatility:.4f} ({volatility*100:.2f}%)")
    print(f"  Annualized: {volatility*np.sqrt(252):.4f} ({volatility*np.sqrt(252)*100:.2f}%)")
    
    # Risk contribution analysis
    risk_contrib = risk_manager.calculate_risk_contribution(risk_weights)
    print(f"\nRisk Contributions:")
    for asset, contribution in risk_contrib.items():
        print(f"  {asset}: {contribution:.4f} ({contribution*100:.2f}%)")
    
    # Stress testing
    stress_scenarios = {
        'Market Crash': {asset: -0.20 for asset in risk_weights.keys()},
        'Tech Selloff': {
            'AAPL': -0.15, 'GOOGL': -0.18, 'MSFT': -0.12, 
            'AMZN': -0.20, 'TSLA': -0.25
        },
        'Interest Rate Shock': {asset: -0.10 for asset in risk_weights.keys()}
    }
    
    stress_results = risk_manager.stress_test_portfolio(risk_weights, stress_scenarios)
    print(f"\nStress Test Results:")
    for scenario, result in stress_results.items():
        print(f"  {scenario}: {result['percentage_loss']:.2f}% loss")
    
    # Monte Carlo simulation
    print(f"\nRunning Monte Carlo Simulation...")
    mc_results = risk_manager.monte_carlo_simulation(
        weights=risk_weights,
        initial_value=float(portfolio.get_total_value()),
        time_horizon_days=252,
        num_simulations=1000
    )
    
    final_stats = mc_results['final_value_statistics']
    prob_analysis = mc_results['probability_analysis']
    
    print(f"Monte Carlo Results (1 year projection):")
    print(f"  Expected Value: ${final_stats['mean']:,.0f}")
    print(f"  95% Confidence Interval: ${final_stats['percentile_5']:,.0f} - ${final_stats['percentile_95']:,.0f}")
    print(f"  Probability of Positive Return: {prob_analysis['prob_positive_return']*100:.1f}%")
    print(f"  Probability of >10% Loss: {prob_analysis['prob_loss_greater_10pct']*100:.1f}%")


def demo_portfolio_optimization(returns_data, current_prices):
    """Demonstrate portfolio optimization using Modern Portfolio Theory."""
    print_section("6. Portfolio Optimization")
    
    # Prepare data for optimization
    expected_returns = returns_data.mean() * 252  # Annualize
    covariance_matrix = returns_data.cov() * 252  # Annualize
    
    print(f"Expected Annual Returns:")
    for asset, ret in expected_returns.items():
        print(f"  {asset}: {ret*100:.2f}%")
    
    # Create optimizer
    optimizer = PortfolioOptimizer(
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        risk_free_rate=0.02
    )
    
    # Maximum Sharpe Ratio optimization
    constraints = OptimizationConstraints(
        min_weights={asset: 0.0 for asset in expected_returns.index},
        max_weights={asset: 0.4 for asset in expected_returns.index},
        long_only=True
    )
    
    print(f"\nOptimizing for Maximum Sharpe Ratio...")
    max_sharpe_result = optimizer.optimize(
        objective=OptimizationObjective.MAX_SHARPE,
        constraints=constraints
    )
    
    if max_sharpe_result.success:
        print(f"Optimization successful!")
        print(f"Expected Return: {max_sharpe_result.expected_return*100:.2f}%")
        print(f"Volatility: {max_sharpe_result.volatility*100:.2f}%")
        print(f"Sharpe Ratio: {max_sharpe_result.sharpe_ratio:.3f}")
        print(f"Optimal Weights:")
        for asset, weight in max_sharpe_result.weights.items():
            if weight > 0.001:  # Only show significant weights
                print(f"  {asset}: {weight*100:.2f}%")
    
    # Minimum Volatility optimization
    print(f"\nOptimizing for Minimum Volatility...")
    min_vol_result = optimizer.optimize(
        objective=OptimizationObjective.MIN_VOLATILITY,
        constraints=constraints
    )
    
    if min_vol_result.success:
        print(f"Minimum Volatility Portfolio:")
        print(f"Expected Return: {min_vol_result.expected_return*100:.2f}%")
        print(f"Volatility: {min_vol_result.volatility*100:.2f}%")
        print(f"Weights:")
        for asset, weight in min_vol_result.weights.items():
            if weight > 0.001:
                print(f"  {asset}: {weight*100:.2f}%")
    
    # Maximum Return optimization
    print(f"\nOptimizing for Maximum Return...")
    max_return_result = optimizer.optimize(
        objective=OptimizationObjective.MAX_RETURN,
        constraints=constraints
    )
    
    if max_return_result.success:
        print(f"Maximum Return Portfolio:")
        print(f"Expected Return: {max_return_result.expected_return*100:.2f}%")
        print(f"Volatility: {max_return_result.volatility*100:.2f}%")
        print(f"Weights:")
        for asset, weight in max_return_result.weights.items():
            if weight > 0.001:
                print(f"  {asset}: {weight*100:.2f}%")
    
    # Generate efficient frontier
    print(f"\nGenerating Efficient Frontier...")
    frontier = optimizer.efficient_frontier(num_portfolios=20, constraints=constraints)
    if not frontier.empty:
        print(f"Generated {len(frontier)} efficient portfolios")
        print(f"Return range: {frontier['return'].min()*100:.2f}% to {frontier['return'].max()*100:.2f}%")
        print(f"Volatility range: {frontier['volatility'].min()*100:.2f}% to {frontier['volatility'].max()*100:.2f}%")
        print(f"Max Sharpe ratio on frontier: {frontier['sharpe_ratio'].max():.3f}")
    
    return max_sharpe_result


def demo_rebalancing_analysis(portfolio):
    """Demonstrate rebalancing analysis and recommendations."""
    print_section("7. Rebalancing Analysis")
    
    # Set target weights
    target_weights = {
        'AAPL': Decimal('0.30'),
        'GOOGL': Decimal('0.25'),
        'MSFT': Decimal('0.25'),
        'AMZN': Decimal('0.15'),
        'CASH': Decimal('0.05')
    }
    
    portfolio.set_target_weights(target_weights)
    print(f"Set target weights:")
    for asset, weight in target_weights.items():
        print(f"  {asset}: {float(weight)*100:.1f}%")
    
    # Calculate current weights
    current_weights = portfolio.get_position_weights()
    print(f"\nCurrent weights:")
    for asset, weight in current_weights.items():
        print(f"  {asset}: {float(weight)*100:.1f}%")
    
    # Calculate rebalancing needs
    rebalancing_needs = portfolio.calculate_rebalancing_needs(tolerance=Decimal('0.05'))
    
    if rebalancing_needs:
        print(f"\nRebalancing Recommendations (5% tolerance):")
        for asset, recommendation in rebalancing_needs.items():
            print(f"  {asset}:")
            print(f"    Current: {recommendation['current_weight']*100:.1f}%")
            print(f"    Target: {recommendation['target_weight']*100:.1f}%")
            print(f"    Action: {recommendation['action']} ${abs(recommendation['value_difference']):,.0f}")
    else:
        print(f"\nNo rebalancing needed - all weights within tolerance")


def demo_performance_tracking(portfolio):
    """Demonstrate performance tracking and metrics."""
    print_section("8. Performance Tracking")
    
    # Calculate performance metrics
    performance_metrics = portfolio.calculate_performance_metrics(period_days=30)
    
    if performance_metrics:
        print(f"Performance Metrics (30-day period):")
        print(f"  Total Return: {performance_metrics['total_return']*100:.2f}%")
        print(f"  Annualized Return: {performance_metrics['annualized_return']*100:.2f}%")
        print(f"  Annualized Volatility: {performance_metrics['annualized_volatility']*100:.2f}%")
        print(f"  Sharpe Ratio: {performance_metrics['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {performance_metrics['max_drawdown']*100:.2f}%")
    
    # Show transaction history
    transactions = portfolio.get_transaction_history()
    print(f"\nTransaction History ({len(transactions)} transactions):")
    for txn in transactions[-5:]:  # Show last 5 transactions
        print(f"  {txn['transaction_date']}: {txn['transaction_type'].upper()} "
              f"{txn['quantity']} {txn['symbol']} @ ${txn['price']}")


def demo_tax_reporting(portfolio):
    """Demonstrate tax reporting capabilities."""
    print_section("9. Tax Reporting")
    
    # Generate tax report for current year
    tax_report = portfolio.generate_tax_report(datetime.now().year)
    
    print(f"Tax Report for {tax_report['tax_year']}:")
    print(f"  Total Realized Gains: ${tax_report['total_realized_gains']:,.2f}")
    print(f"  Total Dividends: ${tax_report['total_dividends']:,.2f}")
    print(f"  Total Fees: ${tax_report['total_fees']:,.2f}")
    print(f"  Number of Sales: {tax_report['summary']['number_of_sales']}")
    print(f"  Number of Dividend Payments: {tax_report['summary']['number_of_dividend_payments']}")


def main():
    """Run the complete portfolio management demo."""
    print("Portfolio Management Package - Comprehensive Demo")
    print("=" * 60)
    
    # Generate sample data
    returns_data, current_prices, initial_prices = generate_sample_data()
    
    # 1. Create portfolio
    portfolio = demo_portfolio_creation()
    
    # 2. Demonstrate transactions and positions
    updated_prices = demo_transactions_and_positions(portfolio, current_prices)
    
    # 3. Advanced position features
    demo_advanced_position_features(portfolio)
    
    # 4. Dividend tracking
    demo_dividend_tracking(portfolio)
    
    # 5. Risk analysis
    demo_risk_analysis(portfolio, returns_data)
    
    # 6. Portfolio optimization
    optimal_result = demo_portfolio_optimization(returns_data, current_prices)
    
    # 7. Rebalancing analysis
    demo_rebalancing_analysis(portfolio)
    
    # 8. Performance tracking
    demo_performance_tracking(portfolio)
    
    # 9. Tax reporting
    demo_tax_reporting(portfolio)
    
    # Final portfolio summary
    print_section("Final Portfolio Summary")
    final_summary = portfolio.get_portfolio_summary()
    print(f"Portfolio: {final_summary['name']}")
    print(f"Total Value: ${final_summary['total_value']:,.2f}")
    print(f"Total Return: {final_summary['total_return_pct']:.2f}%")
    print(f"Number of Positions: {final_summary['number_of_positions']}")
    print(f"Total Transactions: {final_summary['total_transactions']}")
    print(f"Cash Balance: ${final_summary['cash_balance']:,.2f}")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"The portfolio management package provides comprehensive functionality for:")
    print(f"  ✓ Position and transaction management")
    print(f"  ✓ Risk analysis and stress testing")
    print(f"  ✓ Portfolio optimization")
    print(f"  ✓ Performance tracking")
    print(f"  ✓ Tax reporting and compliance")


if __name__ == "__main__":
    main()
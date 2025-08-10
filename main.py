#!/usr/bin/env python3
"""
Portfolio Prediction with Sentiment Analysis
Main entry point for the portfolio forecasting system.
"""

import argparse
import sys
import os
import json
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_ingestion import DataIngestion
from src.sentiment_predictor import SentimentPredictor
from src.portfolio_analyzer import PortfolioAnalyzer
from src.monte_carlo import MonteCarloSimulation
from src.scenario_analysis import ScenarioAnalysis


def run_prediction_pipeline(portfolio_path, prices_path, sentiment_path, config_path, output_dir):
    """
    Run the complete portfolio prediction pipeline.
    
    Args:
        portfolio_path: Path to portfolio CSV file
        prices_path: Path to historical prices CSV file
        sentiment_path: Path to sentiment data CSV file
        config_path: Path to configuration JSON file
        output_dir: Output directory for results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("PORTFOLIO PREDICTION PIPELINE")
    print("="*60)
    
    # Step 1: Data Ingestion
    print("\n1. Loading and validating data...")
    data_ingestion = DataIngestion(config_path)
    
    # Load all data
    portfolio_data = data_ingestion.load_portfolio(portfolio_path)
    price_data = data_ingestion.load_prices(prices_path)
    sentiment_data = data_ingestion.load_sentiment(sentiment_path)
    
    # Align data
    aligned_prices, aligned_sentiment, aligned_returns = data_ingestion.align_data()
    
    # Get PyPortfolioOpt formatted data
    pypfopt_prices, weights_dict = data_ingestion.create_pypfopt_data()
    
    # Check for holdings data
    holdings_data = data_ingestion.get_current_holdings()
    rebalancing_data = data_ingestion.calculate_rebalancing_needs()
    
    if holdings_data:
        print(f"   ✓ Holdings data found: ${holdings_data['total_portfolio_value']:,.0f} total value")
        if rebalancing_data:
            rebalancing_pct = rebalancing_data['summary']['rebalancing_percentage']
            print(f"   ⚠ Portfolio needs rebalancing: {rebalancing_pct:.1%} deviation")
    
    print("   ✓ Data loaded and aligned successfully")
    print(f"   ✓ Portfolio: {len(portfolio_data)} assets")
    print(f"   ✓ Data range: {len(aligned_prices)} days")
    
    # Print data summary
    summary = data_ingestion.summary()
    print(f"   ✓ Summary: {summary}")
    
    # Step 2: Sentiment-Based ML Prediction
    print("\n2. Training sentiment prediction models...")
    sentiment_predictor = SentimentPredictor(data_ingestion.config)
    
    # Create features and prepare training data
    features_df = sentiment_predictor.create_features(aligned_prices, aligned_sentiment, aligned_returns)
    training_data = sentiment_predictor.prepare_training_data(features_df)
    
    # Train models
    training_results = sentiment_predictor.train_models(training_data)
    
    print("   ✓ ML models trained successfully")
    for asset, metrics in training_results.items():
        print(f"     {asset}: R² = {metrics['r2']:.3f}, MSE = {metrics['mse']:.6f}")
    
    # Step 3: Portfolio Risk Analysis
    print("\n3. Calculating portfolio risk metrics...")
    portfolio_analyzer = PortfolioAnalyzer(data_ingestion.config)
    
    # Calculate base metrics
    base_returns, cov_matrix = portfolio_analyzer.calculate_base_metrics(pypfopt_prices, weights_dict)
    
    # Generate current sentiment predictions (using latest data)
    current_sentiment_features = {}
    latest_sentiment = aligned_sentiment.iloc[-1]
    for col in latest_sentiment.index:
        current_sentiment_features[col] = latest_sentiment[col]
    
    # Get sentiment predictions (only if models were trained successfully)
    if sentiment_predictor.models:
        sentiment_predictions = sentiment_predictor.predict_returns(current_sentiment_features)
    else:
        print("   ⚠ Warning: No models trained, using base returns only")
        sentiment_predictions = {}
    
    # Adjust returns for sentiment
    adjusted_returns = portfolio_analyzer.adjust_returns_for_sentiment(sentiment_predictions)
    
    # Calculate portfolio metrics
    base_metrics = portfolio_analyzer.calculate_portfolio_metrics(use_sentiment_adjusted=False)
    adjusted_metrics = portfolio_analyzer.calculate_portfolio_metrics(use_sentiment_adjusted=True)
    
    print("   ✓ Portfolio metrics calculated")
    print(f"     Base Expected Return: {base_metrics['expected_return']:.4f}")
    print(f"     Sentiment-Adjusted Return: {adjusted_metrics['expected_return']:.4f}")
    print(f"     Portfolio Volatility: {base_metrics['volatility']:.4f}")
    print(f"     Sharpe Ratio (Base/Adjusted): {base_metrics['sharpe_ratio']:.3f} / {adjusted_metrics['sharpe_ratio']:.3f}")
    
    # Calculate VaR/CVaR
    var_cvar = portfolio_analyzer.calculate_var_cvar(aligned_returns, weights_dict)
    print(f"     VaR (95%): {var_cvar['VaR_0.95']:.4f}")
    print(f"     CVaR (95%): {var_cvar['CVaR_0.95']:.4f}")
    
    # Portfolio holdings analysis if available
    holdings_analysis = None
    rebalancing_trades = None
    if holdings_data:
        holdings_analysis = portfolio_analyzer.calculate_current_portfolio_value(holdings_data)
        print(f"\n   Portfolio Holdings Analysis:")
        print(f"     Total Portfolio Value: ${holdings_analysis['total_value']:,.0f}")
        
        # Show largest weight deviations
        max_deviation = 0
        for ticker, analysis in holdings_analysis['weight_analysis'].items():
            deviation = abs(analysis['difference'])
            if deviation > max_deviation:
                max_deviation = deviation
        print(f"     Maximum Weight Deviation: {max_deviation:.1%}")
        
        # Rebalancing analysis
        if rebalancing_data:
            rebalancing_trades = portfolio_analyzer.calculate_rebalancing_trades(rebalancing_data)
            if rebalancing_trades['summary']['recommended']:
                print(f"     Rebalancing Recommended: Net benefit ${rebalancing_trades['summary']['net_benefit']:.0f}")
            else:
                print(f"     Rebalancing Not Recommended: Cost exceeds benefit")
    
    # Step 4: Monte Carlo Simulation
    print("\n4. Running Monte Carlo simulations...")
    monte_carlo = MonteCarloSimulation(data_ingestion.config)
    
    # Run base Monte Carlo
    current_weights = pd.Series(weights_dict)
    mc_results = monte_carlo.monte_carlo_portfolio_simulation(
        adjusted_returns, cov_matrix, 
        current_weights, 100000
    )
    
    print("   ✓ Monte Carlo simulation completed")
    print(f"     Mean Final Value: ${mc_results['final_values']['mean']:.0f}")
    print(f"     95% Confidence Interval: ${mc_results['final_values']['percentiles'][5]:.0f} - ${mc_results['final_values']['percentiles'][95]:.0f}")
    print(f"     Probability of Loss: {mc_results['risk_metrics']['probability_of_loss']:.3f}")
    
    # Bootstrap simulation
    print("\n   Running bootstrap simulation...")
    bootstrap_results = monte_carlo.bootstrap_simulation(aligned_returns, current_weights)
    print(f"   ✓ Bootstrap Mean Final Value: ${bootstrap_results['final_values']['mean']:.0f}")
    
    # Step 5: Scenario Analysis
    print("\n5. Performing scenario analysis...")
    scenario_analyzer = ScenarioAnalysis(data_ingestion.config)
    
    # Run scenario analysis
    scenario_results = scenario_analyzer.portfolio_scenario_analysis(
        base_returns, cov_matrix, current_weights
    )
    
    print("   ✓ Scenario analysis completed")
    for scenario, results in scenario_results.items():
        metrics = results['portfolio_metrics']
        print(f"     {scenario.title()}: Return = {metrics['expected_return']:.4f}, Sharpe = {metrics['sharpe_ratio']:.3f}")
    
    # Monte Carlo scenario comparison
    print("\n   Running Monte Carlo for all scenarios...")
    mc_scenario_results = scenario_analyzer.monte_carlo_scenario_comparison(
        base_returns, cov_matrix, current_weights, monte_carlo
    )
    
    # Step 6: Generate Reports and Save Results
    print("\n6. Generating reports and saving results...")
    
    # Save main results
    results = {
        'data_summary': summary,
        'training_results': training_results,
        'sentiment_predictions': sentiment_predictions,
        'portfolio_metrics': {
            'base': base_metrics,
            'sentiment_adjusted': adjusted_metrics,
            'var_cvar': var_cvar
        },
        'monte_carlo': {
            'base_simulation': {
                'mean_return': mc_results['returns']['mean'],
                'volatility': mc_results['returns']['std'],
                'percentiles': mc_results['returns']['percentiles'],
                'risk_metrics': mc_results['risk_metrics']
            },
            'bootstrap': {
                'mean_return': bootstrap_results['returns']['mean'],
                'volatility': bootstrap_results['returns']['std'],
                'percentiles': bootstrap_results['returns']['percentiles']
            }
        },
        'scenario_analysis': scenario_results,
        'scenario_monte_carlo': {
            scenario: {
                'mean_return': results['returns']['mean'],
                'percentiles': results['returns']['percentiles']
            } for scenario, results in mc_scenario_results.items()
        }
    }
    
    # Add holdings analysis if available
    if holdings_data:
        results['holdings_analysis'] = holdings_analysis
        if rebalancing_trades:
            results['rebalancing_analysis'] = rebalancing_trades
    
    # Save to JSON
    results_file = os.path.join(output_dir, 'portfolio_prediction_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"   ✓ Results saved to {results_file}")
    
    # Create scenario summary
    scenario_summary = scenario_analyzer.scenario_summary_report(scenario_results)
    scenario_file = os.path.join(output_dir, 'scenario_summary.csv')
    scenario_summary.to_csv(scenario_file, index=False)
    print(f"   ✓ Scenario summary saved to {scenario_file}")
    
    # Create holdings and rebalancing reports if available
    if holdings_data:
        # Holdings breakdown report
        holdings_df = pd.DataFrame({
            'ticker': list(holdings_analysis['holdings_breakdown'].keys()),
            'shares': [h['shares'] for h in holdings_analysis['holdings_breakdown'].values()],
            'current_price': [h['price'] for h in holdings_analysis['holdings_breakdown'].values()],
            'market_value': [h['market_value'] for h in holdings_analysis['holdings_breakdown'].values()],
            'current_weight': [h['percentage'] for h in holdings_analysis['holdings_breakdown'].values()],
            'target_weight': [holdings_analysis['weight_analysis'][ticker]['target_weight'] 
                             for ticker in holdings_analysis['holdings_breakdown'].keys()],
            'weight_difference': [holdings_analysis['weight_analysis'][ticker]['difference'] 
                                 for ticker in holdings_analysis['holdings_breakdown'].keys()]
        })
        
        holdings_file = os.path.join(output_dir, 'current_holdings.csv')
        holdings_df.to_csv(holdings_file, index=False)
        print(f"   ✓ Current holdings report saved to {holdings_file}")
        
        # Rebalancing trades report if available
        if rebalancing_trades and rebalancing_trades['summary']['total_trades'] > 0:
            trades_data = []
            for ticker, trade in rebalancing_trades.items():
                if ticker == 'summary' or not isinstance(trade, dict) or 'action' not in trade:
                    continue
                trades_data.append({
                    'ticker': ticker,
                    'action': trade['action'],
                    'shares': trade['shares'],
                    'trade_value': trade['trade_value'],
                    'transaction_cost': trade['transaction_cost'],
                    'current_weight': trade['current_weight'],
                    'target_weight': trade['target_weight']
                })
            
            if trades_data:
                trades_df = pd.DataFrame(trades_data)
                trades_file = os.path.join(output_dir, 'rebalancing_trades.csv')
                trades_df.to_csv(trades_file, index=False)
                print(f"   ✓ Rebalancing trades report saved to {trades_file}")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nKey Results:")
    print(f"• Expected Return (Sentiment-Adjusted): {adjusted_metrics['expected_return']:.2%}")
    print(f"• Portfolio Volatility: {adjusted_metrics['volatility']:.2%}")
    print(f"• Sharpe Ratio: {adjusted_metrics['sharpe_ratio']:.3f}")
    print(f"• VaR (95%): {var_cvar['VaR_0.95']:.2%}")
    print(f"• Probability of Loss: {mc_results['risk_metrics']['probability_of_loss']:.1%}")
    
    # Portfolio holdings summary
    if holdings_data:
        print(f"\nCurrent Portfolio:")
        print(f"• Total Value: ${holdings_analysis['total_value']:,.0f}")
        print(f"• Number of Holdings: {len(holdings_analysis['holdings_breakdown'])}")
        if rebalancing_trades and rebalancing_trades['summary']['recommended']:
            print(f"• Rebalancing Recommended: {rebalancing_trades['summary']['total_trades']} trades")
            print(f"• Est. Net Benefit: ${rebalancing_trades['summary']['net_benefit']:.0f}")
    
    print(f"\nScenario Comparison:")
    for scenario in ['bearish', 'neutral', 'bullish']:
        if scenario in scenario_results:
            ret = scenario_results[scenario]['portfolio_metrics']['expected_return']
            print(f"• {scenario.title()}: {ret:.2%}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run portfolio prediction with sentiment analysis")
    parser.add_argument("--portfolio", default="data/sample_portfolio.csv", help="Portfolio CSV file path")
    parser.add_argument("--prices", default="data/sample_prices.csv", help="Historical prices CSV file path")
    parser.add_argument("--sentiment", default="data/sample_sentiment.csv", help="Sentiment data CSV file path")
    parser.add_argument("--config", default="config/config.json", help="Configuration file path")
    parser.add_argument("--output", default="output", help="Output directory")
    
    args = parser.parse_args()
    
    print("=== Portfolio Prediction with Sentiment Analysis ===")
    print(f"Portfolio: {args.portfolio}")
    print(f"Prices: {args.prices}")
    print(f"Sentiment: {args.sentiment}")
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")
    
    try:
        results = run_prediction_pipeline(
            args.portfolio, args.prices, args.sentiment, args.config, args.output
        )
        print("\n✅ Pipeline executed successfully!")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
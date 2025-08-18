#!/usr/bin/env python3
"""
Demo script showing how to use BenchMarkData class for calculating index returns
This is useful for beta calculations in the portfolio prediction system.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from benchmark_data import BenchMarkData, get_project_root

def demo_index_returns():
    """Demonstrate how to calculate index returns from CSV files"""
    
    project_root = get_project_root()
    data_dir = os.path.join(project_root, 'data', 'benchmarks')
    
    # Example index files (from the benchmark downloader)
    indices = {
        'Nifty 50': 'nifty_50_01012024_to_15082025.csv',
        'Nifty IT': 'nifty_it_01012024_to_15082025.csv', 
        'Nifty Bank': 'nifty_bank_01012024_to_15082025.csv'
    }
    
    print("INDEX RETURNS CALCULATION DEMO")
    print("="*50)
    
    for index_name, filename in indices.items():
        csv_path = os.path.join(data_dir, filename)
        
        if os.path.exists(csv_path):
            print(f"\n{index_name}:")
            print("-" * len(index_name) + "-")
            
            # Method 1: Quick calculation using static method
            returns = BenchMarkData.calculate_returns_from_csv(csv_path)
            
            print(f"Returns calculated: {len(returns)} days")
            print(f"Mean daily return: {returns.mean():.6f} ({returns.mean()*100:.4f}%)")
            print(f"Daily volatility: {returns.std():.6f} ({returns.std()*100:.4f}%)")
            print(f"Annualized return: {returns.mean()*252:.6f} ({returns.mean()*252*100:.2f}%)")
            print(f"Annualized volatility: {returns.std()*252**0.5:.6f} ({returns.std()*252**0.5*100:.2f}%)")
            
            # Show first few returns
            print(f"First 5 daily returns:")
            for date, ret in returns.head().items():
                print(f"  {date.strftime('%Y-%m-%d')}: {ret:.6f} ({ret*100:.4f}%)")
                
        else:
            print(f"\nFile not found: {csv_path}")
            print(f"Run 'python src/benchmark_data.py' first to download data.")

def demo_beta_calculation_setup():
    """Show how this data can be used for beta calculations"""
    
    print(f"\n\n" + "="*50)
    print("BETA CALCULATION SETUP")
    print("="*50)
    
    project_root = get_project_root()
    data_dir = os.path.join(project_root, 'data', 'benchmarks')
    
    # Get Nifty 50 returns (market benchmark)
    nifty_file = os.path.join(data_dir, 'nifty_50_01012024_to_15082025.csv')
    
    if os.path.exists(nifty_file):
        # Calculate market returns
        market_returns = BenchMarkData.calculate_returns_from_csv(nifty_file)
        
        print(f"Market (Nifty 50) returns ready for beta calculation:")
        print(f"- Data points: {len(market_returns)}")
        print(f"- Date range: {market_returns.index[0].strftime('%Y-%m-%d')} to {market_returns.index[-1].strftime('%Y-%m-%d')}")
        print(f"- Market return stats: mean={market_returns.mean():.6f}, std={market_returns.std():.6f}")
        
        print(f"\nTo calculate beta for your portfolio:")
        print(f"1. Calculate your portfolio daily returns using similar method")
        print(f"2. Use the calculate_beta() function from market_scenario.py")
        print(f"3. Pass portfolio_returns and market_returns (this Nifty 50 data)")
        
        # Example usage pattern
        print(f"\nExample code pattern:")
        print(f"```python")
        print(f"from src.benchmark_data import BenchMarkData")
        print(f"from src.market_scenario import calculate_beta")
        print(f"")
        print(f"# Get market returns")
        print(f"market_returns = BenchMarkData.calculate_returns_from_csv('{nifty_file}')")
        print(f"")
        print(f"# Get your portfolio returns (you need to implement this)")
        print(f"# portfolio_returns = your_portfolio_returns_calculation()")
        print(f"")
        print(f"# Calculate beta")
        print(f"# beta = calculate_beta(portfolio_returns, market_returns)")
        print(f"```")
        
    else:
        print(f"Nifty 50 data not found. Run download first.")

if __name__ == "__main__":
    demo_index_returns()
    demo_beta_calculation_setup()
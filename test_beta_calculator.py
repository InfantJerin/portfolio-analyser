#!/usr/bin/env python3
"""
Demonstration of the Beta Calculator for portfolio sensitivity analysis
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from portfolio import Portfolio
from beta_calculator import BetaCalculator
from benchmark_data import get_project_root

def demo_portfolio_sensitivity():
    """Demonstrate portfolio sensitivity analysis to different indices"""
    
    print("PORTFOLIO SENSITIVITY ANALYSIS DEMO")
    print("="*60)
    
    # Get test portfolio
    project_root = get_project_root()
    test_portfolio_path = os.path.join(project_root, 'data', 'test_portfolio.csv')
    
    if not os.path.exists(test_portfolio_path):
        print(f"Test portfolio not found. Creating sample data...")
        # Could generate sample data here if needed
        return
    
    # Create portfolio object
    portfolio = Portfolio(test_portfolio_path)
    
    # Create beta calculator
    beta_calc = BetaCalculator(portfolio)
    
    print(f"\nPortfolio loaded: {test_portfolio_path}")
    
    # Analyze sensitivity to key indices
    key_indices = ['Nifty 50', 'Nifty IT', 'Nifty Bank']
    
    print(f"\nAnalyzing sensitivity to: {', '.join(key_indices)}")
    print("-" * 60)
    
    # Get analysis results
    results = beta_calc.calculate_multiple_betas(key_indices)
    
    print(f"\nQuick Results:")
    for index_name, (beta, metadata) in results.items():
        sensitivity = beta_calc._interpret_beta(beta)
        print(f"  {index_name}: Beta = {beta:.4f} ({sensitivity})")
    
    # Show detailed report
    print(f"\n" + "="*60)
    print("DETAILED SENSITIVITY REPORT")
    print("="*60)
    
    beta_calc.print_sensitivity_report(key_indices)

def show_usage_examples():
    """Show practical usage examples"""
    
    print(f"\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    
    print(f"\n1. Basic Usage:")
    print("-" * 20)
    print("""
# Load your portfolio
portfolio = Portfolio('path/to/your_portfolio.csv')

# Create beta calculator  
beta_calc = BetaCalculator(portfolio)

# Calculate beta for Nifty 50
result = beta_calc.calculate_single_beta('Nifty 50')
if result:
    beta, metadata = result
    print(f'Portfolio Beta vs Nifty 50: {beta:.4f}')
""")
    
    print(f"\n2. Multiple Index Analysis:")
    print("-" * 30)
    print("""
# Analyze against multiple indices
indices = ['Nifty 50', 'Nifty IT', 'Nifty Bank', 'Nifty Auto']
results = beta_calc.calculate_multiple_betas(indices)

# Print all results
for index_name, (beta, metadata) in results.items():
    print(f'{index_name}: {beta:.4f}')
""")
    
    print(f"\n3. Comprehensive Report:")
    print("-" * 25)
    print("""
# Get full sensitivity analysis report
beta_calc.print_sensitivity_report()

# Or get as DataFrame for further analysis
df = beta_calc.get_sensitivity_analysis()
print(df[['Index', 'Beta', 'Sensitivity', 'Correlation']])
""")
    
    print(f"\n4. Interpretation:")
    print("-" * 17)
    print("""
Beta Values:
- Beta > 1.0: Portfolio is more volatile than the index (aggressive)
- Beta < 1.0: Portfolio is less volatile than the index (defensive)  
- Beta ≈ 1.0: Portfolio moves in line with the index
- Negative Beta: Portfolio moves opposite to the index

Sensitivity Categories:
- Highly Aggressive (β > 1.2)
- Aggressive (β > 1.0) 
- Moderately Aggressive (β > 0.8)
- Defensive (β > 0.5)
- Highly Defensive (β > 0)
- Negative Sensitivity (β < 0)
""")

if __name__ == "__main__":
    demo_portfolio_sensitivity()
    show_usage_examples()
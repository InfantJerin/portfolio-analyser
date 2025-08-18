import pandas as pd
import os
import numpy as np
from datetime import datetime

from benchmark_data import get_project_root

class Portfolio:

    def __init__(self, portfolio_csv_path):
        self.portfolio_csv_path = portfolio_csv_path
       

    def get_portfolio_data(self):
        return pd.read_csv(self.portfolio_csv_path)

    
    def calculate_daily_returns(self):
        """
        Calculate daily returns for a portfolio from combined CSV file.
        
        Args:
            portfolio_csv_path: Path to CSV with columns: date, {ticker}_price, {ticker}_weight
        
        Returns:
            pandas.Series with daily returns for the portfolio
        """
        
        # Read combined portfolio data
        df = pd.read_csv(self.portfolio_csv_path)
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%d-%m-%Y')
        df.set_index('TIMESTAMP', inplace=True)
        
        # Extract tickers from column names
        price_cols = [col for col in df.columns if col.endswith('_price')]
        tickers = [col.replace('_price', '') for col in price_cols]
        
        # Calculate daily returns for each stock
        portfolio_returns = pd.Series(index=df.index[1:], dtype=float)
        
        for i in range(1, len(df)):
            daily_return = 0
            for ticker in tickers:
                price_col = f"{ticker}_price"
                weight_col = f"{ticker}_weight"
                
                # Calculate single day return for this stock
                prev_price = df.iloc[i-1][price_col]
                curr_price = df.iloc[i][price_col]
                stock_return = (curr_price - prev_price) / prev_price
                
                # Weight by portfolio allocation
                weight = df.iloc[i][weight_col]
                daily_return += weight * stock_return
            
            portfolio_returns.iloc[i-1] = daily_return
        
        return portfolio_returns


    def calculate_CAGR(self):
        """
        Calculate the Compound Annual Growth Rate (CAGR) for a portfolio.
        
        Returns:
            float: CAGR value
        """
        # Get portfolio returns
        returns = self.calculate_daily_returns()
        
        if len(returns) == 0:
            return 0.0
        
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        print(cumulative_returns)
        
        # Get beginning and ending values
        beginning_value = 1.0  # Starting with 1 unit of investment
        ending_value = cumulative_returns.iloc[-1]
        
        # Calculate number of years
        first_date = returns.index[0]
        last_date = returns.index[-1]
        days = (last_date - first_date).days
        years = days / 365.25
        
        if years <= 0 or ending_value <= 0:
            return 0.0
        
        # Calculate CAGR: ((Ending Value / Beginning Value) ^ (1 / Number of Years)) - 1
        cagr = (ending_value / beginning_value) ** (1 / years) - 1
        
        return cagr


    def calculate_XIRR(self):
        """
        Calculate XIRR (Extended Internal Rate of Return) using Newton-Raphson method.
        
        Returns:
            float: XIRR value as a decimal (e.g., 0.12 for 12%)
        """
        # Read portfolio data
        df = pd.read_csv(self.portfolio_csv_path)
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%d-%m-%Y')
        df = df.sort_values('TIMESTAMP')
        
        # Extract tickers from column names
        price_cols = [col for col in df.columns if col.endswith('_price')]
        tickers = [col.replace('_price', '') for col in price_cols]
        
        # Calculate cash flows (negative for investments)
        cash_flows = []
        dates = []
        
        # Initial investment (negative cash flow)
        initial_investment = 0
        for ticker in tickers:
            price_col = f"{ticker}_price"
            weight_col = f"{ticker}_weight"
            initial_investment += df.iloc[0][price_col] * df.iloc[0][weight_col]
        
        cash_flows.append(-initial_investment)
        dates.append(df.iloc[0]['TIMESTAMP'])
        
        # Final value (positive cash flow)
        final_value = 0
        for ticker in tickers:
            price_col = f"{ticker}_price"
            weight_col = f"{ticker}_weight"
            final_value += df.iloc[-1][price_col] * df.iloc[-1][weight_col]
        
        cash_flows.append(final_value)
        dates.append(df.iloc[-1]['TIMESTAMP'])
        
        # Convert dates to number of years from start date
        start_date = dates[0]
        years = [(date - start_date).days / 365.25 for date in dates]
        
        # Newton-Raphson method to solve for IRR
        def npv_func(rate, cash_flows, years):
            """Calculate Net Present Value for given rate"""
            return sum(cf / (1 + rate) ** year for cf, year in zip(cash_flows, years))
        
        def npv_derivative(rate, cash_flows, years):
            """Calculate derivative of NPV function"""
            return sum(-year * cf / (1 + rate) ** (year + 1) for cf, year in zip(cash_flows, years))
        
        # Initial guess for rate
        rate = 0.1  # 10%
        tolerance = 1e-6
        max_iterations = 100
        
        for i in range(max_iterations):
            npv = npv_func(rate, cash_flows, years)
            npv_prime = npv_derivative(rate, cash_flows, years)
            
            if abs(npv) < tolerance:
                break
                
            if npv_prime == 0:
                # Avoid division by zero
                break
                
            rate = rate - npv / npv_prime
            
            # Avoid negative rates that would cause issues
            if rate < -0.99:
                rate = -0.99
        
        return rate

    def calculate_sharpe_ratio(self, risk_free_rate=0.04):   
        """
        Calculate the Sharpe ratio for the portfolio.
        
        Args:
            risk_free_rate (float): Risk-free rate (default 2% annually)
        
        Returns:
            float: Sharpe ratio
        """
        # Get portfolio returns
        returns = self.calculate_daily_returns()
        
        if len(returns) == 0:
            return 0.0
        
        # Convert daily returns to annual metrics
        mean_daily_return = returns.mean()
        std_daily_return = returns.std()
        
        # Annualize returns and volatility
        annual_return = (1 + mean_daily_return) ** 252 - 1  # 252 trading days
        annual_volatility = std_daily_return * np.sqrt(252)
        
        # Avoid division by zero
        if annual_volatility == 0:
            return 0.0
        
        # Calculate Sharpe ratio
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        
        return sharpe_ratio

    def calculate_volatility(self):
        """
        Calculate the annualized volatility of the portfolio.
        
        Returns:
            float: Annualized volatility as a decimal
        """
        # Get portfolio returns
        returns = self.calculate_daily_returns()
        
        if len(returns) == 0:
            return 0.0
        
        # Calculate daily volatility (standard deviation)
        daily_volatility = returns.std()
        
        # Annualize volatility
        annual_volatility = daily_volatility * np.sqrt(252)  # 252 trading days
        
        return annual_volatility

if __name__ == "__main__":
    root = get_project_root()
    portfolio = Portfolio(os.path.join(root, "data", "test_portfolio.csv"))
    print(f"CAGR: {portfolio.calculate_CAGR():.4f}")
    print(f"XIRR: {portfolio.calculate_XIRR():.4f}")
    print(f"Sharpe Ratio: {portfolio.calculate_sharpe_ratio():.4f}")
    print(f"Volatility: {portfolio.calculate_volatility():.4f}")
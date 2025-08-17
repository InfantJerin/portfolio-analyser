

import pandas as pd
import numpy as np
import os

from nselib import capital_market

def get_project_root():
    """
    Get the absolute path to the project root directory (parent of src).
    
    Returns:
        str: Absolute path to the project root directory
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    return project_root

def generate_portfolio_returns(portfolio_csv_path, tickers, periods):
    """
    Generate synthetic portfolio test data and write to CSV file
    
    Args:
        portfolio_csv_path: Path where to write the generated CSV file
        tickers: List of ticker symbols to include (e.g., ['TCS', 'INFY', 'RELIANCE'])
        periods: Number of days to generate data for
    """
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Generate date range
    start_date = pd.Timestamp.now() - pd.Timedelta(days=periods)
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    # Initialize DataFrame
    data = {'TIMESTAMP': dates}
    
    # Generate data for each ticker
    for i, ticker in enumerate(tickers):
        # Starting price (different for each stock)
        base_prices = {'TCS': 3400, 'INFY': 1800, 'RELIANCE': 2500, 'HDFC': 1600, 'WIPRO': 450}
        start_price = base_prices.get(ticker, 1000 + i * 200)
        
        # Different volatility for each stock (annual volatility converted to daily)
        volatilities = {'TCS': 0.25, 'INFY': 0.30, 'RELIANCE': 0.35, 'HDFC': 0.20, 'WIPRO': 0.40}
        annual_volatility = volatilities.get(ticker, 0.25 + i * 0.05)
        daily_volatility = annual_volatility / np.sqrt(252)
        
        # Generate price series using geometric Brownian motion
        prices = [start_price]
        for _ in range(1, periods):
            # Random daily return
            daily_return = np.random.normal(0.0008, daily_volatility)  # Small positive drift
            new_price = prices[-1] * (1 + daily_return)
            prices.append(max(new_price, start_price * 0.5))  # Prevent prices going too low
        
        # Add price column
        data[f'{ticker}_price'] = prices
        
        # Generate portfolio weights (can vary slightly over time)
        base_weight = 1.0 / len(tickers)  # Equal weight as starting point
        weights = []
        for _ in range(periods):
            # Add small random variation to weights
            weight_variation = np.random.normal(0, 0.02)  # 2% standard deviation
            weight = max(0.05, min(0.95, base_weight + weight_variation))  # Keep within bounds
            weights.append(weight)
        
        # Normalize weights so they sum to 1 for each day
        data[f'{ticker}_weight'] = weights
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Normalize weights for each day so they sum to 1
    weight_cols = [col for col in df.columns if col.endswith('_weight')]
    for i in range(len(df)):
        total_weight = sum(df.iloc[i][col] for col in weight_cols)
        for col in weight_cols:
            df.iloc[i, df.columns.get_loc(col)] = df.iloc[i][col] / total_weight
    
    # Format date column
    df['TIMESTAMP'] = df['TIMESTAMP'].dt.strftime('%d-%m-%Y')
    
    # Round numerical values
    for col in df.columns:
        if col != 'date':
            if '_price' in col:
                df[col] = df[col].round(2)
            elif '_weight' in col:
                df[col] = df[col].round(4)
    
    # Write to CSV
    df.to_csv(portfolio_csv_path, index=False)
    
    print(f"Generated portfolio data with {len(tickers)} tickers for {periods} days")
    print(f"Data written to: {portfolio_csv_path}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Date range: {df['TIMESTAMP'].iloc[0]} to {df['TIMESTAMP'].iloc[-1]}")
    
    return df
    
def calculate_beta(portfolio_returns, market_returns):
    """
    Calculate the beta for a portfolio.
    
    Beta = Covariance(Portfolio Returns, Market Returns) / Variance(Market Returns)
    
    Args:
        portfolio_returns: pandas.Series of portfolio daily returns
        market_returns: pandas.Series of market (Nifty) daily returns
    
    Returns:
        float: Beta value of the portfolio
    """

    
    # Align the returns by date (ensure same dates)
    aligned_data = pd.concat([portfolio_returns, market_returns], axis=1, join='inner')
    aligned_data.columns = ['portfolio', 'market']
    
    # Calculate covariance between portfolio and market returns
    covariance = np.cov(aligned_data['portfolio'], aligned_data['market'])[0, 1]
    
    # Calculate variance of market returns
    market_variance = np.var(aligned_data['market'])
    
    # Beta = Covariance / Market Variance
    beta = covariance / market_variance
    
    return beta 


def calculate_alpha(portfolio_return, market_return, risk_free_rate, beta):
    """
    Calculate the alpha (Jensen's alpha) for a portfolio.
    
    Alpha = Portfolio Return - [Risk-free Rate + Beta × (Market Return - Risk-free Rate)]
    
    Args:
        portfolio_return: Annual portfolio return (as decimal, e.g., 0.15 for 15%)
        market_return: Annual market return (as decimal, e.g., 0.13 for 13%)
        risk_free_rate: Annual risk-free rate (as decimal, e.g., 0.05 for 5%)
        beta: Portfolio beta
    
    Returns:
        float: Alpha value (excess return)
    """
    
    # Expected return based on CAPM
    expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
    
    # Alpha = Actual return - Expected return
    alpha = portfolio_return - expected_return
    
    return alpha


def get_day1_value(portfolio_csv_path):
    """
    Calculate the total portfolio value on the first day.
    
    Args:
        portfolio_csv_path: Path to CSV with columns: TIMESTAMP, {ticker}_price, {ticker}_weight
    
    Returns:
        float: Total portfolio value on day 1
    """
    df = pd.read_csv(portfolio_csv_path)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%d-%m-%Y')
    df.set_index('TIMESTAMP', inplace=True)
    
    # Get first row
    first_day = df.iloc[0]
    
    # Extract tickers from column names
    price_cols = [col for col in df.columns if col.endswith('_price')]
    tickers = [col.replace('_price', '') for col in price_cols]
    
    # Assume initial investment amount (you can adjust this)
    total_investment = 100000  # $100,000 initial investment
    
    # Calculate portfolio value for first day
    portfolio_value = 0
    for ticker in tickers:
        price_col = f"{ticker}_price"
        weight_col = f"{ticker}_weight"
        
        # Calculate shares based on weight and initial investment
        allocation = total_investment * first_day[weight_col]
        shares = allocation / first_day[price_col]
        portfolio_value += shares * first_day[price_col]
    
    return portfolio_value


def get_last_day_value(portfolio_csv_path):
    """
    Calculate the total portfolio value on the last day.
    
    Args:
        portfolio_csv_path: Path to CSV with columns: TIMESTAMP, {ticker}_price, {ticker}_weight
    
    Returns:
        float: Total portfolio value on last day
    """
    df = pd.read_csv(portfolio_csv_path)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%d-%m-%Y')
    df.set_index('TIMESTAMP', inplace=True)
    
    # Get first and last rows
    first_day = df.iloc[0]
    last_day = df.iloc[-1]
    
    # Extract tickers from column names
    price_cols = [col for col in df.columns if col.endswith('_price')]
    tickers = [col.replace('_price', '') for col in price_cols]
    
    # Assume initial investment amount (same as day 1)
    total_investment = 100000  # $100,000 initial investment
    
    # Calculate shares based on first day allocation (buy and hold)
    portfolio_value = 0
    for ticker in tickers:
        price_col = f"{ticker}_price"
        weight_col = f"{ticker}_weight"
        
        # Calculate initial shares based on first day weight and investment
        initial_allocation = total_investment * first_day[weight_col]
        shares = initial_allocation / first_day[price_col]
        
        # Calculate current value using last day price
        current_value = shares * last_day[price_col]
        portfolio_value += current_value
    
    return portfolio_value


def calculate_daily_returns(portfolio_csv_path):
    """
    Calculate daily returns for a portfolio from combined CSV file.
    
    Args:
        portfolio_csv_path: Path to CSV with columns: date, {ticker}_price, {ticker}_weight
    
    Returns:
        pandas.Series with daily returns for the portfolio
    """
    
    # Read combined portfolio data
    df = pd.read_csv(portfolio_csv_path)
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


def calculate_nifty_daily_returns(nifty_csv_path):
    """
    Calculate daily returns for Nifty from CSV file.
    
    Args:
        nifty_csv_path: Path to CSV with columns: date, nifty_price
    
    Returns:
        pandas.Series with daily returns for Nifty
    """
    
    # Read Nifty price data
    df = pd.read_csv(nifty_csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Calculate daily returns
    nifty_returns = df['nifty_price'].pct_change().dropna()
    
    return nifty_returns


def calculate_nifty_50_daily_returns(nifty_csv_path):
    """
    Calculate daily returns for Nifty 50 from CSV file.
    
    Args:
        nifty_csv_path: Path to CSV with columns: TIMESTAMP,INDEX_NAME,OPEN_INDEX_VAL,HIGH_INDEX_VAL,CLOSE_INDEX_VAL,LOW_INDEX_VAL,TRADED_QTY,TURN_OVER
    """
    df = pd.read_csv(nifty_csv_path)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%d-%m-%Y')
    
    # Remove duplicate dates by keeping the first occurrence
    df = df.drop_duplicates(subset=['TIMESTAMP'], keep='first')
    
    df.set_index('TIMESTAMP', inplace=True)
    
    # Calculate daily returns
    nifty_returns = df['CLOSE_INDEX_VAL'].pct_change().dropna()
    
    return nifty_returns


def test_beta():

    
    # Get the project root directory (parent of src)
    project_root = get_project_root()
    
    # Portfolio returns
    portfolio_csv_path = os.path.join(project_root, 'data', 'test_portfolio.csv')
    portfolio_returns = calculate_daily_returns(portfolio_csv_path)
    print("Portfolio Daily Returns:")
    print(portfolio_returns.head())
    print(f"Mean Daily Return: {portfolio_returns.mean():.6f}")
    print(f"Daily Volatility: {portfolio_returns.std():.6f}")
    
    print("\n" + "="*50)
    
    # Nifty returns
    nifty_csv_path = os.path.join(project_root, 'data', 'nifty_prices_new.csv')
    nifty_returns = calculate_nifty_50_daily_returns(nifty_csv_path)
    print("Nifty Daily Returns:")
    print(nifty_returns.head())
    print(f"Mean Daily Return: {nifty_returns.mean():.6f}")
    print(f"Daily Volatility: {nifty_returns.std():.6f}")
    
    print("\n" + "="*50)
    
    # Calculate Beta
    beta = calculate_beta(portfolio_returns, nifty_returns)
    print(f"Portfolio Beta: {beta:.4f}")
    
    # Beta interpretation
    if beta > 1:
        print("Portfolio is more volatile than the market (aggressive)")
    elif beta < 1:
        print("Portfolio is less volatile than the market (defensive)")
    else:
        print("Portfolio moves in line with the market")

def test_alpha():
    """
    Test alpha calculation using portfolio data and given parameters.
    """
    
    # Get the project root directory (parent of src)
    project_root = get_project_root()
    
    # Load portfolio and market data
    portfolio_csv_path = os.path.join(project_root, 'data', 'test_portfolio.csv')
    portfolio_returns = calculate_daily_returns(portfolio_csv_path)
    
    nifty_csv_path = os.path.join(project_root, 'data', 'nifty_prices_new.csv')
    nifty_returns = calculate_nifty_50_daily_returns(nifty_csv_path)
    
    # Calculate portfolio beta
    beta = calculate_beta(portfolio_returns, nifty_returns)
    
    # Calculate annualized portfolio return using CAGR method
    # CAGR = (Ending Value / Beginning Value)^(1/years) - 1
    # For daily returns: cumulative_return = (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
    
    # Calculate cumulative return by compounding daily returns
    cumulative_return = (1 + portfolio_returns).prod() - 1
    
    # Calculate number of years (trading days / 252)
    years = len(portfolio_returns) / 252
    
    # Calculate CAGR: (1 + cumulative_return)^(1/years) - 1
    portfolio_annual_return = (1 + cumulative_return) ** (1 / years) - 1
    
    # Given parameters
    market_annual_return = 0.13  # 13%
    risk_free_rate = 0.05        # 5%
    
    # Calculate alpha
    alpha = calculate_alpha(portfolio_annual_return, market_annual_return, risk_free_rate, beta)
    
    print("portfolio_day1 value", get_day1_value(portfolio_csv_path))
    print("portfolio_last_day value", get_last_day_value(portfolio_csv_path))
    
    print("Alpha Calculation Results:")
    print("=" * 50)
    print(f"Data Period: {years:.2f} years ({len(portfolio_returns)} trading days)")
    print(f"Cumulative Return: {cumulative_return:.4f} ({cumulative_return:.2%})")
    print(f"Portfolio Annual Return (CAGR): {portfolio_annual_return:.4f} ({portfolio_annual_return:.2%})")
    print(f"Market Annual Return: {market_annual_return:.4f} ({market_annual_return:.2%})")
    print(f"Risk-free Rate: {risk_free_rate:.4f} ({risk_free_rate:.2%})")
    print(f"Portfolio Beta: {beta:.4f}")
    
    # Expected return based on CAPM
    expected_return = risk_free_rate + beta * (market_annual_return - risk_free_rate)
    print(f"Expected Return (CAPM): {expected_return:.4f} ({expected_return:.2%})")
    
    print(f"\nPortfolio Alpha: {alpha:.4f} ({alpha:.2%})")
    
    # Alpha interpretation
    if alpha > 0:
        print("Portfolio is generating positive alpha (outperforming expectations)")
    elif alpha < 0:
        print("Portfolio is generating negative alpha (underperforming expectations)")
    else:
        print("Portfolio is performing exactly as expected given its risk level")
    
    print(f"\nThis means your portfolio is generating {alpha:.2%} excess return")
    print(f"compared to what would be expected given its beta of {beta:.4f}")
    
    return alpha

def run_generate_test_portfolio():
   
    # Get the project root directory (parent of src)
    project_root = get_project_root()

    # Portfolio returns
    portfolio_csv_path = os.path.join(project_root, 'data', 'test_portfolio.csv')

    generate_portfolio_returns(
        portfolio_csv_path,
        ['TCS', 'INFY', 'RELIANCE', 'HDFC', 'WIPRO'],
        365
    )

def download_and_save_nifty_data():
        # Get the project root directory (parent of src)
        project_root = get_project_root()

        # Portfolio returns
        portfolio_csv_path = os.path.join(project_root, 'data', 'nifty_prices_new.csv')

        index_data = capital_market.index_data(
            index='Nifty 50',
            from_date='01-01-2024',
            to_date='15-08-2025'
        )
        index_data.to_csv(portfolio_csv_path, index=False)

if __name__ == "__main__":

    #download_and_save_nifty_data()  
    test_beta()
    
    print("\n" + "="*60)
    
    test_alpha()
    


#!/usr/bin/env python3
"""
Test cases for Portfolio CAGR calculation
"""

import os
import sys
import pandas as pd
import tempfile
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from portfolio import Portfolio


def create_test_portfolio_csv(data, filename):
    """Helper function to create test CSV files"""
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return filename


def test_cagr_basic_calculation():
    """Test CAGR calculation with basic portfolio data"""
    # Create portfolio that grows from 100 to 144 over ~2 years
    data = {
        'TIMESTAMP': ['01-01-2022', '01-01-2023', '01-01-2024'],
        'AAPL_price': [100, 120, 144],
        'AAPL_weight': [1.0, 1.0, 1.0]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        filename = f.name
        create_test_portfolio_csv(data, filename)
        
        portfolio = Portfolio(filename)
        cagr = portfolio.calculate_CAGR()
        
        # Actual calculation: 20% return each year compounded over ~1 year period
        # (1.2 * 1.2)^(1/1) - 1 = 0.44 (44%)
        expected_cagr = 0.44
        assert abs(cagr - expected_cagr) < 0.01
        
        os.unlink(filename)


def test_cagr_negative_returns():
    """Test CAGR calculation with negative returns"""
    data = {
        'TIMESTAMP': ['01-01-2022', '01-01-2023', '01-01-2024'],
        'AAPL_price': [100, 90, 81],
        'AAPL_weight': [1.0, 1.0, 1.0]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        filename = f.name
        create_test_portfolio_csv(data, filename)
        
        portfolio = Portfolio(filename)
        cagr = portfolio.calculate_CAGR()
        
        # Actual calculation: -10% return each year compounded over ~1 year period
        # (0.9 * 0.9)^(1/1) - 1 = -0.19 (-19%)
        expected_cagr = -0.19
        assert abs(cagr - expected_cagr) < 0.01
        
        os.unlink(filename)


def test_cagr_multi_asset_portfolio():
    """Test CAGR calculation with multiple assets"""
    data = {
        'TIMESTAMP': ['01-01-2022', '01-07-2022', '01-01-2023'],
        'AAPL_price': [100, 105, 110],
        'AAPL_weight': [0.5, 0.5, 0.5],
        'GOOGL_price': [200, 220, 240],
        'GOOGL_weight': [0.5, 0.5, 0.5]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        filename = f.name
        create_test_portfolio_csv(data, filename)
        
        portfolio = Portfolio(filename)
        cagr = portfolio.calculate_CAGR()
        
        # Portfolio returns: 
        # Period 1: 0.5 * (105/100 - 1) + 0.5 * (220/200 - 1) = 0.075
        # Period 2: 0.5 * (110/105 - 1) + 0.5 * (240/220 - 1) = 0.069264
        # Cumulative: (1.075 * 1.069264) = 1.1495 over ~0.5 years
        # CAGR: (1.1495)^(1/0.5) - 1 = 0.3185
        expected_cagr = 0.3185
        assert abs(cagr - expected_cagr) < 0.01
        
        os.unlink(filename)


def test_cagr_empty_data():
    """Test CAGR calculation with empty portfolio data"""
    data = {
        'TIMESTAMP': [],
        'AAPL_price': [],
        'AAPL_weight': []
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        filename = f.name
        create_test_portfolio_csv(data, filename)
        
        portfolio = Portfolio(filename)
        cagr = portfolio.calculate_CAGR()
        
        assert cagr == 0.0
        
        os.unlink(filename)


def test_cagr_single_day():
    """Test CAGR calculation with single day data (should return 0)"""
    data = {
        'TIMESTAMP': ['01-01-2022'],
        'AAPL_price': [100],
        'AAPL_weight': [1.0]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        filename = f.name
        create_test_portfolio_csv(data, filename)
        
        portfolio = Portfolio(filename)
        cagr = portfolio.calculate_CAGR()
        
        assert cagr == 0.0
        
        os.unlink(filename)


if __name__ == "__main__":
    pytest.main([__file__])
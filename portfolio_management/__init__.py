"""
Portfolio Management Package

A comprehensive portfolio management framework for advanced investment analysis,
risk management, and portfolio optimization.

Key Components:
- Core: Portfolio, Position, and Transaction management
- Analytics: Risk analysis, optimization, and performance attribution
- Data: Pluggable data providers for market data
- Strategies: Investment strategy framework
- Rebalancing: Automated rebalancing engine
- Utils: Validation and utility functions

Author: Portfolio Management Framework
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Portfolio Management Framework"

# Core imports
from .core.portfolio import Portfolio
from .core.position import Position
from .core.transaction import Transaction

# Analytics imports
from .analytics.risk_manager import RiskManager
from .analytics.optimizer import PortfolioOptimizer
# from .analytics.performance import PerformanceAnalyzer  # TODO: Implement

# Data provider imports
# from .data.providers import DataProvider, CSVDataProvider  # TODO: Implement

# Strategy imports
# from .strategies.base import BaseStrategy  # TODO: Implement

# Rebalancing imports
# from .rebalancing.engine import RebalancingEngine  # TODO: Implement

# Utility imports
# from .utils.validators import PortfolioValidator  # TODO: Implement

__all__ = [
    # Core
    'Portfolio',
    'Position', 
    'Transaction',
    # Analytics
    'RiskManager',
    'PortfolioOptimizer',
    # Data
    # 'DataProvider',
    # 'CSVDataProvider',
    # Strategies
    # 'BaseStrategy',
    # Rebalancing
    # 'RebalancingEngine',
    # Utils
    # 'PortfolioValidator'
]
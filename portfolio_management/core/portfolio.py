"""
Advanced Portfolio Management Module

This module provides the Portfolio class that serves as the main container
for managing positions, transactions, and portfolio-level analytics.
"""

from datetime import datetime, date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any, Union
import uuid
import json
from collections import defaultdict

from .position import Position, PositionType
from .transaction import Transaction, TransactionType, TransactionStatus, TransactionFees, TransactionManager


class PortfolioType(str):
    """Portfolio type constants."""
    GROWTH = "growth"
    INCOME = "income"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    CUSTOM = "custom"


class Portfolio:
    """
    Advanced Portfolio Management System.
    
    This class provides comprehensive portfolio management including:
    - Position tracking with cost basis and P&L
    - Transaction management and audit trail
    - Cash management and dividend tracking
    - Performance analytics and attribution
    - Risk monitoring and compliance
    - Tax-loss harvesting and optimization
    - Rebalancing recommendations
    """
    
    def __init__(self, 
                 name: str,
                 portfolio_type: str = PortfolioType.CUSTOM,
                 base_currency: str = "USD",
                 initial_cash: Decimal = Decimal('0'),
                 benchmark_symbol: str = "SPY"):
        """
        Initialize a new portfolio.
        
        Args:
            name: Portfolio name
            portfolio_type: Type of portfolio strategy
            base_currency: Base currency for portfolio
            initial_cash: Initial cash balance
            benchmark_symbol: Benchmark for performance comparison
        """
        self.portfolio_id = str(uuid.uuid4())
        self.name = name
        self.portfolio_type = portfolio_type
        self.base_currency = base_currency
        self.benchmark_symbol = benchmark_symbol
        
        # Core components
        self.positions: Dict[str, Position] = {}
        self.transaction_manager = TransactionManager()
        
        # Cash management
        self.cash_balance = initial_cash
        self.initial_value = initial_cash
        self.committed_cash = Decimal('0')  # Cash allocated for pending orders
        
        # Performance tracking
        self.total_deposits = initial_cash
        self.total_withdrawals = Decimal('0')
        self.total_dividends = Decimal('0')
        self.total_fees = Decimal('0')
        
        # Portfolio metrics
        self.target_weights: Dict[str, Decimal] = {}
        self.risk_tolerance = Decimal('0.15')  # 15% default volatility target
        self.max_position_size = Decimal('0.20')  # 20% max single position
        
        # Timestamps
        self.created_date = datetime.now()
        self.last_updated = datetime.now()
        self.last_rebalance_date = None
        
        # Settings
        self.auto_reinvest_dividends = True
        self.tax_loss_harvesting = False
        self.fractional_shares_enabled = False
        
        # Compliance and constraints
        self.compliance_rules: Dict[str, Any] = {}
        self.position_limits: Dict[str, Decimal] = {}
        
        # Performance history
        self.daily_values: List[Dict] = []
        self.performance_snapshots: List[Dict] = []
    
    def add_cash(self, amount: Decimal, description: str = "Cash deposit") -> str:
        """
        Add cash to the portfolio.
        
        Args:
            amount: Amount of cash to add
            description: Description of the deposit
            
        Returns:
            Transaction ID
        """
        transaction = Transaction(
            transaction_type=TransactionType.CASH_DEPOSIT,
            symbol="CASH",
            quantity=Decimal('1'),
            price=amount,
            description=description
        )
        
        transaction_id = self.transaction_manager.add_transaction(transaction)
        transaction.settle_transaction()
        
        self.cash_balance += amount
        self.total_deposits += amount
        self._update_timestamp()
        
        return transaction_id
    
    def withdraw_cash(self, amount: Decimal, description: str = "Cash withdrawal") -> str:
        """
        Withdraw cash from the portfolio.
        
        Args:
            amount: Amount of cash to withdraw
            description: Description of the withdrawal
            
        Returns:
            Transaction ID
        """
        if amount > self.cash_balance:
            raise ValueError(f"Insufficient cash balance. Available: {self.cash_balance}, Requested: {amount}")
        
        transaction = Transaction(
            transaction_type=TransactionType.CASH_WITHDRAWAL,
            symbol="CASH",
            quantity=Decimal('1'),
            price=amount,
            description=description
        )
        
        transaction_id = self.transaction_manager.add_transaction(transaction)
        transaction.settle_transaction()
        
        self.cash_balance -= amount
        self.total_withdrawals += amount
        self._update_timestamp()
        
        return transaction_id
    
    def buy_shares(self, 
                   symbol: str, 
                   shares: Decimal, 
                   price: Decimal,
                   transaction_date: date = None,
                   fees: TransactionFees = None,
                   order_type: str = "market") -> str:
        """
        Buy shares of an asset.
        
        Args:
            symbol: Asset symbol
            shares: Number of shares to buy
            price: Price per share
            transaction_date: Transaction date
            fees: Transaction fees
            order_type: Type of order
            
        Returns:
            Transaction ID
        """
        if fees is None:
            fees = TransactionFees()
        
        total_cost = shares * price + fees.total_fees
        
        if total_cost > self.cash_balance:
            raise ValueError(f"Insufficient cash. Required: {total_cost}, Available: {self.cash_balance}")
        
        # Create transaction
        transaction = Transaction(
            transaction_type=TransactionType.BUY,
            symbol=symbol,
            quantity=shares,
            price=price,
            transaction_date=transaction_date,
            fees=fees,
            description=f"{order_type.upper()} buy order for {shares} shares of {symbol}"
        )
        
        transaction_id = self.transaction_manager.add_transaction(transaction)
        
        # Update position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        
        lot_id = self.positions[symbol].add_shares(
            shares=shares,
            price_per_share=price,
            transaction_date=transaction_date or date.today(),
            transaction_costs=fees.total_fees
        )
        
        # Update transaction with tax lot information
        cost_basis_per_share = (shares * price + fees.total_fees) / shares
        transaction.update_tax_information(lot_id, cost_basis_per_share)
        
        # Update cash and settle transaction
        self.cash_balance -= total_cost
        self.total_fees += fees.total_fees
        transaction.settle_transaction()
        
        self._update_timestamp()
        return transaction_id
    
    def sell_shares(self, 
                    symbol: str, 
                    shares: Decimal, 
                    price: Decimal,
                    transaction_date: date = None,
                    fees: TransactionFees = None,
                    order_type: str = "market") -> str:
        """
        Sell shares of an asset.
        
        Args:
            symbol: Asset symbol
            shares: Number of shares to sell
            price: Price per share
            transaction_date: Transaction date
            fees: Transaction fees
            order_type: Type of order
            
        Returns:
            Transaction ID
        """
        if symbol not in self.positions:
            raise ValueError(f"No position found for {symbol}")
        
        position = self.positions[symbol]
        if shares > position.total_shares:
            raise ValueError(f"Cannot sell {shares} shares. Position has {position.total_shares} shares")
        
        if fees is None:
            fees = TransactionFees()
        
        # Create transaction
        transaction = Transaction(
            transaction_type=TransactionType.SELL,
            symbol=symbol,
            quantity=shares,
            price=price,
            transaction_date=transaction_date,
            fees=fees,
            description=f"{order_type.upper()} sell order for {shares} shares of {symbol}"
        )
        
        transaction_id = self.transaction_manager.add_transaction(transaction)
        
        # Execute sale
        sale_details = position.remove_shares(
            shares=shares,
            price_per_share=price,
            transaction_date=transaction_date or date.today(),
            transaction_costs=fees.total_fees
        )
        
        # Update transaction with sale details
        transaction.update_tax_information(
            tax_lot_id=",".join([lot.lot_id for lot in sale_details['sold_lots']]),
            cost_basis_per_share=sale_details['cost_basis_sold'] / shares,
            realized_pnl=sale_details['realized_pnl']
        )
        
        # Update cash and settle transaction
        net_proceeds = sale_details['net_proceeds']
        self.cash_balance += net_proceeds
        self.total_fees += fees.total_fees
        transaction.settle_transaction()
        
        # Remove position if fully sold
        if position.total_shares == 0:
            del self.positions[symbol]
        
        self._update_timestamp()
        return transaction_id
    
    def record_dividend(self, 
                       symbol: str, 
                       dividend_per_share: Decimal,
                       ex_date: date = None,
                       pay_date: date = None) -> str:
        """
        Record dividend payment.
        
        Args:
            symbol: Asset symbol
            dividend_per_share: Dividend amount per share
            ex_date: Ex-dividend date
            pay_date: Payment date
            
        Returns:
            Transaction ID
        """
        if symbol not in self.positions:
            raise ValueError(f"No position found for {symbol}")
        
        position = self.positions[symbol]
        total_dividend = position.total_shares * dividend_per_share
        
        transaction = Transaction(
            transaction_type=TransactionType.DIVIDEND,
            symbol=symbol,
            quantity=position.total_shares,
            price=dividend_per_share,
            transaction_date=pay_date or date.today(),
            description=f"Dividend payment: {dividend_per_share} per share"
        )
        
        transaction.add_metadata("ex_date", ex_date.isoformat() if ex_date else None)
        transaction.add_metadata("pay_date", pay_date.isoformat() if pay_date else None)
        
        transaction_id = self.transaction_manager.add_transaction(transaction)
        
        # Add dividend to cash
        self.cash_balance += total_dividend
        self.total_dividends += total_dividend
        transaction.settle_transaction()
        
        # Auto-reinvest if enabled
        if self.auto_reinvest_dividends and total_dividend > 0:
            try:
                # This would need current market price - simplified for example
                self._reinvest_dividend(symbol, total_dividend, transaction_date=pay_date)
            except Exception:
                # If reinvestment fails, keep dividend as cash
                pass
        
        self._update_timestamp()
        return transaction_id
    
    def _reinvest_dividend(self, symbol: str, dividend_amount: Decimal, transaction_date: date = None):
        """
        Reinvest dividend by purchasing additional shares.
        
        Args:
            symbol: Asset symbol
            dividend_amount: Amount to reinvest
            transaction_date: Transaction date
        """
        # This is a simplified implementation
        # In a real system, you'd need current market price
        if symbol in self.positions:
            current_price = self.positions[symbol].current_price
            if current_price > 0:
                shares_to_buy = dividend_amount / current_price
                
                if self.fractional_shares_enabled or shares_to_buy >= 1:
                    self.buy_shares(
                        symbol=symbol,
                        shares=shares_to_buy,
                        price=current_price,
                        transaction_date=transaction_date,
                        fees=TransactionFees()  # Usually no fees for dividend reinvestment
                    )
    
    def update_market_prices(self, prices: Dict[str, Decimal], timestamp: datetime = None):
        """
        Update market prices for all positions.
        
        Args:
            prices: Dictionary of symbol -> price
            timestamp: Timestamp of price update
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.update_market_price(prices[symbol], timestamp)
        
        self._update_timestamp()
        
        # Record daily value snapshot
        self._record_daily_value(timestamp)
    
    def _record_daily_value(self, timestamp: datetime):
        """Record daily portfolio value for performance tracking."""
        total_value = self.get_total_value()
        
        daily_record = {
            'date': timestamp.date().isoformat(),
            'timestamp': timestamp.isoformat(),
            'total_value': float(total_value),
            'cash_balance': float(self.cash_balance),
            'market_value': float(self.get_market_value()),
            'unrealized_pnl': float(self.get_unrealized_pnl()),
            'realized_pnl': float(self.get_realized_pnl())
        }
        
        # Avoid duplicate daily records
        if not self.daily_values or self.daily_values[-1]['date'] != daily_record['date']:
            self.daily_values.append(daily_record)
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions."""
        return self.positions.copy()
    
    def get_cash_balance(self) -> Decimal:
        """Get current cash balance."""
        return self.cash_balance
    
    def get_market_value(self) -> Decimal:
        """Get total market value of all positions."""
        return sum(position.market_value for position in self.positions.values())
    
    def get_total_value(self) -> Decimal:
        """Get total portfolio value (cash + market value)."""
        return self.cash_balance + self.get_market_value()
    
    def get_total_cost_basis(self) -> Decimal:
        """Get total cost basis of all positions."""
        return sum(position.total_cost_basis for position in self.positions.values())
    
    def get_unrealized_pnl(self) -> Decimal:
        """Get total unrealized P&L."""
        return sum(position.unrealized_pnl for position in self.positions.values())
    
    def get_realized_pnl(self) -> Decimal:
        """Get total realized P&L from all transactions."""
        realized_pnl = sum(position.realized_pnl for position in self.positions.values())
        
        # Add realized P&L from sold positions (from transactions)
        for transaction in self.transaction_manager.transactions.values():
            if transaction.transaction_type == TransactionType.SELL:
                realized_pnl += transaction.realized_pnl
        
        return realized_pnl
    
    def get_total_pnl(self) -> Decimal:
        """Get total P&L (realized + unrealized)."""
        return self.get_realized_pnl() + self.get_unrealized_pnl()
    
    def get_position_weights(self) -> Dict[str, Decimal]:
        """
        Get current position weights as percentage of total portfolio value.
        
        Returns:
            Dictionary of symbol -> weight (as decimal, e.g., 0.25 = 25%)
        """
        total_value = self.get_total_value()
        if total_value == 0:
            return {}
        
        weights = {}
        for symbol, position in self.positions.items():
            weights[symbol] = position.market_value / total_value
        
        # Add cash weight
        weights['CASH'] = self.cash_balance / total_value
        
        return weights
    
    def set_target_weights(self, target_weights: Dict[str, Decimal]):
        """
        Set target allocation weights for the portfolio.
        
        Args:
            target_weights: Dictionary of symbol -> target weight (as decimal)
        """
        # Validate that weights sum to 1.0 (allow small tolerance)
        total_weight = sum(target_weights.values())
        if abs(total_weight - Decimal('1.0')) > Decimal('0.01'):
            raise ValueError(f"Target weights must sum to 1.0, got {total_weight}")
        
        self.target_weights = target_weights.copy()
        self._update_timestamp()
    
    def calculate_rebalancing_needs(self, tolerance: Decimal = Decimal('0.05')) -> Dict[str, Dict]:
        """
        Calculate rebalancing needs based on target weights.
        
        Args:
            tolerance: Tolerance for weight deviation (default 5%)
            
        Returns:
            Dictionary with rebalancing recommendations
        """
        if not self.target_weights:
            return {}
        
        current_weights = self.get_position_weights()
        total_value = self.get_total_value()
        rebalancing_needs = {}
        
        for symbol, target_weight in self.target_weights.items():
            current_weight = current_weights.get(symbol, Decimal('0'))
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) > tolerance:
                target_value = target_weight * total_value
                current_value = current_weight * total_value
                value_diff = target_value - current_value
                
                if symbol == 'CASH':
                    shares_diff = Decimal('0')
                    current_price = Decimal('1')
                else:
                    position = self.positions.get(symbol)
                    current_price = position.current_price if position else Decimal('0')
                    shares_diff = value_diff / current_price if current_price > 0 else Decimal('0')
                
                rebalancing_needs[symbol] = {
                    'current_weight': float(current_weight),
                    'target_weight': float(target_weight),
                    'weight_difference': float(weight_diff),
                    'current_value': float(current_value),
                    'target_value': float(target_value),
                    'value_difference': float(value_diff),
                    'shares_difference': float(shares_diff),
                    'action': 'buy' if value_diff > 0 else 'sell',
                    'current_price': float(current_price)
                }
        
        return rebalancing_needs
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive portfolio summary.
        
        Returns:
            Dictionary with complete portfolio information
        """
        total_value = self.get_total_value()
        market_value = self.get_market_value()
        
        return {
            'portfolio_id': self.portfolio_id,
            'name': self.name,
            'portfolio_type': self.portfolio_type,
            'base_currency': self.base_currency,
            'benchmark_symbol': self.benchmark_symbol,
            
            # Values
            'total_value': float(total_value),
            'market_value': float(market_value),
            'cash_balance': float(self.cash_balance),
            'total_cost_basis': float(self.get_total_cost_basis()),
            
            # Performance
            'unrealized_pnl': float(self.get_unrealized_pnl()),
            'realized_pnl': float(self.get_realized_pnl()),
            'total_pnl': float(self.get_total_pnl()),
            'total_return_pct': float(self.get_total_pnl() / self.initial_value * 100) if self.initial_value > 0 else 0.0,
            
            # Cash flows
            'total_deposits': float(self.total_deposits),
            'total_withdrawals': float(self.total_withdrawals),
            'total_dividends': float(self.total_dividends),
            'total_fees': float(self.total_fees),
            'net_cash_flow': float(self.total_deposits - self.total_withdrawals),
            
            # Positions
            'number_of_positions': len(self.positions),
            'position_weights': {k: float(v) for k, v in self.get_position_weights().items()},
            'target_weights': {k: float(v) for k, v in self.target_weights.items()},
            
            # Transactions
            'total_transactions': len(self.transaction_manager.transactions),
            'unsettled_transactions': len(self.transaction_manager.get_unsettled_transactions()),
            'unreconciled_transactions': len(self.transaction_manager.get_unreconciled_transactions()),
            
            # Settings
            'auto_reinvest_dividends': self.auto_reinvest_dividends,
            'tax_loss_harvesting': self.tax_loss_harvesting,
            'fractional_shares_enabled': self.fractional_shares_enabled,
            'risk_tolerance': float(self.risk_tolerance),
            'max_position_size': float(self.max_position_size),
            
            # Timestamps
            'created_date': self.created_date.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'last_rebalance_date': self.last_rebalance_date.isoformat() if self.last_rebalance_date else None
        }
    
    def get_detailed_positions(self) -> List[Dict]:
        """Get detailed information for all positions."""
        return [position.get_position_summary() for position in self.positions.values()]
    
    def get_transaction_history(self, 
                              start_date: date = None, 
                              end_date: date = None,
                              symbol: str = None) -> List[Dict]:
        """
        Get transaction history with optional filtering.
        
        Args:
            start_date: Start date filter
            end_date: End date filter
            symbol: Symbol filter
            
        Returns:
            List of transaction summaries
        """
        transactions = list(self.transaction_manager.transactions.values())
        
        if start_date and end_date:
            transactions = [
                t for t in transactions 
                if start_date <= t.transaction_date <= end_date
            ]
        
        if symbol:
            transactions = [t for t in transactions if t.symbol == symbol]
        
        return [t.get_transaction_summary() for t in sorted(transactions, key=lambda x: x.transaction_date)]
    
    def calculate_performance_metrics(self, period_days: int = 365) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics.
        
        Args:
            period_days: Period for performance calculation
            
        Returns:
            Dictionary with performance metrics
        """
        if len(self.daily_values) < 2:
            return {}
        
        # Get values for the period
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=period_days)
        
        period_values = [
            v for v in self.daily_values 
            if start_date <= datetime.fromisoformat(v['date']).date() <= end_date
        ]
        
        if len(period_values) < 2:
            return {}
        
        # Calculate returns
        start_value = period_values[0]['total_value']
        end_value = period_values[-1]['total_value']
        total_return = (end_value - start_value) / start_value if start_value > 0 else 0
        
        # Calculate daily returns for volatility
        daily_returns = []
        for i in range(1, len(period_values)):
            prev_value = period_values[i-1]['total_value']
            curr_value = period_values[i]['total_value']
            daily_return = (curr_value - prev_value) / prev_value if prev_value > 0 else 0
            daily_returns.append(daily_return)
        
        if not daily_returns:
            return {}
        
        # Calculate volatility (standard deviation of daily returns)
        mean_return = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean_return) ** 2 for r in daily_returns) / len(daily_returns)
        volatility = variance ** 0.5
        
        # Annualize metrics
        trading_days = 252
        annualized_return = (1 + total_return) ** (trading_days / len(period_values)) - 1
        annualized_volatility = volatility * (trading_days ** 0.5)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
        
        return {
            'period_days': period_days,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self._calculate_max_drawdown(period_values),
            'number_of_observations': len(period_values)
        }
    
    def _calculate_max_drawdown(self, values: List[Dict]) -> float:
        """Calculate maximum drawdown from value series."""
        if len(values) < 2:
            return 0.0
        
        peak = values[0]['total_value']
        max_drawdown = 0.0
        
        for value_record in values[1:]:
            current_value = value_record['total_value']
            
            if current_value > peak:
                peak = current_value
            else:
                drawdown = (peak - current_value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _update_timestamp(self):
        """Update the last modified timestamp."""
        self.last_updated = datetime.now()
    
    def export_to_json(self) -> str:
        """Export portfolio to JSON format."""
        data = self.get_portfolio_summary()
        data['positions'] = self.get_detailed_positions()
        data['transactions'] = self.get_transaction_history()
        data['daily_values'] = self.daily_values
        return json.dumps(data, indent=2, default=str)
    
    def generate_tax_report(self, tax_year: int) -> Dict[str, Any]:
        """
        Generate tax report for a specific year.
        
        Args:
            tax_year: Tax year for the report
            
        Returns:
            Tax report dictionary
        """
        start_date = date(tax_year, 1, 1)
        end_date = date(tax_year, 12, 31)
        
        # Get all sell transactions for the year
        sell_transactions = [
            t for t in self.transaction_manager.transactions.values()
            if (t.transaction_type == TransactionType.SELL and 
                start_date <= t.transaction_date <= end_date)
        ]
        
        # Get dividend transactions
        dividend_transactions = [
            t for t in self.transaction_manager.transactions.values()
            if (t.transaction_type == TransactionType.DIVIDEND and 
                start_date <= t.transaction_date <= end_date)
        ]
        
        total_realized_gains = sum(t.realized_pnl for t in sell_transactions)
        total_dividends = sum(t.gross_amount for t in dividend_transactions)
        total_fees = sum(t.fees.total_fees for t in self.transaction_manager.transactions.values()
                        if start_date <= t.transaction_date <= end_date)
        
        return {
            'tax_year': tax_year,
            'total_realized_gains': float(total_realized_gains),
            'total_dividends': float(total_dividends),
            'total_fees': float(total_fees),
            'sell_transactions': [t.get_tax_report_data() for t in sell_transactions],
            'dividend_transactions': [t.get_tax_report_data() for t in dividend_transactions],
            'summary': {
                'number_of_sales': len(sell_transactions),
                'number_of_dividend_payments': len(dividend_transactions),
                'net_investment_income': float(total_dividends),
                'net_capital_gains': float(total_realized_gains)
            }
        }
    
    def __str__(self) -> str:
        """String representation of portfolio."""
        return f"Portfolio('{self.name}', {len(self.positions)} positions, ${self.get_total_value():,.2f})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"Portfolio(id={self.portfolio_id}, name='{self.name}', "
                f"type={self.portfolio_type}, value=${self.get_total_value()})")
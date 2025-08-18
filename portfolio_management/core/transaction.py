"""
Transaction Management Module

This module provides comprehensive transaction recording and audit trail
functionality for portfolio management.
"""

from datetime import datetime, date
from decimal import Decimal
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json


class TransactionType(Enum):
    """Transaction type enumeration."""
    BUY = "buy"
    SELL = "sell"
    DIVIDEND = "dividend"
    SPLIT = "split"
    SPIN_OFF = "spin_off"
    MERGER = "merger"
    CASH_DEPOSIT = "cash_deposit"
    CASH_WITHDRAWAL = "cash_withdrawal"
    FEE = "fee"
    INTEREST = "interest"
    TAX = "tax"
    ADJUSTMENT = "adjustment"


class TransactionStatus(Enum):
    """Transaction status enumeration."""
    PENDING = "pending"
    SETTLED = "settled"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class TransactionFees:
    """Structure for transaction fees and costs."""
    commission: Decimal = Decimal('0')
    sec_fee: Decimal = Decimal('0')  # SEC fee
    finra_fee: Decimal = Decimal('0')  # FINRA fee
    exchange_fee: Decimal = Decimal('0')
    other_fees: Decimal = Decimal('0')
    total_fees: Decimal = field(init=False)
    
    def __post_init__(self):
        """Calculate total fees."""
        self.total_fees = (self.commission + self.sec_fee + 
                          self.finra_fee + self.exchange_fee + self.other_fees)


class Transaction:
    """
    Comprehensive transaction record for portfolio management.
    
    This class provides:
    - Complete transaction audit trail
    - Multiple transaction types (buy, sell, dividend, etc.)
    - Fee and cost tracking
    - Settlement and reconciliation
    - Tax reporting support
    - Performance attribution
    """
    
    def __init__(self,
                 transaction_type: TransactionType,
                 symbol: str,
                 quantity: Decimal = Decimal('0'),
                 price: Decimal = Decimal('0'),
                 transaction_date: date = None,
                 settlement_date: date = None,
                 fees: TransactionFees = None,
                 description: str = "",
                 metadata: Dict[str, Any] = None):
        """
        Initialize a new transaction.
        
        Args:
            transaction_type: Type of transaction
            symbol: Asset symbol
            quantity: Number of shares/units
            price: Price per share/unit
            transaction_date: Date transaction was executed
            settlement_date: Date transaction settles
            fees: Transaction fees structure
            description: Human-readable description
            metadata: Additional transaction metadata
        """
        self.transaction_id = str(uuid.uuid4())
        self.transaction_type = transaction_type
        self.symbol = symbol
        self.quantity = quantity
        self.price = price
        
        # Dates
        self.transaction_date = transaction_date or date.today()
        self.settlement_date = settlement_date or self._calculate_settlement_date()
        self.created_timestamp = datetime.now()
        self.last_modified = datetime.now()
        
        # Financial details
        self.fees = fees or TransactionFees()
        self.gross_amount = self._calculate_gross_amount()
        self.net_amount = self._calculate_net_amount()
        
        # Status and metadata
        self.status = TransactionStatus.PENDING
        self.description = description
        self.metadata = metadata or {}
        
        # Reconciliation
        self.reconciled = False
        self.reconciliation_date = None
        self.reconciliation_notes = ""
        
        # Tax and reporting
        self.tax_lot_id = None
        self.cost_basis_per_share = Decimal('0')
        self.realized_pnl = Decimal('0')
        
        # Audit trail
        self.audit_trail: List[Dict] = []
        self._add_audit_entry("Transaction created")
    
    def _calculate_settlement_date(self) -> date:
        """
        Calculate settlement date based on transaction type and date.
        Default is T+2 for stock transactions.
        """
        from datetime import timedelta
        
        if self.transaction_type in [TransactionType.BUY, TransactionType.SELL]:
            # T+2 settlement for stocks
            settlement_days = 2
        elif self.transaction_type == TransactionType.DIVIDEND:
            # Dividends typically settle same day
            settlement_days = 0
        else:
            # Default to T+1 for other transactions
            settlement_days = 1
        
        return self.transaction_date + timedelta(days=settlement_days)
    
    def _calculate_gross_amount(self) -> Decimal:
        """Calculate gross transaction amount (before fees)."""
        if self.transaction_type in [TransactionType.BUY, TransactionType.SELL]:
            return self.quantity * self.price
        elif self.transaction_type == TransactionType.DIVIDEND:
            return self.quantity * self.price  # quantity = shares, price = dividend per share
        else:
            return self.price  # For other types, price might be the total amount
    
    def _calculate_net_amount(self) -> Decimal:
        """Calculate net transaction amount (after fees)."""
        if self.transaction_type == TransactionType.BUY:
            # For buys, add fees to cost
            return self.gross_amount + self.fees.total_fees
        elif self.transaction_type == TransactionType.SELL:
            # For sells, subtract fees from proceeds
            return self.gross_amount - self.fees.total_fees
        else:
            # For other types, typically subtract fees
            return self.gross_amount - self.fees.total_fees
    
    def _add_audit_entry(self, action: str, details: Dict[str, Any] = None):
        """Add entry to audit trail."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details or {},
            'user': 'system'  # In a real system, this would be the actual user
        }
        self.audit_trail.append(entry)
        self.last_modified = datetime.now()
    
    def update_status(self, new_status: TransactionStatus, notes: str = ""):
        """
        Update transaction status with audit trail.
        
        Args:
            new_status: New transaction status
            notes: Optional notes about the status change
        """
        old_status = self.status
        self.status = new_status
        
        self._add_audit_entry(
            f"Status changed from {old_status.value} to {new_status.value}",
            {'old_status': old_status.value, 'new_status': new_status.value, 'notes': notes}
        )
    
    def settle_transaction(self, actual_settlement_date: date = None):
        """
        Mark transaction as settled.
        
        Args:
            actual_settlement_date: Actual settlement date (defaults to expected)
        """
        if actual_settlement_date:
            self.settlement_date = actual_settlement_date
        
        self.update_status(TransactionStatus.SETTLED, "Transaction settled")
    
    def cancel_transaction(self, reason: str = ""):
        """
        Cancel the transaction.
        
        Args:
            reason: Reason for cancellation
        """
        self.update_status(TransactionStatus.CANCELLED, f"Transaction cancelled: {reason}")
    
    def reconcile_transaction(self, notes: str = ""):
        """
        Mark transaction as reconciled.
        
        Args:
            notes: Reconciliation notes
        """
        self.reconciled = True
        self.reconciliation_date = datetime.now()
        self.reconciliation_notes = notes
        
        self._add_audit_entry("Transaction reconciled", {'notes': notes})
    
    def update_tax_information(self, 
                              tax_lot_id: str, 
                              cost_basis_per_share: Decimal,
                              realized_pnl: Decimal = None):
        """
        Update tax-related information for the transaction.
        
        Args:
            tax_lot_id: Associated tax lot ID
            cost_basis_per_share: Cost basis per share
            realized_pnl: Realized P&L (for sell transactions)
        """
        self.tax_lot_id = tax_lot_id
        self.cost_basis_per_share = cost_basis_per_share
        
        if realized_pnl is not None:
            self.realized_pnl = realized_pnl
        
        self._add_audit_entry(
            "Tax information updated",
            {
                'tax_lot_id': tax_lot_id,
                'cost_basis_per_share': float(cost_basis_per_share),
                'realized_pnl': float(realized_pnl) if realized_pnl else None
            }
        )
    
    def add_metadata(self, key: str, value: Any):
        """
        Add metadata to the transaction.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
        self._add_audit_entry(f"Metadata added: {key}", {'key': key, 'value': value})
    
    def get_transaction_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive transaction summary.
        
        Returns:
            Dictionary with complete transaction information
        """
        return {
            'transaction_id': self.transaction_id,
            'transaction_type': self.transaction_type.value,
            'symbol': self.symbol,
            'quantity': float(self.quantity),
            'price': float(self.price),
            
            # Dates
            'transaction_date': self.transaction_date.isoformat(),
            'settlement_date': self.settlement_date.isoformat(),
            'created_timestamp': self.created_timestamp.isoformat(),
            'last_modified': self.last_modified.isoformat(),
            
            # Financial
            'gross_amount': float(self.gross_amount),
            'net_amount': float(self.net_amount),
            'fees': {
                'commission': float(self.fees.commission),
                'sec_fee': float(self.fees.sec_fee),
                'finra_fee': float(self.fees.finra_fee),
                'exchange_fee': float(self.fees.exchange_fee),
                'other_fees': float(self.fees.other_fees),
                'total_fees': float(self.fees.total_fees)
            },
            
            # Status
            'status': self.status.value,
            'reconciled': self.reconciled,
            'reconciliation_date': self.reconciliation_date.isoformat() if self.reconciliation_date else None,
            'reconciliation_notes': self.reconciliation_notes,
            
            # Tax
            'tax_lot_id': self.tax_lot_id,
            'cost_basis_per_share': float(self.cost_basis_per_share),
            'realized_pnl': float(self.realized_pnl),
            
            # Other
            'description': self.description,
            'metadata': self.metadata
        }
    
    def get_tax_report_data(self) -> Dict[str, Any]:
        """
        Get transaction data formatted for tax reporting.
        
        Returns:
            Dictionary with tax-relevant transaction information
        """
        return {
            'transaction_id': self.transaction_id,
            'symbol': self.symbol,
            'transaction_type': self.transaction_type.value,
            'transaction_date': self.transaction_date,
            'settlement_date': self.settlement_date,
            'quantity': self.quantity,
            'price': self.price,
            'gross_proceeds': self.gross_amount if self.transaction_type == TransactionType.SELL else None,
            'cost_basis': self.quantity * self.cost_basis_per_share if self.cost_basis_per_share > 0 else None,
            'realized_pnl': self.realized_pnl if self.transaction_type == TransactionType.SELL else None,
            'fees': self.fees.total_fees,
            'tax_lot_id': self.tax_lot_id,
            'holding_period_days': (self.transaction_date - self.settlement_date).days if self.tax_lot_id else None
        }
    
    def export_to_json(self) -> str:
        """
        Export transaction to JSON format.
        
        Returns:
            JSON string representation of the transaction
        """
        data = self.get_transaction_summary()
        data['audit_trail'] = self.audit_trail
        return json.dumps(data, indent=2, default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Transaction':
        """
        Create transaction from JSON string.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            Transaction instance
        """
        data = json.loads(json_str)
        
        # Create fees object
        fees_data = data.get('fees', {})
        fees = TransactionFees(
            commission=Decimal(str(fees_data.get('commission', 0))),
            sec_fee=Decimal(str(fees_data.get('sec_fee', 0))),
            finra_fee=Decimal(str(fees_data.get('finra_fee', 0))),
            exchange_fee=Decimal(str(fees_data.get('exchange_fee', 0))),
            other_fees=Decimal(str(fees_data.get('other_fees', 0)))
        )
        
        # Create transaction
        transaction = cls(
            transaction_type=TransactionType(data['transaction_type']),
            symbol=data['symbol'],
            quantity=Decimal(str(data['quantity'])),
            price=Decimal(str(data['price'])),
            transaction_date=date.fromisoformat(data['transaction_date']),
            settlement_date=date.fromisoformat(data['settlement_date']),
            fees=fees,
            description=data.get('description', ''),
            metadata=data.get('metadata', {})
        )
        
        # Update additional fields
        transaction.transaction_id = data['transaction_id']
        transaction.status = TransactionStatus(data['status'])
        transaction.reconciled = data['reconciled']
        if data.get('reconciliation_date'):
            transaction.reconciliation_date = datetime.fromisoformat(data['reconciliation_date'])
        transaction.reconciliation_notes = data.get('reconciliation_notes', '')
        transaction.tax_lot_id = data.get('tax_lot_id')
        transaction.cost_basis_per_share = Decimal(str(data.get('cost_basis_per_share', 0)))
        transaction.realized_pnl = Decimal(str(data.get('realized_pnl', 0)))
        transaction.audit_trail = data.get('audit_trail', [])
        
        return transaction
    
    def __str__(self) -> str:
        """String representation of transaction."""
        return (f"Transaction({self.transaction_type.value.upper()} {self.quantity} "
                f"{self.symbol} @ ${self.price} on {self.transaction_date})")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"Transaction(id={self.transaction_id}, type={self.transaction_type.value}, "
                f"symbol={self.symbol}, quantity={self.quantity}, price={self.price}, "
                f"date={self.transaction_date}, status={self.status.value})")


class TransactionManager:
    """
    Manager class for handling multiple transactions and providing
    aggregated views and analysis.
    """
    
    def __init__(self):
        """Initialize transaction manager."""
        self.transactions: Dict[str, Transaction] = {}
        self.created_date = datetime.now()
    
    def add_transaction(self, transaction: Transaction) -> str:
        """
        Add transaction to manager.
        
        Args:
            transaction: Transaction to add
            
        Returns:
            Transaction ID
        """
        self.transactions[transaction.transaction_id] = transaction
        return transaction.transaction_id
    
    def get_transaction(self, transaction_id: str) -> Optional[Transaction]:
        """Get transaction by ID."""
        return self.transactions.get(transaction_id)
    
    def get_transactions_by_symbol(self, symbol: str) -> List[Transaction]:
        """Get all transactions for a symbol."""
        return [t for t in self.transactions.values() if t.symbol == symbol]
    
    def get_transactions_by_type(self, transaction_type: TransactionType) -> List[Transaction]:
        """Get all transactions of a specific type."""
        return [t for t in self.transactions.values() if t.transaction_type == transaction_type]
    
    def get_transactions_by_date_range(self, 
                                     start_date: date, 
                                     end_date: date) -> List[Transaction]:
        """Get transactions within date range."""
        return [
            t for t in self.transactions.values() 
            if start_date <= t.transaction_date <= end_date
        ]
    
    def get_unsettled_transactions(self) -> List[Transaction]:
        """Get all unsettled transactions."""
        return [
            t for t in self.transactions.values() 
            if t.status != TransactionStatus.SETTLED
        ]
    
    def get_unreconciled_transactions(self) -> List[Transaction]:
        """Get all unreconciled transactions."""
        return [t for t in self.transactions.values() if not t.reconciled]
    
    def calculate_total_fees(self, 
                           start_date: date = None, 
                           end_date: date = None) -> Decimal:
        """
        Calculate total fees for a date range.
        
        Args:
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            Total fees
        """
        transactions = self.transactions.values()
        
        if start_date and end_date:
            transactions = [
                t for t in transactions 
                if start_date <= t.transaction_date <= end_date
            ]
        
        return sum(t.fees.total_fees for t in transactions)
    
    def generate_transaction_report(self, 
                                  start_date: date = None, 
                                  end_date: date = None) -> Dict[str, Any]:
        """
        Generate comprehensive transaction report.
        
        Args:
            start_date: Start date for report
            end_date: End date for report
            
        Returns:
            Transaction report dictionary
        """
        transactions = list(self.transactions.values())
        
        if start_date and end_date:
            transactions = [
                t for t in transactions 
                if start_date <= t.transaction_date <= end_date
            ]
        
        # Calculate summary statistics
        total_transactions = len(transactions)
        buy_transactions = len([t for t in transactions if t.transaction_type == TransactionType.BUY])
        sell_transactions = len([t for t in transactions if t.transaction_type == TransactionType.SELL])
        total_fees = sum(t.fees.total_fees for t in transactions)
        total_volume = sum(t.gross_amount for t in transactions)
        
        # Group by symbol
        by_symbol = {}
        for transaction in transactions:
            symbol = transaction.symbol
            if symbol not in by_symbol:
                by_symbol[symbol] = []
            by_symbol[symbol].append(transaction)
        
        return {
            'report_period': {
                'start_date': start_date.isoformat() if start_date else None,
                'end_date': end_date.isoformat() if end_date else None
            },
            'summary': {
                'total_transactions': total_transactions,
                'buy_transactions': buy_transactions,
                'sell_transactions': sell_transactions,
                'total_fees': float(total_fees),
                'total_volume': float(total_volume),
                'average_transaction_size': float(total_volume / total_transactions) if total_transactions > 0 else 0
            },
            'by_symbol': {
                symbol: {
                    'transaction_count': len(symbol_transactions),
                    'total_volume': float(sum(t.gross_amount for t in symbol_transactions)),
                    'total_fees': float(sum(t.fees.total_fees for t in symbol_transactions))
                }
                for symbol, symbol_transactions in by_symbol.items()
            },
            'transactions': [t.get_transaction_summary() for t in transactions]
        }
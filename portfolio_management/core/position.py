"""
Position Management Module

This module provides the Position class for tracking individual asset positions
with cost basis calculation, P&L tracking, and tax-lot management.
"""

from datetime import datetime, date
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid


class PositionType(Enum):
    """Position type enumeration."""
    LONG = "long"
    SHORT = "short"


@dataclass
class TaxLot:
    """
    Represents a single tax lot for cost basis tracking.
    
    A tax lot tracks a specific purchase of shares at a specific price
    and date, which is important for tax calculations.
    """
    lot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    purchase_date: date = field(default_factory=date.today)
    shares: Decimal = Decimal('0')
    cost_basis_per_share: Decimal = Decimal('0')
    total_cost_basis: Decimal = field(init=False)
    
    def __post_init__(self):
        """Calculate total cost basis after initialization."""
        self.total_cost_basis = self.shares * self.cost_basis_per_share
    
    def partial_sale(self, shares_sold: Decimal) -> Tuple['TaxLot', Optional['TaxLot']]:
        """
        Handle partial sale of shares from this tax lot.
        
        Args:
            shares_sold: Number of shares to sell
            
        Returns:
            Tuple of (sold_lot, remaining_lot). remaining_lot is None if all shares sold.
        """
        if shares_sold > self.shares:
            raise ValueError(f"Cannot sell {shares_sold} shares from lot with {self.shares} shares")
        
        # Create sold lot
        sold_lot = TaxLot(
            lot_id=f"{self.lot_id}_sold_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            purchase_date=self.purchase_date,
            shares=shares_sold,
            cost_basis_per_share=self.cost_basis_per_share
        )
        
        # Update remaining shares
        remaining_shares = self.shares - shares_sold
        if remaining_shares > 0:
            remaining_lot = TaxLot(
                lot_id=self.lot_id,
                purchase_date=self.purchase_date,
                shares=remaining_shares,
                cost_basis_per_share=self.cost_basis_per_share
            )
            return sold_lot, remaining_lot
        else:
            return sold_lot, None


class Position:
    """
    Represents a position in a single asset with comprehensive tracking.
    
    This class manages:
    - Share quantity and market value tracking
    - Cost basis calculation with FIFO/LIFO methods
    - Realized and unrealized P&L calculation
    - Tax lot management for tax optimization
    - Position sizing and risk metrics
    """
    
    def __init__(self, 
                 symbol: str,
                 position_type: PositionType = PositionType.LONG,
                 cost_basis_method: str = "FIFO"):
        """
        Initialize a new position.
        
        Args:
            symbol: Asset symbol (e.g., 'AAPL', 'GOOGL')
            position_type: Long or short position
            cost_basis_method: Method for cost basis calculation ('FIFO', 'LIFO', 'AVERAGE')
        """
        self.symbol = symbol
        self.position_type = position_type
        self.cost_basis_method = cost_basis_method
        
        # Position tracking
        self.tax_lots: List[TaxLot] = []
        self.total_shares = Decimal('0')
        self.total_cost_basis = Decimal('0')
        self.average_cost_basis = Decimal('0')
        
        # P&L tracking
        self.realized_pnl = Decimal('0')
        self.unrealized_pnl = Decimal('0')
        self.total_pnl = Decimal('0')
        
        # Current market data
        self.current_price = Decimal('0')
        self.market_value = Decimal('0')
        self.last_updated = datetime.now()
        
        # Metadata
        self.created_date = datetime.now()
        self.position_id = str(uuid.uuid4())
    
    def add_shares(self, 
                   shares: Decimal, 
                   price_per_share: Decimal, 
                   transaction_date: date = None,
                   transaction_costs: Decimal = Decimal('0')) -> str:
        """
        Add shares to the position (buy transaction).
        
        Args:
            shares: Number of shares to add
            price_per_share: Price per share
            transaction_date: Date of transaction (defaults to today)
            transaction_costs: Transaction costs (commissions, fees)
            
        Returns:
            Tax lot ID for the new purchase
        """
        if transaction_date is None:
            transaction_date = date.today()
        
        # Calculate total cost including transaction costs
        total_cost = shares * price_per_share + transaction_costs
        cost_per_share_with_costs = total_cost / shares
        
        # Create new tax lot
        new_lot = TaxLot(
            purchase_date=transaction_date,
            shares=shares,
            cost_basis_per_share=cost_per_share_with_costs
        )
        
        self.tax_lots.append(new_lot)
        
        # Update position totals
        self.total_shares += shares
        self.total_cost_basis += total_cost
        self.average_cost_basis = self.total_cost_basis / self.total_shares if self.total_shares > 0 else Decimal('0')
        
        return new_lot.lot_id
    
    def remove_shares(self, 
                      shares: Decimal, 
                      price_per_share: Decimal,
                      transaction_date: date = None,
                      transaction_costs: Decimal = Decimal('0')) -> Dict:
        """
        Remove shares from the position (sell transaction).
        
        Args:
            shares: Number of shares to sell
            price_per_share: Price per share
            transaction_date: Date of transaction
            transaction_costs: Transaction costs
            
        Returns:
            Dictionary with sale details including realized P&L
        """
        if shares > self.total_shares:
            raise ValueError(f"Cannot sell {shares} shares from position with {self.total_shares} shares")
        
        if transaction_date is None:
            transaction_date = date.today()
        
        # Calculate gross proceeds
        gross_proceeds = shares * price_per_share
        net_proceeds = gross_proceeds - transaction_costs
        
        # Process sale using specified cost basis method
        sold_lots = []
        remaining_shares = shares
        cost_basis_sold = Decimal('0')
        
        # Sort tax lots based on cost basis method
        if self.cost_basis_method == "FIFO":
            lots_to_process = sorted(self.tax_lots, key=lambda x: x.purchase_date)
        elif self.cost_basis_method == "LIFO":
            lots_to_process = sorted(self.tax_lots, key=lambda x: x.purchase_date, reverse=True)
        else:  # AVERAGE
            # For average cost, we'll use FIFO but with average cost basis
            lots_to_process = sorted(self.tax_lots, key=lambda x: x.purchase_date)
        
        # Process lots until all shares are sold
        lots_to_remove = []
        for i, lot in enumerate(lots_to_process):
            if remaining_shares <= 0:
                break
            
            if lot.shares <= remaining_shares:
                # Sell entire lot
                sold_lots.append(lot)
                cost_basis_sold += lot.total_cost_basis
                remaining_shares -= lot.shares
                lots_to_remove.append(i)
            else:
                # Partial sale
                sold_lot, remaining_lot = lot.partial_sale(remaining_shares)
                sold_lots.append(sold_lot)
                cost_basis_sold += sold_lot.total_cost_basis
                
                # Replace original lot with remaining lot
                lots_to_process[i] = remaining_lot
                remaining_shares = Decimal('0')
        
        # Remove fully sold lots
        for i in reversed(lots_to_remove):
            lots_to_process.pop(i)
        
        # Update tax lots list
        self.tax_lots = [lot for lot in lots_to_process if lot.shares > 0]
        
        # Calculate realized P&L
        realized_pnl = net_proceeds - cost_basis_sold
        self.realized_pnl += realized_pnl
        
        # Update position totals
        self.total_shares -= shares
        self.total_cost_basis -= cost_basis_sold
        self.average_cost_basis = self.total_cost_basis / self.total_shares if self.total_shares > 0 else Decimal('0')
        
        return {
            'shares_sold': shares,
            'price_per_share': price_per_share,
            'gross_proceeds': gross_proceeds,
            'net_proceeds': net_proceeds,
            'cost_basis_sold': cost_basis_sold,
            'realized_pnl': realized_pnl,
            'transaction_costs': transaction_costs,
            'sold_lots': sold_lots,
            'transaction_date': transaction_date
        }
    
    def update_market_price(self, new_price: Decimal, timestamp: datetime = None):
        """
        Update current market price and recalculate unrealized P&L.
        
        Args:
            new_price: New market price per share
            timestamp: Timestamp of price update
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.current_price = new_price
        self.market_value = self.total_shares * new_price
        self.unrealized_pnl = self.market_value - self.total_cost_basis
        self.total_pnl = self.realized_pnl + self.unrealized_pnl
        self.last_updated = timestamp
    
    def get_position_summary(self) -> Dict:
        """
        Get comprehensive position summary.
        
        Returns:
            Dictionary with complete position information
        """
        return {
            'symbol': self.symbol,
            'position_id': self.position_id,
            'position_type': self.position_type.value,
            'cost_basis_method': self.cost_basis_method,
            
            # Quantity and value
            'total_shares': float(self.total_shares),
            'current_price': float(self.current_price),
            'market_value': float(self.market_value),
            'total_cost_basis': float(self.total_cost_basis),
            'average_cost_basis': float(self.average_cost_basis),
            
            # P&L
            'realized_pnl': float(self.realized_pnl),
            'unrealized_pnl': float(self.unrealized_pnl),
            'total_pnl': float(self.total_pnl),
            'return_percentage': float(self.total_pnl / self.total_cost_basis * 100) if self.total_cost_basis > 0 else 0.0,
            
            # Tax lots
            'number_of_lots': len(self.tax_lots),
            'tax_lots': [
                {
                    'lot_id': lot.lot_id,
                    'purchase_date': lot.purchase_date.isoformat(),
                    'shares': float(lot.shares),
                    'cost_basis_per_share': float(lot.cost_basis_per_share),
                    'total_cost_basis': float(lot.total_cost_basis)
                }
                for lot in self.tax_lots
            ],
            
            # Timestamps
            'created_date': self.created_date.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }
    
    def get_tax_lot_details(self) -> List[Dict]:
        """
        Get detailed tax lot information for tax reporting.
        
        Returns:
            List of tax lot dictionaries with all relevant information
        """
        return [
            {
                'lot_id': lot.lot_id,
                'symbol': self.symbol,
                'purchase_date': lot.purchase_date,
                'shares': lot.shares,
                'cost_basis_per_share': lot.cost_basis_per_share,
                'total_cost_basis': lot.total_cost_basis,
                'current_price': self.current_price,
                'current_market_value': lot.shares * self.current_price,
                'unrealized_pnl': (lot.shares * self.current_price) - lot.total_cost_basis,
                'days_held': (date.today() - lot.purchase_date).days
            }
            for lot in self.tax_lots
        ]
    
    def calculate_wash_sale_risk(self, sale_date: date, days_window: int = 30) -> List[Dict]:
        """
        Calculate potential wash sale violations.
        
        A wash sale occurs when you sell a security at a loss and purchase
        the same or substantially identical security within 30 days.
        
        Args:
            sale_date: Date of the sale
            days_window: Days before and after to check (default 30)
            
        Returns:
            List of potential wash sale violations
        """
        wash_sales = []
        
        for lot in self.tax_lots:
            days_diff = abs((lot.purchase_date - sale_date).days)
            if days_diff <= days_window:
                wash_sales.append({
                    'lot_id': lot.lot_id,
                    'purchase_date': lot.purchase_date,
                    'sale_date': sale_date,
                    'days_difference': days_diff,
                    'shares': lot.shares,
                    'risk_level': 'HIGH' if days_diff <= 30 else 'MEDIUM'
                })
        
        return wash_sales
    
    def split_position(self, ratio: Decimal) -> None:
        """
        Handle stock splits by adjusting shares and cost basis.
        
        Args:
            ratio: Split ratio (e.g., 2.0 for 2-for-1 split)
        """
        # Update all tax lots
        for lot in self.tax_lots:
            lot.shares *= ratio
            lot.cost_basis_per_share /= ratio
            lot.total_cost_basis = lot.shares * lot.cost_basis_per_share
        
        # Update position totals
        self.total_shares *= ratio
        self.average_cost_basis /= ratio
        self.current_price /= ratio
        self.market_value = self.total_shares * self.current_price
    
    def close_position(self, price_per_share: Decimal, transaction_date: date = None) -> Dict:
        """
        Close the entire position.
        
        Args:
            price_per_share: Final sale price
            transaction_date: Date of closure
            
        Returns:
            Final position summary with all realized P&L
        """
        if self.total_shares == 0:
            return self.get_position_summary()
        
        sale_details = self.remove_shares(
            shares=self.total_shares,
            price_per_share=price_per_share,
            transaction_date=transaction_date
        )
        
        return {
            'position_summary': self.get_position_summary(),
            'final_sale_details': sale_details,
            'position_closed': True,
            'closure_date': transaction_date or date.today()
        }
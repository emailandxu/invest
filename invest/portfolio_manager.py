import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional

from ._paths import data_path
from .backtest import StrategyRegistry, Strategy

@dataclass
class PortfolioConfig:
    """Defines a complete investment approach."""
    name: str                        # Display Name (e.g., "Classic 60/40")
    strategy_name: str               # Algorithm (e.g., "RebalanceStrategy")
    assets: List[str]                # Asset Universe (e.g., ["VTI", "BND"])
    allocation: Dict[str, float]     # Target Weights (e.g., {"VTI": 0.6, "BND": 0.4})
    strategy_params: Dict[str, Any]  # Algo Params (e.g., {"period_days": 90})
    note: str = ""                   # User notes (optional)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PortfolioConfig':
        """Create from dictionary."""
        return cls(**data)

    def create_strategy(self) -> Strategy:
        """Factory method to instantiate the strategy."""
        # Start with base params
        params = self.strategy_params.copy()
        
        # Inject allocation if the strategy accepts it (e.g., DCA, Rebalance)
        # Note: Some strategies might not accept 'allocation' in __init__, 
        # so ideally StrategyRegistry handles specific logic or we trust kwargs.
        # Most of our strategies that need allocation accept it.
        # We'll optimistically pass it if not present.
        if self.allocation and 'allocation' not in params:
             params['allocation'] = self.allocation
             
        return StrategyRegistry.create(self.strategy_name, **params)


class PortfolioManager:
    """Singleton-like manager for persisting portfolios."""
    
    FILE_NAME = "portfolios.json"
    
    @classmethod
    def get_file_path(cls):
        return data_path(cls.FILE_NAME)

    @classmethod
    def load_all(cls) -> Dict[str, PortfolioConfig]:
        """Returns dict of {name: PortfolioConfig} from JSON."""
        path = cls.get_file_path()
        if not path.exists():
            return {}
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            portfolios = {}
            for name, p_data in data.items():
                try:
                    portfolios[name] = PortfolioConfig.from_dict(p_data)
                except Exception as e:
                    print(f"Error loading portfolio '{name}': {e}")
            return portfolios
        except Exception as e:
            print(f"Error reading portfolios file: {e}")
            return {}

    @classmethod
    def save(cls, config: PortfolioConfig):
        """Save or overwrite a portfolio config."""
        portfolios = cls.load_all()
        portfolios[config.name] = config
        cls._write_file(portfolios)

    @classmethod
    def delete(cls, name: str):
        """Delete a portfolio by name."""
        portfolios = cls.load_all()
        if name in portfolios:
            del portfolios[name]
            cls._write_file(portfolios)

    @classmethod
    def _write_file(cls, portfolios: Dict[str, PortfolioConfig]):
        """Write all portfolios to JSON."""
        path = cls.get_file_path()
        data = {name: p.to_dict() for name, p in portfolios.items()}
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error writing portfolios file: {e}")

"""Investment simulator package with GUI and data utilities."""

from .invest_gui import InvestmentSimulatorGui, main
from .invest_simulator import InvestmentParams, StrategyBasic, InvestmentYearsResult

__all__ = [
    "InvestmentSimulatorGui",
    "InvestmentParams",
    "InvestmentYearsResult",
    "StrategyBasic",
    "main",
]

"""Investment simulator package with GUI and data utilities."""

from .invest_gui import InvestmentSimulatorGui, main
from .invest_simulator import InvestmentParams, StrategyBasic, InvestmentYearsResult
from .analysis import analysis
from .download_data import download_data

__all__ = [
    "InvestmentSimulatorGui",
    "InvestmentParams",
    "InvestmentYearsResult",
    "StrategyBasic",
    "main",
    "analysis",
    "download_data",
]

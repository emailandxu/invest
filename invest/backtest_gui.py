"""
Backtest GUI - Standalone window for running backtests.

This module provides a Qt-based GUI (PySide6) for:
- Selecting and configuring trading strategies
- Running backtests with customizable parameters  
- Visualizing equity curves and performance metrics
"""

import sys
from datetime import date, datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt, QDate, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QComboBox, QPushButton, QTextEdit,
    QSpinBox, QDoubleSpinBox, QDateEdit, QListWidget, QListWidgetItem,
    QAbstractItemView, QSplitter, QGroupBox, QCheckBox, QMessageBox
)

from .backtest import Backtester, BacktestResult, StrategyRegistry
from ._paths import data_path

# Import strategies to ensure they are registered
from . import strategies


class BacktestControlPanel(QWidget):
    """Control panel for backtest configuration."""
    
    def __init__(self, on_run_backtest=None):
        super().__init__()
        self.on_run_backtest = on_run_backtest
        self._setup_ui()
        self._load_assets()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # --- PORTFOLIO MANAGEMENT ---
        portfolio_group = QGroupBox("Saved Portfolios")
        portfolio_layout = QHBoxLayout()
        
        self.portfolio_combo = QComboBox()
        self.portfolio_combo.addItem("- Select -")
        self.portfolio_combo.currentTextChanged.connect(self._on_portfolio_changed)
        
        self.save_portfolio_btn = QPushButton("Save")
        self.save_portfolio_btn.clicked.connect(self._on_save_portfolio)
        
        self.del_portfolio_btn = QPushButton("Delete")
        self.del_portfolio_btn.clicked.connect(self._on_delete_portfolio)
        
        portfolio_layout.addWidget(self.portfolio_combo, 7)
        portfolio_layout.addWidget(self.save_portfolio_btn, 2)
        portfolio_layout.addWidget(self.del_portfolio_btn, 2)
        portfolio_group.setLayout(portfolio_layout)
        layout.addWidget(portfolio_group)

        # Strategy selection
        strategy_group = QGroupBox("Strategy")
        strategy_layout = QVBoxLayout(strategy_group)
        
        self.strategy_combo = QComboBox()
        self._populate_strategies()
        strategy_layout.addWidget(self.strategy_combo)
        
        # Strategy description
        self.strategy_desc = QLabel("")
        self.strategy_desc.setWordWrap(True)
        self.strategy_desc.setStyleSheet("color: gray; font-size: 10px;")
        strategy_layout.addWidget(self.strategy_desc)
        self.strategy_combo.currentTextChanged.connect(self._update_strategy_desc)
        self.strategy_combo.currentTextChanged.connect(self._update_parameter_panel)
        self._update_strategy_desc()
        
        layout.addWidget(strategy_group)
        
        # Strategy Parameters (dynamic)
        self.params_group = QGroupBox("Strategy Parameters")
        self.params_layout = QGridLayout(self.params_group)
        self.param_widgets = {}  # Store parameter widgets for value retrieval
        layout.addWidget(self.params_group)
        self._update_parameter_panel()
        
        # Asset selection
        asset_group = QGroupBox("Assets")
        asset_layout = QVBoxLayout(asset_group)
        
        self.asset_list = QListWidget()
        self.asset_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.asset_list.setMaximumHeight(120)
        self.asset_list.itemSelectionChanged.connect(self._update_allocation_sliders)
        asset_layout.addWidget(self.asset_list)
        
        layout.addWidget(asset_group)
        
        # Asset Allocation (weights)
        self.allocation_group = QGroupBox("Asset Allocation (%)")
        self.allocation_layout = QGridLayout(self.allocation_group)
        self.allocation_sliders = {}  # symbol -> slider
        layout.addWidget(self.allocation_group)
        
        # Date range
        date_group = QGroupBox("Date Range")
        date_layout = QGridLayout(date_group)
        
        date_layout.addWidget(QLabel("Start:"), 0, 0)
        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(QDate(2010, 1, 1))
        self.start_date.setDisplayFormat("yyyy-MM-dd")
        date_layout.addWidget(self.start_date, 0, 1)
        
        date_layout.addWidget(QLabel("End:"), 1, 0)
        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setDisplayFormat("yyyy-MM-dd")
        date_layout.addWidget(self.end_date, 1, 1)
        
        layout.addWidget(date_group)
        
        # Capital
        capital_group = QGroupBox("Initial Capital")
        capital_layout = QHBoxLayout(capital_group)
        
        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(1000, 100000000)
        self.capital_spin.setValue(100000)
        self.capital_spin.setPrefix("$ ")
        self.capital_spin.setSingleStep(10000)
        capital_layout.addWidget(self.capital_spin)
        
        layout.addWidget(capital_group)
        
        # Periodic Withdrawal (Consumption)
        withdrawal_group = QGroupBox("Periodic Withdrawal")
        withdrawal_layout = QGridLayout(withdrawal_group)
        
        withdrawal_layout.addWidget(QLabel("Amount:"), 0, 0)
        self.withdrawal_amount = QDoubleSpinBox()
        self.withdrawal_amount.setRange(0, 1000000)
        self.withdrawal_amount.setValue(0)
        self.withdrawal_amount.setPrefix("$ ")
        self.withdrawal_amount.setSingleStep(100)
        withdrawal_layout.addWidget(self.withdrawal_amount, 0, 1)
        
        withdrawal_layout.addWidget(QLabel("Period (Days):"), 1, 0)
        self.withdrawal_period = QSpinBox()
        self.withdrawal_period.setRange(1, 365)
        self.withdrawal_period.setValue(30)
        withdrawal_layout.addWidget(self.withdrawal_period, 1, 1)
        
        withdrawal_layout.addWidget(QLabel("Method:"), 2, 0)
        self.withdrawal_method = QComboBox()
        self.withdrawal_method.addItems([
            "Proportional",
            "Rebalance",
            "Sell Winners",
            "Sell Losers"
        ])
        withdrawal_layout.addWidget(self.withdrawal_method, 2, 1)
        
        self.adjust_inflation = QCheckBox("Adjust for US Inflation")
        self.adjust_inflation.setChecked(True)
        self.adjust_inflation.setToolTip("Withdrawal amount increases with historical US inflation")
        withdrawal_layout.addWidget(self.adjust_inflation, 3, 0, 1, 2)
        
        layout.addWidget(withdrawal_group)
        
        # Benchmark toggle
        self.show_benchmark = QCheckBox("Show Benchmark (Buy & Hold)")
        self.show_benchmark.setChecked(True)
        layout.addWidget(self.show_benchmark)
        
        self.show_vti_benchmark = QCheckBox("Show VTI Price Benchmark")
        self.show_vti_benchmark.setChecked(False)
        layout.addWidget(self.show_vti_benchmark)
        
        self.show_vti_withdrawal = QCheckBox("Show VTI + Withdrawal Benchmark")
        self.show_vti_withdrawal.setChecked(False)
        layout.addWidget(self.show_vti_withdrawal)
        
        # Run button
        self.run_btn = QPushButton("▶ Run Backtest")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.run_btn.clicked.connect(self._on_run_clicked)
        layout.addWidget(self.run_btn)
        
        # Show report button
        self.report_btn = QPushButton("📊 Show Report")
        self.report_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        layout.addWidget(self.report_btn)
        
        layout.addStretch()
        
        layout.addStretch()
        
        # Load saved portfolios
        self._load_portfolios_to_combo()
    
    def _populate_strategies(self):
        """Populate strategy dropdown from registry."""
        self.strategy_combo.clear()
        for name in StrategyRegistry.list_strategies():
            self.strategy_combo.addItem(name)
    
    def _update_strategy_desc(self):
        """Update strategy description label."""
        name = self.strategy_combo.currentText()
        strategy_class = StrategyRegistry.get(name)
        if strategy_class:
            self.strategy_desc.setText(strategy_class.description)
    
    def _update_parameter_panel(self):
        """Dynamically create parameter controls based on selected strategy."""
        # Clear existing widgets
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.param_widgets.clear()
        
        name = self.strategy_combo.currentText()
        strategy_class = StrategyRegistry.get(name)
        
        if not strategy_class or not hasattr(strategy_class, 'parameters'):
            self.params_group.setVisible(False)
            return
        
        params = strategy_class.parameters
        if not params:
            self.params_group.setVisible(False)
            return
        
        self.params_group.setVisible(True)
        
        for row, param in enumerate(params):
            label = QLabel(param.label + ":")
            self.params_layout.addWidget(label, row, 0)
            
            if param.param_type == 'int':
                widget = QSpinBox()
                widget.setRange(int(param.min_value or 0), int(param.max_value or 10000))
                widget.setSingleStep(int(param.step))
                widget.setValue(int(param.default))
            elif param.param_type == 'float':
                widget = QDoubleSpinBox()
                widget.setRange(param.min_value or 0, param.max_value or 1000000)
                widget.setSingleStep(param.step)
                widget.setValue(param.default)
                widget.setDecimals(2)
            elif param.param_type == 'bool':
                widget = QCheckBox()
                widget.setChecked(bool(param.default))
            else:
                widget = QLabel(str(param.default))
            
            self.params_layout.addWidget(widget, row, 1)
            self.param_widgets[param.name] = widget
    
    def get_strategy_params(self) -> dict:
        """Get current parameter values from UI controls."""
        params = {}
        name = self.strategy_combo.currentText()
        strategy_class = StrategyRegistry.get(name)
        
        if not strategy_class or not hasattr(strategy_class, 'parameters'):
            return params
        
        for param in strategy_class.parameters:
            widget = self.param_widgets.get(param.name)
            if widget is None:
                continue
            
            if param.param_type == 'int':
                params[param.name] = widget.value()
            elif param.param_type == 'float':
                params[param.name] = widget.value()
            elif param.param_type == 'bool':
                params[param.name] = widget.isChecked()
        
        return params
    
    def _load_assets(self):
        """Load available assets from data directory."""
        self.asset_list.clear()
        stock_dir = data_path("STOCK")
        
        if stock_dir.exists():
            for csv_file in sorted(stock_dir.glob("*.csv")):
                if csv_file.name.startswith("."):
                    continue
                if csv_file.name.endswith("_dividends.csv"):
                    continue
                item = QListWidgetItem(csv_file.stem)
                self.asset_list.addItem(item)
        
        # Select first item by default
        if self.asset_list.count() > 0:
            self.asset_list.item(0).setSelected(True)
        
        self._update_allocation_sliders()
    
    def _update_allocation_sliders(self):
        """Create/update allocation sliders for selected assets."""
        # Clear existing sliders
        while self.allocation_layout.count():
            item = self.allocation_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.allocation_sliders.clear()
        
        selected_assets = self.get_selected_assets()
        if not selected_assets:
            self.allocation_group.setVisible(False)
            return
        
        self.allocation_group.setVisible(True)
        equal_weight = 100 // len(selected_assets)
        
        for row, symbol in enumerate(selected_assets):
            label = QLabel(f"{symbol}:")
            self.allocation_layout.addWidget(label, row, 0)
            
            spinbox = QSpinBox()
            spinbox.setRange(0, 100)
            spinbox.setSuffix("%")
            spinbox.setValue(equal_weight)
            self.allocation_layout.addWidget(spinbox, row, 1)
            self.allocation_sliders[symbol] = spinbox
    
    def _on_allocation_change(self):
        # Allocation change hook; currently used when loading saved portfolios.
        return

    def get_allocation(self) -> dict:
        """Get current allocation weights as a dict (symbol -> weight 0-1)."""
        allocation = {}
        total = sum(s.value() for s in self.allocation_sliders.values())
        
        if total == 0:
            # Equal weight if all zeros
            n = len(self.allocation_sliders)
            for symbol in self.allocation_sliders:
                allocation[symbol] = 1.0 / n if n > 0 else 0
        else:
            # Normalize to sum to 1.0
            for symbol, spinbox in self.allocation_sliders.items():
                allocation[symbol] = spinbox.value() / total
        
        return allocation
    
    def _on_run_clicked(self):
        if self.on_run_backtest:
            self.on_run_backtest()
    
    def get_selected_strategy_name(self) -> str:
        return self.strategy_combo.currentText()
    
    def get_selected_assets(self) -> List[str]:
        return [item.text() for item in self.asset_list.selectedItems()]
    
    def get_start_date(self) -> date:
        qdate = self.start_date.date()
        return date(qdate.year(), qdate.month(), qdate.day())
    
    def get_end_date(self) -> date:
        qdate = self.end_date.date()
        return date(qdate.year(), qdate.month(), qdate.day())
    
    def get_initial_capital(self) -> float:
        return self.capital_spin.value()
    
    def should_show_benchmark(self) -> bool:
        return self.show_benchmark.isChecked()
    
    def should_show_vti_benchmark(self) -> bool:
        return self.show_vti_benchmark.isChecked()
    
    def get_withdrawal_amount(self) -> float:
        return self.withdrawal_amount.value()
    
    def get_withdrawal_period(self) -> int:
        return self.withdrawal_period.value()
    
    def get_withdrawal_method(self) -> str:
        # Convert display name to internal name
        method_map = {
            "Proportional": "proportional",
            "Rebalance": "rebalance",
            "Sell Winners": "sell_winners",
            "Sell Losers": "sell_losers"
        }
        return method_map.get(self.withdrawal_method.currentText(), "proportional")
    
    def should_show_vti_withdrawal(self) -> bool:
        return self.show_vti_withdrawal.isChecked()
    
    def should_adjust_for_inflation(self) -> bool:
        return self.adjust_inflation.isChecked()
    
    def get_params(self) -> dict:
        """Get all current parameters as a dictionary for saving."""
        return {
            'strategy': self.strategy_combo.currentText(),
            'strategy_params': self.get_strategy_params(),
            'selected_assets': self.get_selected_assets(),
            'allocation': self.get_allocation(),
            'start_date': self.get_start_date().isoformat(),
            'end_date': self.get_end_date().isoformat(),
            'initial_capital': self.capital_spin.value(),
            'show_benchmark': self.show_benchmark.isChecked(),
            'show_vti_benchmark': self.show_vti_benchmark.isChecked(),
            'withdrawal_amount': self.withdrawal_amount.value(),
            'withdrawal_period': self.withdrawal_period.value(),
            'withdrawal_method': self.withdrawal_method.currentText(),
            'show_vti_withdrawal': self.show_vti_withdrawal.isChecked(),
            'adjust_inflation': self.adjust_inflation.isChecked()
        }
    
    def set_params(self, params: dict) -> None:
        """Restore parameters from a dictionary."""
        if not params:
            return
        
        # Strategy
        if 'strategy' in params:
            idx = self.strategy_combo.findText(params['strategy'])
            if idx >= 0:
                # This will trigger parameter panel update via signal
                self.strategy_combo.setCurrentIndex(idx)
                
                # Restore strategy parameters if available
                if 'strategy_params' in params:
                    s_params = params['strategy_params']
                    for name, value in s_params.items():
                        if name in self.param_widgets:
                            widget = self.param_widgets[name]
                            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                                widget.setValue(value)
                            elif isinstance(widget, QCheckBox):
                                widget.setChecked(bool(value))
        
        # Assets selection
        if 'selected_assets' in params:
            self.asset_list.blockSignals(True)
            for i in range(self.asset_list.count()):
                item = self.asset_list.item(i)
                item.setSelected(item.text() in params['selected_assets'])
            self.asset_list.blockSignals(False)
            # Manually trigger update once after bulk selection
            self._update_allocation_sliders()
        
        # Allocation
        if 'allocation' in params:
            for symbol, pct in params['allocation'].items():
                if symbol in self.allocation_sliders:
                    # Convert float (0.5) to percentage int (50)
                    self.allocation_sliders[symbol].setValue(int(pct * 100))
        
        # Dates
        if 'start_date' in params:
            d = date.fromisoformat(params['start_date'])
            self.start_date.setDate(QDate(d.year, d.month, d.day))
        if 'end_date' in params:
            d = date.fromisoformat(params['end_date'])
            self.end_date.setDate(QDate(d.year, d.month, d.day))
        
        # Capital
        if 'initial_capital' in params:
            self.capital_spin.setValue(params['initial_capital'])
        
        # Checkboxes
        if 'show_benchmark' in params:
            self.show_benchmark.setChecked(params['show_benchmark'])
        if 'show_vti_benchmark' in params:
            self.show_vti_benchmark.setChecked(params['show_vti_benchmark'])
        if 'show_vti_withdrawal' in params:
            self.show_vti_withdrawal.setChecked(params['show_vti_withdrawal'])
        if 'adjust_inflation' in params:
            self.adjust_inflation.setChecked(params['adjust_inflation'])
        
        # Withdrawal
        if 'withdrawal_amount' in params:
            self.withdrawal_amount.setValue(params['withdrawal_amount'])
        if 'withdrawal_period' in params:
            self.withdrawal_period.setValue(params['withdrawal_period'])
        if 'withdrawal_method' in params:
            idx = self.withdrawal_method.findText(params['withdrawal_method'])
            if idx >= 0:
                self.withdrawal_method.setCurrentIndex(idx)


    
    # --- PORTFOLIO MANAGEMENT METHODS ---

    def _load_portfolios_to_combo(self):
        """Reload portfolios from disk into ComboBox."""
        from .portfolio_manager import PortfolioManager
        self.portfolio_combo.blockSignals(True)
        self.portfolio_combo.clear()
        self.portfolio_combo.addItem("- Select -")
        
        portfolios = PortfolioManager.load_all()
        for name in sorted(portfolios.keys()):
            self.portfolio_combo.addItem(name)
            
        self.portfolio_combo.blockSignals(False)

    def _on_portfolio_changed(self, name):
        """Load selected portfolio configuration."""
        if name == "- Select -" or not name:
            return
            
        from .portfolio_manager import PortfolioManager
        portfolios = PortfolioManager.load_all()
        config = portfolios.get(name)
        if not config:
            return
            
        # Apply config to UI
        self.blockSignals(True)
        try:
            # 1. Strategy
            idx = self.strategy_combo.findText(config.strategy_name)
            if idx >= 0:
                self.strategy_combo.setCurrentIndex(idx)
            
            # 2. Assets
            # Critical: Update QListWidget selection because get_selected_assets() uses it
            self.asset_list.blockSignals(True)
            self.asset_list.clearSelection()
            for i in range(self.asset_list.count()):
                item = self.asset_list.item(i)
                if item.text() in config.assets:
                    item.setSelected(True)
            self.asset_list.blockSignals(False)
            
            self._update_allocation_sliders() # Rebuild sliders based on new selection
            
            # 3. Allocation
            if config.allocation:
                for symbol, weight in config.allocation.items():
                    if symbol in self.allocation_sliders:
                        self.allocation_sliders[symbol].setValue(int(weight * 100))
                self._on_allocation_change()
                
            # 4. Strategy Params
            # Note: Strategy change triggers param panel rebuild.
            # We need to explicitly set values now.
            self._update_parameter_panel() # Ensure fields exist
            
            strategy_class = StrategyRegistry.get(config.strategy_name)
            if strategy_class and strategy_class.parameters:
                for row, param in enumerate(strategy_class.parameters):
                    item = self.params_layout.itemAtPosition(row, 1)
                    if item and item.widget():
                        widget = item.widget()
                        val = config.strategy_params.get(param.name)
                        if val is not None:
                            if isinstance(widget, QSpinBox):
                                widget.setValue(int(val))
                            elif isinstance(widget, QDoubleSpinBox):
                                widget.setValue(float(val))
                            elif isinstance(widget, QCheckBox):
                                 widget.setChecked(bool(val))
                                 widget.setChecked(bool(val))
        except Exception as e:
            print(f"Error applying portfolio config: {e}")
        finally:
             self.blockSignals(False)
        
        # Auto-run backtest if callback provided
        # Use QTimer to allow UI state to settle and signal blocks to clear
        if self.on_run_backtest:
            QTimer.singleShot(50, self.on_run_backtest)

    def _on_save_portfolio(self):
        """Save current configuration as a new portfolio."""
        from PySide6.QtWidgets import QInputDialog, QMessageBox
        from .portfolio_manager import PortfolioManager, PortfolioConfig
        
        # Pre-fill with current selection if valid
        current_name = self.portfolio_combo.currentText()
        default_name = current_name if current_name != "- Select -" else ""
        
        name, ok = QInputDialog.getText(self, "Save Portfolio", "Portfolio Name:", text=default_name)
        if not ok or not name.strip():
            return
            
        name = name.strip()
        
        # Check for overwrite
        existing_portfolios = PortfolioManager.load_all()
        if name in existing_portfolios:
            reply = QMessageBox.question(
                self, "Confirm Overwrite", 
                f"Portfolio '{name}' already exists.\nDo you want to overwrite it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        
        # Gather current state
        params = self.get_params()
        
        config = PortfolioConfig(
            name=name,
            strategy_name=params['strategy'],
            assets=params['selected_assets'],
            allocation=params['allocation'],
            strategy_params=params['strategy_params'],
            note=""
        )
        
        try:
            PortfolioManager.save(config)
            QMessageBox.information(self, "Success", f"Portfolio '{name}' saved.")
            
            # Reload combo and select
            self._load_portfolios_to_combo()
            self.portfolio_combo.setCurrentText(name)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def _on_delete_portfolio(self):
        """Delete current portfolio."""
        from PySide6.QtWidgets import QMessageBox
        from .portfolio_manager import PortfolioManager
        
        name = self.portfolio_combo.currentText()
        if name == "- Select -":
            return
            
        reply = QMessageBox.question(self, "Confirm Delete", 
                                   f"Are you sure you want to delete '{name}'?",
                                   QMessageBox.Yes | QMessageBox.No)
                                   
        if reply == QMessageBox.Yes:
            try:
                PortfolioManager.delete(name)
                self._load_portfolios_to_combo()
                self.portfolio_combo.setCurrentIndex(0)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete: {e}")


class BacktestPlotPanel(QWidget):
    """Panel for displaying backtest charts."""
    
    def __init__(self):
        super().__init__()
        self._setup_ui()
        self.result: Optional[BacktestResult] = None
        self.benchmark_result: Optional[BacktestResult] = None
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Main equity curve plot
        self.equity_plot = pg.PlotWidget(title="Equity Curve", axisItems={'bottom': pg.DateAxisItem()})
        self.equity_plot.setLabel('left', 'Equity ($)')
        self.equity_plot.setLabel('bottom', 'Date')
        self.equity_plot.showGrid(x=True, y=True, alpha=0.3)
        self.equity_plot.addLegend()
        layout.addWidget(self.equity_plot, stretch=2)
        
        # Hover tooltip for equity chart
        self.equity_vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#888', style=Qt.DashLine))
        self.equity_plot.addItem(self.equity_vline, ignoreBounds=True)
        self.equity_vline.setVisible(False)
        self.equity_label = pg.TextItem(anchor=(0, 1), fill=pg.mkBrush('#333'))
        self.equity_plot.addItem(self.equity_label, ignoreBounds=True)
        self.equity_label.setVisible(False)
        self.equity_label.setZValue(1000)
        self.equity_vline.setZValue(1000)
        self.equity_plot.scene().sigMouseMoved.connect(self._on_equity_hover)
        
        # Drawdown plot
        self.drawdown_plot = pg.PlotWidget(title="Drawdown", axisItems={'bottom': pg.DateAxisItem()})
        self.drawdown_plot.setLabel('left', 'Drawdown (%)')
        self.drawdown_plot.setLabel('bottom', 'Date')
        self.drawdown_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.drawdown_plot, stretch=1)
        
        # Hover tooltip for drawdown chart
        self.drawdown_vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#888', style=Qt.DashLine))
        self.drawdown_plot.addItem(self.drawdown_vline, ignoreBounds=True)
        self.drawdown_vline.setVisible(False)
        self.drawdown_label = pg.TextItem(anchor=(0, 1), fill=pg.mkBrush('#333'))
        self.drawdown_plot.addItem(self.drawdown_label, ignoreBounds=True)
        self.drawdown_label.setVisible(False)
        self.drawdown_label.setZValue(1000)
        self.drawdown_vline.setZValue(1000)
        self.drawdown_plot.scene().sigMouseMoved.connect(self._on_drawdown_hover)
        
        # Allocation chart
        self.allocation_plot = pg.PlotWidget(title="Asset Allocation", axisItems={'bottom': pg.DateAxisItem()})
        self.allocation_plot.setLabel('left', 'Allocation (%)')
        self.allocation_plot.setLabel('bottom', 'Date')
        self.allocation_plot.showGrid(x=True, y=True, alpha=0.3)
        self.allocation_plot.addLegend()
        layout.addWidget(self.allocation_plot, stretch=1)
        
        # Hover tooltip for allocation chart
        self.allocation_vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#888', style=Qt.DashLine))
        self.allocation_plot.addItem(self.allocation_vline, ignoreBounds=True)
        self.allocation_vline.setVisible(False)
        
        self.allocation_label = pg.TextItem(anchor=(0, 1), fill=pg.mkBrush('#333'))
        self.allocation_plot.addItem(self.allocation_label, ignoreBounds=True)
        self.allocation_label.setVisible(False)
        self.allocation_label.setZValue(1000)
        self.allocation_vline.setZValue(1000)
        
        # Connect mouse move signal
        self.allocation_plot.scene().sigMouseMoved.connect(self._on_allocation_hover)
        
        # Link X-axes for synchronized zooming/panning
        self.drawdown_plot.setXLink(self.equity_plot)
        self.allocation_plot.setXLink(self.equity_plot)
        
        # Disable generic mouse wheel zooming on secondary plots
        # We only want the top plot (Equity) to control zoom via scroll
        # But we still want them to accept Drag events (Panning)
        self.drawdown_plot.wheelEvent = lambda event: None
        self.allocation_plot.wheelEvent = lambda event: None
    
    def update_plots(self, result: BacktestResult, benchmark: Optional[BacktestResult] = None,
                      vti_prices: Optional[list] = None,
                      vti_withdrawal: Optional[BacktestResult] = None):
        """Update plots with backtest results."""
        self.result = result
        self.benchmark_result = benchmark
        
        self.equity_plot.clear()
        self.drawdown_plot.clear()
        
        # Clear legends to prevent accumulation
        for plot in [self.equity_plot, self.drawdown_plot]:
            legend = plot.plotItem.legend
            if legend:
                legend.clear()
        
        # Re-add hover elements after clear
        self.equity_plot.addItem(self.equity_vline, ignoreBounds=True)
        self.equity_plot.addItem(self.equity_label, ignoreBounds=True)
        self.drawdown_plot.addItem(self.drawdown_vline, ignoreBounds=True)
        self.drawdown_plot.addItem(self.drawdown_label, ignoreBounds=True)
        
        if not result.equity_curve:
            return
        
        # Convert dates to numeric for plotting
        dates, equities = result.get_equity_series()
        # Use timestamps for continuous date axis
        import pandas as pd
        self.plot_timestamps = np.array([pd.to_datetime(d).timestamp() for d in dates])
        x = self.plot_timestamps
        
        # Plot strategy equity
        self.equity_plot.plot(
            x, equities,
            pen=pg.mkPen(color='#2196F3', width=2),
            name=result.strategy_name
        )
        
        # Plot benchmark if available
        if benchmark and benchmark.equity_curve:
            _, bench_equities = benchmark.get_equity_series()
            # Align lengths
            min_len = min(len(equities), len(bench_equities))
            self.equity_plot.plot(
                x[:min_len], bench_equities[:min_len],
                pen=pg.mkPen(color='#9E9E9E', width=1, style=Qt.DashLine),
                name="Benchmark (Buy & Hold)"
            )
        
        # Plot VTI + Withdrawal benchmark if available
        if vti_withdrawal and vti_withdrawal.equity_curve:
            _, vti_w_equities = vti_withdrawal.get_equity_series()
            min_len = min(len(x), len(vti_w_equities))
            self.equity_plot.plot(
                x[:min_len], vti_w_equities[:min_len],
                pen=pg.mkPen(color='#00BCD4', width=1, style=Qt.DashLine),
                name="VTI + Withdrawal"
            )
        
        # Plot VTI price benchmark if available
        if vti_prices and len(vti_prices) > 0:
            # Normalize VTI prices to start at initial capital
            initial_price = vti_prices[0]
            if initial_price > 0:
                scale = result.initial_capital / initial_price
                vti_scaled = [p * scale for p in vti_prices]
                min_len = min(len(x), len(vti_scaled))
                self.equity_plot.plot(
                    x[:min_len], vti_scaled[:min_len],
                    pen=pg.mkPen(color='#FF9800', width=1, style=Qt.DotLine),
                    name="VTI Price"
                )
        
        # Mark trades
        for trade in result.trades:
            # Find index of trade date
            try:
                idx = dates.index(trade.date)
                equity_at_trade = equities[idx]
                
                if trade.side.value == 'buy':
                    symbol = '▲'
                    color = '#4CAF50'
                else:
                    symbol = '▼'
                    color = '#F44336'
                
                scatter = pg.ScatterPlotItem(
                    [idx], [equity_at_trade],
                    symbol='t' if trade.side.value == 'buy' else 't1',
                    size=8,
                    brush=pg.mkBrush(color)
                )
                self.equity_plot.addItem(scatter)
            except (ValueError, IndexError):
                pass
        
        # Calculate and plot drawdown
        equities_arr = np.array(equities)
        peak = np.maximum.accumulate(equities_arr)
        drawdown = (equities_arr - peak) / peak * 100
        
        self.drawdown_plot.plot(
            x, drawdown,
            pen=pg.mkPen(color='#2196F3', width=2),
            fillLevel=0,
            brush=pg.mkBrush('#2196F340'),
            name=result.strategy_name
        )
        
        # Plot benchmark drawdown if available
        if benchmark and benchmark.equity_curve:
            _, bench_equities = benchmark.get_equity_series()
            min_len = min(len(equities), len(bench_equities))
            bench_arr = np.array(bench_equities[:min_len])
            bench_peak = np.maximum.accumulate(bench_arr)
            bench_drawdown = (bench_arr - bench_peak) / bench_peak * 100
            
            self.drawdown_plot.plot(
                x[:min_len], bench_drawdown,
                pen=pg.mkPen(color='#9E9E9E', width=1, style=Qt.DashLine),
                name="Benchmark (Buy & Hold)"
            )
        
        # Plot VTI drawdown if available
        if vti_prices and len(vti_prices) > 0:
            vti_arr = np.array(vti_prices[:len(x)])
            vti_peak = np.maximum.accumulate(vti_arr)
            vti_drawdown = (vti_arr - vti_peak) / vti_peak * 100
            min_len = min(len(x), len(vti_drawdown))
            self.drawdown_plot.plot(
                x[:min_len], vti_drawdown[:min_len],
                pen=pg.mkPen(color='#FF9800', width=1, style=Qt.DotLine),
                name="VTI"
            )
        
        # Add legend to drawdown plot
        self.drawdown_plot.addLegend()
        
        # Plot allocation chart
        self.allocation_plot.clear()
        # Remove old legend to prevent accumulation
        legend = self.allocation_plot.plotItem.legend
        if legend:
            legend.clear()
        
        if result.equity_curve and result.equity_curve[0].allocation:
            # Get all symbols from allocation data
            all_symbols = set()
            for ep in result.equity_curve:
                all_symbols.update(ep.allocation.keys())
            
            # Define colors for different assets
            colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336', 
                      '#00BCD4', '#E91E63', '#8BC34A', '#FFC107', '#607D8B']
            
            # Store color mapping for tooltip
            self.symbol_colors = {}
            for idx, symbol in enumerate(sorted(all_symbols)):
                alloc_pct = [ep.allocation.get(symbol, 0) * 100 for ep in result.equity_curve]
                color = colors[idx % len(colors)]
                self.symbol_colors[symbol] = color
                self.allocation_plot.plot(
                    x, alloc_pct,
                    pen=pg.mkPen(color=color, width=2),
                    name=symbol
                )
        
        # Set x-axis labels (showing some date labels)
        # With DateAxisItem, we don't need manual connection, but we can refine it if likely
        # DateAxisItem handles it automatically.
        pass
        
        # Re-add hover elements after clear
        self.allocation_plot.addItem(self.allocation_vline, ignoreBounds=True)
        self.allocation_plot.addItem(self.allocation_label, ignoreBounds=True)
    
    def _on_allocation_hover(self, pos):
        """Handle mouse hover over allocation chart."""
        if not self.result or not self.result.equity_curve or not hasattr(self, 'plot_timestamps'):
            return
        
        # Check if mouse is within plot area
        if not self.allocation_plot.sceneBoundingRect().contains(pos):
            self.allocation_vline.setVisible(False)
            self.allocation_label.setVisible(False)
            return
        
        # Map position to data coordinates
        mouse_point = self.allocation_plot.plotItem.vb.mapSceneToView(pos)
        x_ts = mouse_point.x()
        
        # Find nearest index
        idx = np.searchsorted(self.plot_timestamps, x_ts)
        if idx >= len(self.plot_timestamps):
            idx = len(self.plot_timestamps) - 1
        if idx < 0:
            idx = 0
            
        # Refine nearest
        if idx > 0:
            if abs(x_ts - self.plot_timestamps[idx-1]) < abs(x_ts - self.plot_timestamps[idx]):
                idx -= 1
        
        if idx < 0 or idx >= len(self.result.equity_curve):
            self.allocation_vline.setVisible(False)
            self.allocation_label.setVisible(False)
            return
            
        x_val = self.plot_timestamps[idx] # Snap to actual data point X
        
        # Get data for this date
        ep = self.result.equity_curve[idx]
        
        # Load dividend data if not already loaded or if symbols changed
        current_symbols = frozenset(self.result.symbols)
        if not hasattr(self, '_dividend_data') or getattr(self, '_dividend_symbols', None) != current_symbols:
            from .read_data import load_dividends
            self._dividend_data = {}
            self._price_data = None  # Reset price data too
            for symbol in self.result.symbols:
                self._dividend_data[symbol] = load_dividends(symbol)
            self._dividend_symbols = current_symbols
        
        # Build tooltip text with colored indicators
        html_lines = [
            f"<div style='font-family: monospace; padding: 4px;'>",
            f"<b>📅 {ep.date.strftime('%Y-%m-%d')}</b><br>",
            f"💰 Equity: ${ep.equity:,.0f}<br>"
        ]
        for symbol, pct in sorted(ep.allocation.items()):
            value = ep.equity * pct
            color = getattr(self, 'symbol_colors', {}).get(symbol, '#888')
            
            # Load price data if not already loaded
            if not hasattr(self, '_price_data') or self._price_data is None:
                from .read_data import stock_data_daily
                self._price_data = {}
                for s in self.result.symbols:
                    bars = stock_data_daily(s)
                    self._price_data[s] = {b.date: b.close for b in bars}
            
            # Calculate shares
            price = self._price_data.get(symbol, {}).get(ep.date, 0)
            if price > 0 and symbol != "CASH":
                shares = value / price
                line = f"<span style='color:{color}'>■</span> {symbol}: {pct*100:.1f}%, ${value:,.0f}, {shares:.0f}"
            else:
                line = f"<span style='color:{color}'>■</span> {symbol}: {pct*100:.1f}%, ${value:,.0f}"
            
            # Find the most recent dividend on or before this date
            div_data = self._dividend_data.get(symbol, {})
            recent_div = 0
            recent_div_date = None
            for div_date, div_amount in div_data.items():
                if div_date <= ep.date and (recent_div_date is None or div_date > recent_div_date):
                    recent_div = div_amount
                    recent_div_date = div_date
            
            if recent_div > 0 and price > 0:
                total_div = shares * recent_div
                line += f" <span style='color:#FFD700'>💵${recent_div:.2f}×{shares:.0f}=${total_div:.0f}</span>"
            
            html_lines.append(line + "<br>")
        html_lines.append("</div>")
        
        # Update tooltip
        self.allocation_vline.setPos(x_val)
        self.allocation_vline.setVisible(True)
        
        self.allocation_label.setHtml("".join(html_lines))
        self.allocation_label.setPos(x_val, mouse_point.y())
        self.allocation_label.setVisible(True)
    
    def _on_equity_hover(self, pos):
        """Handle mouse hover over equity chart."""
        if not self.result or not self.result.equity_curve or not hasattr(self, 'plot_timestamps'):
            return
        
        if not self.equity_plot.sceneBoundingRect().contains(pos):
            self.equity_vline.setVisible(False)
            self.equity_label.setVisible(False)
            return
        
        mouse_point = self.equity_plot.plotItem.vb.mapSceneToView(pos)
        x_ts = mouse_point.x()
        
        # Find nearest index
        idx = np.searchsorted(self.plot_timestamps, x_ts)
        if idx >= len(self.plot_timestamps):
            idx = len(self.plot_timestamps) - 1
        if idx < 0:
            idx = 0
            
        # Refine nearest
        if idx > 0:
            if abs(x_ts - self.plot_timestamps[idx-1]) < abs(x_ts - self.plot_timestamps[idx]):
                idx -= 1
        
        if idx < 0 or idx >= len(self.result.equity_curve):
            self.equity_vline.setVisible(False)
            self.equity_label.setVisible(False)
            return
            
        x_val = self.plot_timestamps[idx] # Snap to actual data point X
        
        ep = self.result.equity_curve[idx]
        total_return = (ep.equity / self.result.initial_capital - 1) * 100
        
        lines = [
            f"📅 {ep.date.strftime('%Y-%m-%d')}",
            f"💰 Equity: ${ep.equity:,.0f}",
            f"📈 Return: {total_return:+.1f}%",
            "",  # Separator
            "📊 Allocation:"
        ]
        
        # Sort allocation by percentage descending
        sorted_alloc = sorted(ep.allocation.items(), key=lambda x: x[1], reverse=True)
        
        for symbol, pct in sorted_alloc:
            if pct < 0.001:  # Skip < 0.1% to avoid clutter
                continue
            value = ep.equity * pct
            lines.append(f"• {symbol}: {pct*100:.1f}% (${value:,.0f})")
        
        self.equity_vline.setPos(x_val)
        self.equity_vline.setVisible(True)
        self.equity_label.setText("\n".join(lines))
        self.equity_label.setPos(x_val, mouse_point.y())
        self.equity_label.setVisible(True)
    
    def _on_drawdown_hover(self, pos):
        """Handle mouse hover over drawdown chart."""
        if not self.result or not self.result.equity_curve or not hasattr(self, 'plot_timestamps'):
            return
        
        if not self.drawdown_plot.sceneBoundingRect().contains(pos):
            self.drawdown_vline.setVisible(False)
            self.drawdown_label.setVisible(False)
            return
        
        mouse_point = self.drawdown_plot.plotItem.vb.mapSceneToView(pos)
        x_ts = mouse_point.x()
        
        # Find nearest index
        idx = np.searchsorted(self.plot_timestamps, x_ts)
        if idx >= len(self.plot_timestamps):
            idx = len(self.plot_timestamps) - 1
        if idx < 0:
            idx = 0
            
        # Refine nearest
        if idx > 0:
            if abs(x_ts - self.plot_timestamps[idx-1]) < abs(x_ts - self.plot_timestamps[idx]):
                idx -= 1
        
        if idx < 0 or idx >= len(self.result.equity_curve):
            self.drawdown_vline.setVisible(False)
            self.drawdown_label.setVisible(False)
            return
        
        x_val = self.plot_timestamps[idx] # Snap to actual data point X
        
        # Calculate drawdown at this point
        equities = [ep.equity for ep in self.result.equity_curve[:idx+1]]
        peak = max(equities) if equities else self.result.initial_capital
        ep = self.result.equity_curve[idx]
        drawdown = (ep.equity - peak) / peak * 100 if peak > 0 else 0
        
        lines = [
            f"📅 {ep.date.strftime('%Y-%m-%d')}",
            f"📉 Drawdown: {drawdown:.1f}%",
            f"💰 Equity: ${ep.equity:,.0f}",
            f"🔝 Peak: ${peak:,.0f}"
        ]
        
        self.drawdown_vline.setPos(x_val)
        self.drawdown_vline.setVisible(True)
        self.drawdown_label.setText("\n".join(lines))
        self.drawdown_label.setPos(x_val, mouse_point.y())
        self.drawdown_label.setVisible(True)


class BacktestReportWindow(QWidget):
    """Popup window for displaying backtest report."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Backtest Report")
        self.resize(500, 700)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                background-color: #1e1e1e;
                color: #d4d4d4;
                padding: 10px;
            }
        """)
        layout.addWidget(self.results_text)
    
    def set_report(self, result: BacktestResult, benchmark: BacktestResult = None):
        """Set the report text from backtest result."""
        report = result.summary(benchmark)
        
        # Add transaction log at the bottom
        if result.transaction_log:
            report += "\n\n" + "=" * 60
            report += "\n📋 TRANSACTION LOG"
            report += "\n" + "=" * 60 + "\n"
            # Show last 100 transactions (most recent at bottom)
            recent_log = result.transaction_log
            report += "\n".join(recent_log)
        
        self.results_text.setPlainText(report)
    
    def show_report(self):
        """Show the report window."""
        self.show()
        self.raise_()
        self.activateWindow()


class BacktestWindow(QMainWindow):
    """Main backtest window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Backtest - Investment Strategy Tester")
        self.setGeometry(100, 100, 1200, 800)
        
        self._setup_ui()
        
        # Auto-run backtest on startup (delayed slightly to ensure UI is ready)
        QTimer.singleShot(100, self.run_backtest)

    
    def _setup_ui(self):
        from PySide6.QtWidgets import QTabWidget
        from .compare_gui import ComparisonWidget
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # --- TAB 1: Strategy Lab (Existing UI) ---
        self.lab_tab = QWidget()
        lab_layout = QHBoxLayout(self.lab_tab)
        
        # Left: Control panel
        self.control_panel = BacktestControlPanel(on_run_backtest=self.run_backtest)
        self.control_panel.setFixedWidth(280)
        lab_layout.addWidget(self.control_panel)
        
        # Right: Plot panel
        self.plot_panel = BacktestPlotPanel()
        lab_layout.addWidget(self.plot_panel)
        
        self.tabs.addTab(self.lab_tab, "Strategy Lab")
        
        # --- TAB 2: Comparison (New UI) ---
        self.comparison_tab = ComparisonWidget()
        self.tabs.addTab(self.comparison_tab, "Comparison")
        
        # Report popup window
        self.report_window = BacktestReportWindow()
        
        # Connect Show Report button
        self.control_panel.report_btn.clicked.connect(self._show_report)
        
        # Determine tab change to refresh comparison list
        self.tabs.currentChanged.connect(self._on_tab_changed)
    
    def _on_tab_changed(self, index):
        """Handle tab switch."""
        # Index 1 is Comparison Tab
        if index == 1:
            self.comparison_tab.reload_portfolios()
            
    def _show_report(self):
        """Show the report popup window."""
        self.report_window.show_report()
    
    def run_backtest(self):
        """Execute backtest with current settings."""
        strategy_name = self.control_panel.get_selected_strategy_name()
        assets = self.control_panel.get_selected_assets()
        start_date = self.control_panel.get_start_date()
        end_date = self.control_panel.get_end_date()
        capital = self.control_panel.get_initial_capital()
        strategy_params = self.control_panel.get_strategy_params()
        allocation = self.control_panel.get_allocation()
        
        if not strategy_name or not assets:
            self.report_window.results_text.setPlainText(
                "Error: Please select a strategy and at least one asset."
            )
            return
        
        # Add allocation to strategy params if strategy accepts it
        strategy_params['allocation'] = allocation
        
        # Create strategy instance with parameters from UI
        try:
            strategy = StrategyRegistry.create(strategy_name, **strategy_params)
        except TypeError:
            # Strategy doesn't accept allocation parameter, try without it
            del strategy_params['allocation']
            strategy = StrategyRegistry.create(strategy_name, **strategy_params)
        
        if not strategy:
            self.report_window.results_text.setPlainText(
                f"Error: Could not create strategy '{strategy_name}'"
            )
            return
        
        # Run backtest
        try:
            withdrawal_amount = self.control_panel.get_withdrawal_amount()
            withdrawal_period = self.control_panel.get_withdrawal_period()
            withdrawal_method = self.control_panel.get_withdrawal_method()
            adjust_inflation = self.control_panel.should_adjust_for_inflation()
            
            backtester = Backtester(
                strategy=strategy,
                symbols=assets,
                start_date=start_date,
                end_date=end_date,
                initial_capital=capital,
                withdrawal_amount=withdrawal_amount,
                withdrawal_period_days=withdrawal_period,
                withdrawal_method=withdrawal_method,
                target_allocation=allocation,
                adjust_for_inflation=adjust_inflation
            )
            result = backtester.run()
            
            # Run benchmark if requested
            benchmark = None
            if self.control_panel.should_show_benchmark() and strategy_name != "Buy and Hold":
                bench_strategy = StrategyRegistry.create("Buy and Hold", allocation=allocation)
                if bench_strategy:
                    bench_backtester = Backtester(
                        strategy=bench_strategy,
                        symbols=assets,
                        start_date=start_date,
                        end_date=end_date,
                        initial_capital=capital,
                        withdrawal_amount=withdrawal_amount,
                        withdrawal_period_days=withdrawal_period,
                        withdrawal_method=withdrawal_method,
                        target_allocation=allocation,
                        adjust_for_inflation=adjust_inflation
                    )
                    benchmark = bench_backtester.run()
            
            # Load VTI price benchmark if requested
            # Load VTI price benchmark if requested
            vti_prices_list = None
            if self.control_panel.should_show_vti_benchmark():
                from .read_data import stock_data_daily
                vti_bars = stock_data_daily("VTI")
                
                # Filter to date range for plotting
                dates_set = set(ep.date for ep in result.equity_curve)
                vti_prices_list = [bar.close for bar in vti_bars if bar.date in dates_set]
                
                # Calculate Alpha/Beta using full overlapping data
                vti_dates = [bar.date for bar in vti_bars]
                vti_closes = [bar.close for bar in vti_bars]
                result.calculate_benchmark_metrics(vti_closes, vti_dates)
            
            # Run VTI + Withdrawal benchmark if requested
            vti_withdrawal_result = None
            if self.control_panel.should_show_vti_withdrawal():
                vti_strategy = StrategyRegistry.create("Buy and Hold")
                if vti_strategy:
                    vti_backtester = Backtester(
                        strategy=vti_strategy,
                        symbols=["VTI"],
                        start_date=start_date,
                        end_date=end_date,
                        initial_capital=capital,
                        withdrawal_amount=withdrawal_amount,
                        withdrawal_period_days=withdrawal_period,
                        withdrawal_method=withdrawal_method,
                        adjust_for_inflation=adjust_inflation
                    )
                    vti_withdrawal_result = vti_backtester.run()
            
            # Update UI
            self.plot_panel.update_plots(result, benchmark, vti_prices_list, vti_withdrawal_result)
            self.report_window.set_report(result, benchmark)
            
        except Exception as e:
            self.report_window.results_text.setPlainText(
                f"Error running backtest:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
    
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Escape, Qt.Key_Q):
            self.close()
        super().keyPressEvent(event)


def main():
    """Launch the Backtest GUI."""
    app = QApplication(sys.argv)
    
    # Set dark theme
    app.setStyle('Fusion')
    
    window = BacktestWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

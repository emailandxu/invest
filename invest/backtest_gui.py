"""
Backtest GUI - Standalone window for running backtests.

This module provides a PyQt5-based GUI for:
- Selecting and configuring trading strategies
- Running backtests with customizable parameters  
- Visualizing equity curves and performance metrics
"""

import sys
from datetime import date, datetime
from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QComboBox, QPushButton, QTextEdit,
    QSpinBox, QDoubleSpinBox, QDateEdit, QListWidget, QListWidgetItem,
    QAbstractItemView, QSplitter, QGroupBox, QCheckBox
)
from PyQt5.QtCore import QDate

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
        self.adjust_inflation.setChecked(False)
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
        
        layout.addStretch()
    
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
        self.equity_plot = pg.PlotWidget(title="Equity Curve")
        self.equity_plot.setLabel('left', 'Equity ($)')
        self.equity_plot.setLabel('bottom', 'Date')
        self.equity_plot.showGrid(x=True, y=True, alpha=0.3)
        self.equity_plot.addLegend()
        layout.addWidget(self.equity_plot, stretch=2)
        
        # Drawdown plot
        self.drawdown_plot = pg.PlotWidget(title="Drawdown")
        self.drawdown_plot.setLabel('left', 'Drawdown (%)')
        self.drawdown_plot.setLabel('bottom', 'Date')
        self.drawdown_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.drawdown_plot, stretch=1)
        
        # Allocation chart
        self.allocation_plot = pg.PlotWidget(title="Asset Allocation")
        self.allocation_plot.setLabel('left', 'Allocation (%)')
        self.allocation_plot.setLabel('bottom', 'Date')
        self.allocation_plot.showGrid(x=True, y=True, alpha=0.3)
        self.allocation_plot.addLegend()
        layout.addWidget(self.allocation_plot, stretch=1)
    
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
        
        if not result.equity_curve:
            return
        
        # Convert dates to numeric for plotting
        dates, equities = result.get_equity_series()
        x = np.arange(len(dates))
        
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
            
            for idx, symbol in enumerate(sorted(all_symbols)):
                alloc_pct = [ep.allocation.get(symbol, 0) * 100 for ep in result.equity_curve]
                color = colors[idx % len(colors)]
                self.allocation_plot.plot(
                    x, alloc_pct,
                    pen=pg.mkPen(color=color, width=2),
                    name=symbol
                )
        
        # Set x-axis labels (showing some date labels)
        if len(dates) > 0:
            # Create axis labels for key dates
            step = max(1, len(dates) // 10)
            ticks = [(i, dates[i].strftime('%Y-%m')) 
                     for i in range(0, len(dates), step)]
            
            axis = self.equity_plot.getAxis('bottom')
            axis.setTicks([ticks])
            
            axis_dd = self.drawdown_plot.getAxis('bottom')
            axis_dd.setTicks([ticks])
            
            axis_alloc = self.allocation_plot.getAxis('bottom')
            axis_alloc.setTicks([ticks])


class BacktestResultsPanel(QWidget):
    """Panel for displaying backtest metrics."""
    
    def __init__(self):
        super().__init__()
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
    
    def update_results(self, result: BacktestResult):
        """Update results display."""
        self.results_text.setPlainText(result.summary())


class BacktestWindow(QMainWindow):
    """Main backtest window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Backtest - Investment Strategy Tester")
        self.setGeometry(100, 100, 1200, 800)
        
        self._setup_ui()
    
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        
        # Left: Control panel
        self.control_panel = BacktestControlPanel(on_run_backtest=self.run_backtest)
        self.control_panel.setFixedWidth(280)
        main_layout.addWidget(self.control_panel)
        
        # Right side splitter
        right_splitter = QSplitter(Qt.Vertical)
        
        # Top: Plot panel
        self.plot_panel = BacktestPlotPanel()
        right_splitter.addWidget(self.plot_panel)
        
        # Bottom: Results panel
        self.results_panel = BacktestResultsPanel()
        self.results_panel.setMaximumHeight(200)
        right_splitter.addWidget(self.results_panel)
        
        right_splitter.setSizes([600, 200])
        main_layout.addWidget(right_splitter)
    
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
            self.results_panel.results_text.setPlainText(
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
            self.results_panel.results_text.setPlainText(
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
                bench_strategy = StrategyRegistry.create("Buy and Hold")
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
            vti_prices = None
            if self.control_panel.should_show_vti_benchmark():
                from .read_data import stock_data_daily
                vti_bars = stock_data_daily("VTI")
                # Filter to date range and extract closing prices
                dates_set = set(ep.date for ep in result.equity_curve)
                vti_prices = [bar.close for bar in vti_bars if bar.date in dates_set]
            
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
                        withdrawal_method=withdrawal_method
                    )
                    vti_withdrawal_result = vti_backtester.run()
            
            # Update UI
            self.plot_panel.update_plots(result, benchmark, vti_prices, vti_withdrawal_result)
            self.results_panel.update_results(result)
            
        except Exception as e:
            self.results_panel.results_text.setPlainText(
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

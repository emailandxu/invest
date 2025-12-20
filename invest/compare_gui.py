from datetime import date
from typing import List, Dict
import pandas as pd
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QListWidget, QListWidgetItem, QPushButton, 
    QDateEdit, QDoubleSpinBox, QGroupBox, QTableWidget, 
    QTableWidgetItem, QHeaderView, QMessageBox,
    QSpinBox, QCheckBox, QComboBox, QGridLayout
)
from PySide6.QtCore import Qt, QDate
import pyqtgraph as pg

from .portfolio_manager import PortfolioManager, PortfolioConfig
from .backtest import create_backtester_from_config, BacktestResult
from .read_data import stock_data_daily

class ComparisonControlPanel(QWidget):
    """Left panel: Portfolio selection and common settings."""
    def __init__(self, on_run_comparison):
        super().__init__()
        self.on_run_comparison = on_run_comparison
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # 1. Portfolio Selection
        group_port = QGroupBox("Select Portfolios")
        layout_port = QVBoxLayout()
        
        self.portfolio_list = QListWidget()
        self.portfolio_list.setSelectionMode(QListWidget.MultiSelection)
        layout_port.addWidget(self.portfolio_list)
        
        # Tools to select all/none
        btn_layout = QHBoxLayout()
        self.btn_all = QPushButton("All")
        self.btn_none = QPushButton("None")
        self.btn_all.clicked.connect(self._select_all)
        self.btn_none.clicked.connect(self._select_none)
        btn_layout.addWidget(self.btn_all)
        btn_layout.addWidget(self.btn_none)
        layout_port.addLayout(btn_layout)
        
        group_port.setLayout(layout_port)
        layout.addWidget(group_port)
        
        # 2. Common Settings
        group_settings = QGroupBox("Common Settings")
        layout_settings = QGridLayout()
        
        # Date Range
        layout_settings.addWidget(QLabel("Start:"), 0, 0)
        self.start_date = QDateEdit(calendarPopup=True)
        self.start_date.setDate(QDate(2015, 1, 1))
        layout_settings.addWidget(self.start_date, 0, 1)
        
        layout_settings.addWidget(QLabel("End:"), 1, 0)
        self.end_date = QDateEdit(calendarPopup=True)
        self.end_date.setDate(QDate.currentDate())
        layout_settings.addWidget(self.end_date, 1, 1)
        
        # Initial Capital
        layout_settings.addWidget(QLabel("Capital:"), 2, 0)
        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(1000, 1000000000)
        self.capital_spin.setValue(100000)
        self.capital_spin.setPrefix("$")
        layout_settings.addWidget(self.capital_spin, 2, 1)
        
        group_settings.setLayout(layout_settings)
        layout.addWidget(group_settings)

        # 3. Withdrawal Strategy
        group_withdrawal = QGroupBox("Withdrawal Strategy")
        layout_withdrawal = QGridLayout()
        
        # Amount
        layout_withdrawal.addWidget(QLabel("Amount:"), 0, 0)
        self.withdrawal_amount = QDoubleSpinBox()
        self.withdrawal_amount.setRange(0, 1000000000)
        self.withdrawal_amount.setValue(0)
        self.withdrawal_amount.setPrefix("$")
        layout_withdrawal.addWidget(self.withdrawal_amount, 0, 1)
        
        # Period
        layout_withdrawal.addWidget(QLabel("Period (Days):"), 1, 0)
        self.withdrawal_period = QSpinBox()
        self.withdrawal_period.setRange(1, 3650)
        self.withdrawal_period.setValue(30)
        layout_withdrawal.addWidget(self.withdrawal_period, 1, 1)

        # Method
        layout_withdrawal.addWidget(QLabel("Method:"), 2, 0)
        self.withdrawal_method = QComboBox()
        self.withdrawal_method.addItems([
            "Proportional", "Rebalance", "Sell Winners", "Sell Losers"
        ])
        layout_withdrawal.addWidget(self.withdrawal_method, 2, 1)
        
        # Inflation
        self.adjust_inflation = QCheckBox("Adjust for Inflation")
        layout_withdrawal.addWidget(self.adjust_inflation, 3, 0, 1, 2)

        group_withdrawal.setLayout(layout_withdrawal)
        layout.addWidget(group_withdrawal)

        
        # 3. Action
        self.run_btn = QPushButton("▶ Run Comparison")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #673AB7;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #5E35B1;
            }
        """)
        self.run_btn.clicked.connect(self.on_run_comparison)
        layout.addWidget(self.run_btn)
        
        # Load portfolios initially
        self.reload_portfolios()

    def reload_portfolios(self):
        """Load from PortfolioManager."""
        self.portfolio_list.clear()
        portfolios = PortfolioManager.load_all()
        for name in sorted(portfolios.keys()):
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.portfolio_list.addItem(item)
            
    def _select_all(self):
        for i in range(self.portfolio_list.count()):
            self.portfolio_list.item(i).setCheckState(Qt.Checked)
            
    def _select_none(self):
        for i in range(self.portfolio_list.count()):
            self.portfolio_list.item(i).setCheckState(Qt.Unchecked)
            
    def get_selected_names(self) -> List[str]:
        names = []
        for i in range(self.portfolio_list.count()):
            item = self.portfolio_list.item(i)
            if item.checkState() == Qt.Checked:
                names.append(item.text())
        return names

    def get_settings(self):
        # Map method display name to internal name
        method_map = {
            "Proportional": "proportional",
            "Rebalance": "rebalance",
            "Sell Winners": "sell_winners",
            "Sell Losers": "sell_losers"
        }
        
        return {
            'start_date': self.start_date.date().toPython(),
            'end_date': self.end_date.date().toPython(),
            'capital': self.capital_spin.value(),
            'withdrawal_amount': self.withdrawal_amount.value(),
            'withdrawal_period_days': self.withdrawal_period.value(),
            'withdrawal_method': method_map.get(self.withdrawal_method.currentText(), "proportional"),
            'adjust_for_inflation': self.adjust_inflation.isChecked()
        }


class ComparisonResultWidget(QWidget):
    """Right panel: Plots and Table."""
    def __init__(self):
        super().__init__()
        self.results = {}
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # 1. Plots
        # Create a splitter or just VBox for plots
        plot_layout = QVBoxLayout()
        
        # Equity Plot
        self.equity_plot = pg.PlotWidget(axisItems={'bottom': pg.DateAxisItem()})
        self.equity_plot.setTitle("Equity Curve")
        self.equity_plot.setLabel('left', 'Equity ($)')
        self.equity_plot.showGrid(x=True, y=True, alpha=0.3)
        self.equity_plot.addLegend()
        self.equity_plot.setBackground('w')
        plot_layout.addWidget(self.equity_plot, stretch=2)
        
        # Drawdown Plot
        self.drawdown_plot = pg.PlotWidget(axisItems={'bottom': pg.DateAxisItem()})
        self.drawdown_plot.setTitle("Drawdown")
        self.drawdown_plot.setLabel('left', 'Drawdown (%)')
        self.drawdown_plot.showGrid(x=True, y=True, alpha=0.3)
        self.drawdown_plot.setBackground('w')
        plot_layout.addWidget(self.drawdown_plot, stretch=1)
        
        # Link X-Axis
        self.drawdown_plot.setXLink(self.equity_plot)
        
        layout.addLayout(plot_layout, stretch=2)
        
        # --- Tooltips ---
        # Equity Tooltip
        self.equity_vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#555', style=Qt.DashLine))
        self.equity_plot.addItem(self.equity_vline, ignoreBounds=True)
        self.equity_vline.setVisible(False)
        
        self.equity_label = pg.TextItem(anchor=(0, 0), fill=pg.mkBrush(255, 255, 255, 220), color='#000')
        self.equity_plot.addItem(self.equity_label, ignoreBounds=True)
        self.equity_label.setVisible(False)
        self.equity_plot.scene().sigMouseMoved.connect(self._on_equity_hover)
        
        # Drawdown Tooltip
        self.drawdown_vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#555', style=Qt.DashLine))
        self.drawdown_plot.addItem(self.drawdown_vline, ignoreBounds=True)
        self.drawdown_vline.setVisible(False)
        
        self.drawdown_label = pg.TextItem(anchor=(0, 0), fill=pg.mkBrush(255, 255, 255, 220), color='#000')
        self.drawdown_plot.addItem(self.drawdown_label, ignoreBounds=True)
        self.drawdown_label.setVisible(False)
        self.drawdown_plot.scene().sigMouseMoved.connect(self._on_drawdown_hover)
        
        # 2. Table
        self.table = QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels([
            "Portfolio", "Total Return", "CAGR", "Max DD", 
            "Sharpe", "Sortino", "Vol", "Trades", "Alpha (%)", "Beta"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSortingEnabled(True)
        layout.addWidget(self.table, stretch=1)
        
    def update_results(self, results: Dict[str, BacktestResult]):
        self.results = results
        
        # Clear Plots
        self.equity_plot.clear()
        self.drawdown_plot.clear()
        
        # Clear legends
        for plot in [self.equity_plot, self.drawdown_plot]:
            if plot.plotItem.legend:
                plot.plotItem.legend.clear()
        
        # Re-add hover elements
        self.equity_plot.addItem(self.equity_vline, ignoreBounds=True)
        self.equity_plot.addItem(self.equity_label, ignoreBounds=True)
        self.drawdown_plot.addItem(self.drawdown_vline, ignoreBounds=True)
        self.drawdown_plot.addItem(self.drawdown_label, ignoreBounds=True)
        
        # Use Hex codes for compatibility with both pyqtgraph and HTML tooltips
        colors = [
            '#0000FF', # Blue
            '#FF0000', # Red
            '#008000', # Green
            '#00FFFF', # Cyan
            '#FF00FF', # Magenta
            '#FFFF00', # Yellow
            '#000000', # Black
            '#FF5722', # Orange
            '#795548', # Brown
            '#607D8B'  # Blue Grey
        ]
        self.item_colors = {} # Store for tooltips
        
        for i, (name, res) in enumerate(results.items()):
            if not res.equity_curve:
                continue
                
            # Prepare data
            # Use timestamp for x-axis with DateAxisItem
            dates = [pd.Timestamp(pt.date).timestamp() for pt in res.equity_curve]
            equities = np.array([pt.equity for pt in res.equity_curve])
            
            # Calculate Drawdown
            peak = np.maximum.accumulate(equities)
            drawdown = (equities - peak) / peak * 100
            
            color = colors[i % len(colors)]
            self.item_colors[name] = color
            
            # Plot Equity
            self.equity_plot.plot(
                dates, equities, 
                pen=pg.mkPen(color=color, width=2), 
                name=name
            )
            
            # Plot Drawdown
            self.drawdown_plot.plot(
                dates, drawdown,
                pen=pg.mkPen(color=color, width=2),
                name=name
            )
        
        self.drawdown_plot.addLegend()
        
        # Update Table
        self.table.setRowCount(len(results))
        self.table.setSortingEnabled(False)
        
        for row, (name, res) in enumerate(results.items()):
            self.table.setItem(row, 0, QTableWidgetItem(name))
            
            def item(val, fmt="{:.2f}"):
                it = QTableWidgetItem(fmt.format(val))
                it.setData(Qt.UserRole, val)
                return it
                
            self.table.setItem(row, 1, item(res.total_return_pct, "{:.2f}%"))
            self.table.setItem(row, 2, item(res.cagr * 100, "{:.2f}%"))
            self.table.setItem(row, 3, item(res.max_drawdown_pct, "{:.2f}%"))
            self.table.setItem(row, 4, item(res.sharpe_ratio))
            self.table.setItem(row, 5, item(res.sortino_ratio))
            self.table.setItem(row, 6, item(res.volatility * 100, "{:.2f}%"))
            self.table.setItem(row, 7, item(len(res.trades), "{:d}"))
            
            alpha_val = res.alpha * 100 if res.alpha is not None else 0.0
            beta_val = res.beta if res.beta is not None else 0.0
            self.table.setItem(row, 8, item(alpha_val, "{:+.2f}%"))
            self.table.setItem(row, 9, item(beta_val, "{:.2f}"))
            
        self.table.setSortingEnabled(True)

    def _get_common_data_at_x(self, x_pos):
        """Helper to collect data from all results at a given x timestamp."""
        data = {}
        try:
            # Use pandas for consistent round-trip conversion
            target_date = pd.to_datetime(x_pos, unit='s').date()
        except (ValueError, OverflowError):
            return {}, date.today()
        
        for name, res in self.results.items():
            if not res.equity_curve:
                continue
            
            # Find closest index
            # This is naive linear search, could be improved with bisect but length is small (<5000)
            # Efficient enough for GUI hover
            closest_ep = None
            min_diff = float('inf')
            
            for ep in res.equity_curve:
                # Convert date to comparable
                # ep.date is datetime.date
                day_diff = abs((ep.date - target_date).days)
                if day_diff < min_diff:
                    min_diff = day_diff
                    closest_ep = ep
                if day_diff == 0:
                    break
            
            if closest_ep and min_diff < 5: # Tolerance of 5 days
                data[name] = closest_ep
                
        return data, target_date

    def _on_equity_hover(self, pos):
        if not self.results:
            return
            
        if not self.equity_plot.sceneBoundingRect().contains(pos):
            self.equity_vline.setVisible(False)
            self.equity_label.setVisible(False)
            return
            
        mouse_point = self.equity_plot.plotItem.vb.mapSceneToView(pos)
        x_val = mouse_point.x()
        
        data_map, target_date = self._get_common_data_at_x(x_val)
        
        if not data_map:
            self.equity_vline.setVisible(False)
            self.equity_label.setVisible(False)
            return
            
        lines = [f"<b>📅 {target_date.strftime('%Y-%m-%d')}</b>"]
        
        # Sort by Equity descending
        sorted_items = sorted(data_map.items(), key=lambda x: x[1].equity, reverse=True)
        
        for name, ep in sorted_items:
            color = self.item_colors.get(name, 'black')
            res = self.results[name]
            ret = (ep.equity / res.initial_capital - 1) * 100
            
            line = f"<span style='color:{color}'>■</span> <b>{name}</b>: ${ep.equity:,.0f} ({ret:+.1f}%)"
            lines.append(line)
            
        self.equity_vline.setPos(x_val)
        self.equity_vline.setVisible(True)
        
        self.equity_label.setHtml("<br>".join(lines))
        self.equity_label.setPos(x_val, mouse_point.y())
        self.equity_label.setVisible(True)

    def _on_drawdown_hover(self, pos):
        if not self.results:
            return
            
        if not self.drawdown_plot.sceneBoundingRect().contains(pos):
            self.drawdown_vline.setVisible(False)
            self.drawdown_label.setVisible(False)
            return
            
        mouse_point = self.drawdown_plot.plotItem.vb.mapSceneToView(pos)
        x_val = mouse_point.x()
        
        data_map, target_date = self._get_common_data_at_x(x_val)
        
        if not data_map:
            self.drawdown_vline.setVisible(False)
            self.drawdown_label.setVisible(False)
            return
            
        lines = [f"<b>📅 {target_date.strftime('%Y-%m-%d')}</b>"]
        
        for name, ep in data_map.items():
            color = self.item_colors.get(name, 'black')
            res = self.results[name]
            
            # Re-calculate drawdown for this point (inefficient but safe)
            # Ideally we should cache arrays, but let's do this for now
            # Find index of this ep
            try:
                # We can optimize this by storing arrays in update_results
                # But let's assume res.equity_curve is ordered
                idx = res.equity_curve.index(ep)
                
                # Get history up to this point
                hist_equity = [p.equity for p in res.equity_curve[:idx+1]]
                peak = max(hist_equity)
                dd = (ep.equity - peak) / peak * 100
                
                line = f"<span style='color:{color}'>■</span> <b>{name}</b>: {dd:.2f}%"
                lines.append(line)
            except ValueError:
                pass
            
        self.drawdown_vline.setPos(x_val)
        self.drawdown_vline.setVisible(True)
        
        self.drawdown_label.setHtml("<br>".join(lines))
        self.drawdown_label.setPos(x_val, mouse_point.y())
        self.drawdown_label.setVisible(True)


class ComparisonWidget(QWidget):
    """Main Comparison Widget combining Control and Results."""
    def __init__(self):
        super().__init__()
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        
        self.control_panel = ComparisonControlPanel(self.run_comparison)
        self.control_panel.setFixedWidth(280)
        layout.addWidget(self.control_panel)
        
        self.result_widget = ComparisonResultWidget()
        layout.addWidget(self.result_widget)
        
    def reload_portfolios(self):
        """Reload portfolio list in control panel."""
        self.control_panel.reload_portfolios()
        
    def run_comparison(self):
        names = self.control_panel.get_selected_names()
        if not names:
            QMessageBox.warning(self, "No Selection", "Please select at least one portfolio to compare.")
            return
            
        settings = self.control_panel.get_settings()
        
        portfolios = PortfolioManager.load_all()
        results = {}
        
        # Load Benchmark Data (VTI) for Alpha/Beta Calculation
        try:
            vti_bars = stock_data_daily("VTI")
            # Filter common range roughly to avoid huge loops, exact alignment happens in calculate_settings
            # Actually calculate_benchmark_metrics handles alignment, we just need the data.
            # But we need to separate prices and dates
            bench_dates = [b.date for b in vti_bars]
            bench_prices = [b.close for b in vti_bars] # Use close or adj_close? Backtester uses close?
            # Ideally use same price type as strategy uses. Backtester seems to use bar.close usually.
        except Exception as e:
            print(f"Warning: Could not load VTI for benchmark: {e}")
            bench_dates, bench_prices = [], []
        
        try:
            for name in names:
                config = portfolios.get(name)
                if not config:
                    continue
                    
                backtester = create_backtester_from_config(
                    config, 
                    settings['start_date'], 
                    settings['end_date'], 
                    settings['capital'],
                    withdrawal_amount=settings['withdrawal_amount'],
                    withdrawal_period_days=settings['withdrawal_period_days'],
                    withdrawal_method=settings['withdrawal_method'],
                    adjust_for_inflation=settings['adjust_for_inflation']
                )
                
                # Run
                res = backtester.run()
                res.strategy_name = name # Override internal strategy name with Portfolio Name for display
                
                # Calculate Alpha/Beta vs VTI
                if bench_dates:
                    res.calculate_benchmark_metrics(bench_prices, bench_dates)
                
                results[name] = res
                
            self.result_widget.update_results(results)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Comparison failed: {e}")
            import traceback
            traceback.print_exc()

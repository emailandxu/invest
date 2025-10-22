import sys
from functools import lru_cache, reduce

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QLabel, QSlider, 
                            QCheckBox, QPushButton, QTextEdit, QComboBox)

from .invest_simulator import InvestmentParams, StrategyBasic, InvestmentYearsResult
from .read_data import (
    get_change_rate_by_year,
    get_value_by_year,
    stock_data,
    portfolio_data,
    interest_data,
    inflation_data,
    inflation_rate_multiplier,
)
from .utils import USD
from ._paths import data_path

def get_strategy(params:InvestmentParams):
    strategy = StrategyBasic.from_params(params)
    return strategy

def get_compare_results(params:InvestmentParams, compare_code: str):
    """Run a comparison simulation and overlay its curves without clearing existing overlays."""
    if not compare_code or compare_code == "None":
        return
    # Build parameters for comparison based on current params
    base = params
    compare_params = InvestmentParams(
        start_year=base.start_year,
        duration=base.duration,
        retire_offset=base.retire_offset,
        start_total=base.start_total,
        cost=base.cost,
        cpi=base.cpi,
        interest_rate=base.interest_rate,
        new_savings=base.new_savings,
        asset_code=compare_code,
        portfolio_data=base.portfolio_data,
        use_asset=True,  # ensure we use asset data for comparison
        use_real_interest=base.use_real_interest,
        use_real_cpi=base.use_real_cpi,
        adptive_withdraw_rate=base.adptive_withdraw_rate,
    )
    compare_years_result = get_strategy(compare_params)()
    return compare_years_result


class PortfolioAllocationWidget(QWidget):
    def __init__(self, params:InvestmentParams=None, on_change=None):
        super().__init__()
        self._on_change = on_change
        self.initial_values = params.portfolio_data if params is not None else portfolio_data()
        self.portfolio_sliders = {}
        self.portfolio_value_labels = {}
        self.header_label = QLabel("Portfolio Allocation")
        self.total_text_label = QLabel("Total")
        self.portfolio_total_label = QLabel()
        self._build_ui()
        self.update_labels()

    def _build_ui(self):
        layout = QGridLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(12)
        layout.addWidget(self.header_label, 0, 0, 1, 3)

        for idx, (code, ratio) in enumerate(self.initial_values.items(), start=1):
            code_label = QLabel(code)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setSingleStep(1)
            slider.setPageStep(5)
            slider_value = max(0, min(100, int(round(ratio * 100))))
            slider.setValue(slider_value)
            value_label = QLabel()

            slider.valueChanged.connect(self._handle_change)

            self.portfolio_sliders[code] = slider
            self.portfolio_value_labels[code] = value_label

            layout.addWidget(code_label, idx, 0)
            layout.addWidget(slider, idx, 1)
            layout.addWidget(value_label, idx, 2)

        total_row = len(self.initial_values) + 1
        layout.addWidget(self.total_text_label, total_row, 0)
        layout.addWidget(self.portfolio_total_label, total_row, 2)

    def set_on_change(self, callback):
        self._on_change = callback

    def _handle_change(self):
        self.update_labels()
        if self._on_change:
            self._on_change()

    def update_labels(self):
        if not self.portfolio_sliders:
            return

        total_raw = sum(slider.value() for slider in self.portfolio_sliders.values())
        normalizer = total_raw if total_raw > 0 else 1

        for code, slider in self.portfolio_sliders.items():
            normalized = (slider.value() / normalizer) * 100
            self.portfolio_value_labels[code].setText(f"{normalized:5.1f}%")

        self.portfolio_total_label.setText(f"Total: {total_raw:.0f}%")

    def set_values(self, values):
        if values is None:
            return

        for code, slider in self.portfolio_sliders.items():
            slider.blockSignals(True)
            slider_value = max(0, min(100, int(round(values.get(code, 0.0) * 100))))
            slider.setValue(slider_value)
            slider.blockSignals(False)
        self.update_labels()

    def get_portfolio_data(self):
        if not self.portfolio_sliders:
            return {}

        raw = {code: slider.value() / 100 for code, slider in self.portfolio_sliders.items()}
        total = sum(raw.values())
        if total > 0:
            return {code: value / total for code, value in raw.items()}
        return {code: 0.0 for code in raw}
    
    def show_portfolio_window(self):
        """Show the portfolio allocation controls in a separate window."""
        self.setWindowTitle("Portfolio Allocation")
        self.show()
        self.raise_()
        self.activateWindow()


class AnalysisReportWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analysis Report")
        self.resize(500, 700)
        layout = QVBoxLayout(self)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        layout.addWidget(self.text)

    def set_report_text(self, text: str):
        self.text.setPlainText(text or "")

    def show_window(self):
        self.show()
        self.raise_()
        self.activateWindow()


class InvestmentControlPanel(QWidget):
    """Control panel widget containing all parameter sliders and checkboxes."""
    
    def __init__(self, plot_panel, on_parameter_change=None, initial_params=None):
        super().__init__()
        self.plot_panel = plot_panel
        self.on_parameter_change = on_parameter_change
        self._update_timer = QTimer(self)
        self._update_timer.setInterval(200)
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._emit_parameter_change)

        self.portfolio_widget = PortfolioAllocationWidget(initial_params, on_change=self._queue_update)
        self.report_window = AnalysisReportWindow()
        self._latest_report_text = ""
        self.setup_ui()
    
    def setup_ui(self):
        CONTROL_PANEL_WIDTH = 250
        self.setFixedWidth(CONTROL_PANEL_WIDTH)
        
        layout = QVBoxLayout(self)
        
        # Parameter grid
        grid_widget = QWidget()
        grid_widget.setFixedWidth(CONTROL_PANEL_WIDTH)
        grid_layout = QGridLayout(grid_widget)
                    
        row = 0
        
        # Start Year
        self.start_year_slider = QSlider(Qt.Horizontal)
        self.start_year_slider.setRange(1954, 2100)
        self.start_year_slider.valueChanged.connect(self._queue_update)
        self.start_year_label = QLabel()
        row += 1
        grid_layout.addWidget(self.start_year_slider, row, 0)
        grid_layout.addWidget(self.start_year_label, row, 1)
        row += 1
        # Duration
        self.duration_slider = QSlider(Qt.Horizontal)
        self.duration_slider.setRange(0, 99)
        self.duration_slider.valueChanged.connect(self._queue_update)
        self.duration_label = QLabel()
        row += 1
        grid_layout.addWidget(self.duration_slider, row, 0)
        grid_layout.addWidget(self.duration_label, row, 1)
        row += 1
        # Retirement Offset
        self.retire_offset_slider = QSlider(Qt.Horizontal)
        self.retire_offset_slider.setRange(0, 99)
        self.retire_offset_slider.valueChanged.connect(self._queue_update)
        self.retire_offset_label = QLabel()
        row += 1
        grid_layout.addWidget(self.retire_offset_slider, row, 0)
        grid_layout.addWidget(self.retire_offset_label, row, 1)
        row += 1
        # Start Total
        self.start_total_slider = QSlider(Qt.Horizontal)
        self.start_total_slider.setRange(1, int(USD(2000)))  # Scale by 10 for decimal precision
        self.start_total_slider.valueChanged.connect(self._queue_update)
        self.start_total_label = QLabel()
        row += 1
        grid_layout.addWidget(self.start_total_slider, row, 0)
        grid_layout.addWidget(self.start_total_label, row, 1)
        row += 1
        # Annual Cost
        self.cost_slider = QSlider(Qt.Horizontal)
        self.cost_slider.setRange(1, int(USD(500)))  # Scale by 100 for decimal precision
        self.cost_slider.valueChanged.connect(self._queue_update)
        self.cost_label = QLabel()
        row += 1
        grid_layout.addWidget(self.cost_slider, row, 0)
        grid_layout.addWidget(self.cost_label, row, 1)
        row += 1
        # CPI
        self.cpi_slider = QSlider(Qt.Horizontal)
        self.cpi_slider.setRange(0, 1000)  # Scale by 10000 for decimal precision
        self.cpi_slider.valueChanged.connect(self._queue_update)
        self.cpi_label = QLabel()
        row += 1
        grid_layout.addWidget(self.cpi_slider, row, 0)
        grid_layout.addWidget(self.cpi_label, row, 1)
        row += 1
        # Interest Rate
        self.interest_rate_slider = QSlider(Qt.Horizontal)
        self.interest_rate_slider.setRange(0, 1500)  # Scale by 10000 for decimal precision
        self.interest_rate_slider.valueChanged.connect(self._queue_update)
        self.interest_rate_label = QLabel()
        row += 1
        grid_layout.addWidget(self.interest_rate_slider, row, 0)
        grid_layout.addWidget(self.interest_rate_label, row, 1)
        row += 1
        # Principle Amount
        self.new_savings_slider = QSlider(Qt.Horizontal)
        self.new_savings_slider.setRange(0, int(USD(1000)))  # Scale by 100 for decimal precision
        self.new_savings_slider.valueChanged.connect(self._queue_update)
        self.new_savings_label = QLabel()
        row += 1
        grid_layout.addWidget(self.new_savings_slider, row, 0)
        grid_layout.addWidget(self.new_savings_label, row, 1)
        row += 1

        stock_dir = data_path("STOCK")
        asset_codes = []
        if stock_dir.exists():
            asset_codes = sorted(p.stem for p in stock_dir.glob("*.csv"))
        asset_codes = ["portfolio"] + [c for c in asset_codes if c]
        
        # Stock Code Selection
        asset_widget = QWidget()
        asset_layout = QHBoxLayout(asset_widget)
        asset_layout.setContentsMargins(0, 0, 0, 0)
        self.asset_code_combo = QComboBox()
        self.asset_code_combo.addItems(asset_codes)
        self.asset_code_combo.currentTextChanged.connect(self._queue_update)
        asset_layout.addWidget(QLabel("Red (main):"))
        asset_layout.addWidget(self.asset_code_combo)
        grid_layout.addWidget(asset_widget, row, 0, 1, 2)
        row += 1

        # Compare Code dropdown and button (default None)
        compare_widget = QWidget()
        compare_layout = QHBoxLayout(compare_widget)
        compare_layout.setContentsMargins(0, 0, 0, 0)
        self.compare_code_combo = QComboBox()
        self.compare_code_combo.addItems(["None"] + asset_codes)
        self.compare_code_combo.currentTextChanged.connect(self._queue_update)
        compare_layout.addWidget(QLabel("Yellow:"))
        compare_layout.addWidget(self.compare_code_combo)
        grid_layout.addWidget(compare_widget, row, 0, 1, 2)
        row += 1

        # Compare Code B dropdown and button (default None)
        compare_widget_b = QWidget()
        compare_layout_b = QHBoxLayout(compare_widget_b)
        compare_layout_b.setContentsMargins(0, 0, 0, 0)
        self.compare_code_combo_b = QComboBox()
        self.compare_code_combo_b.addItems(["None"] + asset_codes)
        self.compare_code_combo_b.currentTextChanged.connect(self._queue_update)
        compare_layout_b.addWidget(QLabel("Green:"))
        compare_layout_b.addWidget(self.compare_code_combo_b)
        grid_layout.addWidget(compare_widget_b, row, 0, 1, 2)
        row += 1
        

        layout.addWidget(grid_widget)

        # Reset and Refresh buttons
        button_widget = QWidget()
        button_layout = QVBoxLayout(button_widget)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(4)
        row1_layout = QHBoxLayout()
        row1_layout.setContentsMargins(0, 0, 0, 0)
        row1_layout.setSpacing(6)
        row2_layout = QHBoxLayout()
        row2_layout.setContentsMargins(0, 0, 0, 0)
        row2_layout.setSpacing(6)
        
        reset_button = QPushButton("Reset")
        # reset_button.setFixedWidth((CONTROL_PANEL_WIDTH-15)//2)
        reset_button.clicked.connect(self.reset)
        row1_layout.addWidget(reset_button)

        self.portfolio_button = QPushButton("Portfolio")
        self.portfolio_button.clicked.connect(self.portfolio_widget.show_portfolio_window)
        row2_layout.addWidget(self.portfolio_button)

        self.report_button = QPushButton("Report")
        self.report_button.clicked.connect(self._show_report_window)
        row2_layout.addWidget(self.report_button)
        
        refresh_button = QPushButton("Refresh")
        # refresh_button.setFixedWidth((CONTROL_PANEL_WIDTH-15)//2)
        refresh_button.clicked.connect(self._emit_parameter_change)
        row1_layout.addWidget(refresh_button)
        button_layout.addLayout(row1_layout)
        button_layout.addLayout(row2_layout)
        
        layout.addWidget(button_widget)

        # Removed inline results panel; report now opens in a pop-out window.

        # Real Data checkboxes
        real_data_widget = QWidget()
        real_data_layout = QGridLayout(real_data_widget)

        # Replace two checkboxes with a dropdown for data source selection
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems([
            "Manual (Sliders)",
            "Asset",
            "Interest Rate",
        ])
        self.data_source_combo.currentIndexChanged.connect(self._queue_update)
        real_data_layout.addWidget(QLabel("Data Source:"), 0, 0)
        real_data_layout.addWidget(self.data_source_combo, 1, 0)
        
        self.adptive_withdraw_rate_checkbox = QCheckBox(text="Adaptive Withdraw")
        self.adptive_withdraw_rate_checkbox.stateChanged.connect(self._queue_update)
        real_data_layout.addWidget(self.adptive_withdraw_rate_checkbox, 3, 0)
        
        self.show_benchmark_checkbox = QCheckBox(text="Show Benchmark")
        self.show_benchmark_checkbox.stateChanged.connect(self._on_ui_control_change)
        real_data_layout.addWidget(self.show_benchmark_checkbox, 4, 0)
        layout.addWidget(real_data_widget)

    def _queue_update(self):
        """Throttle parameter change notifications using a short timer."""
        self.update_labels()
        self._update_timer.start()

    def _emit_parameter_change(self):
        if self.on_parameter_change:
            self.on_parameter_change()

    def _on_ui_control_change(self):
        self.plot_panel.show_benchmark = self.show_benchmark_checkbox.isChecked()
        self.plot_panel.update()

    def update_labels(self):
        """Update all parameter labels with current slider values."""
        self.start_year_label.setText(f"From: {self.start_year_slider.value()}")
        self.duration_label.setText(f"Span: {self.duration_slider.value()}")
        self.retire_offset_label.setText(f"Retire: {self.retire_offset_slider.value()}")
        self.start_total_label.setText(f"Invest: {self.start_total_slider.value() / 10:.1f}")
        self.cost_label.setText(f"Cost: {self.cost_slider.value() / 100:.2f}")
        self.cpi_label.setText(f"CPI: {self.cpi_slider.value() / 10000:.3f}")
        self.interest_rate_label.setText(f"Int.: {self.interest_rate_slider.value() / 10000:.3f}")
        self.new_savings_label.setText(f"Gain: {self.new_savings_slider.value() / 100:.2f}")
        self.portfolio_widget.update_labels()

    def set_parameters(self, params: InvestmentParams):
        """Set all sliders and checkboxes from parameter object."""

        self.start_year_slider.setValue(params.start_year)
        self.duration_slider.setValue(params.duration)
        self.retire_offset_slider.setValue(params.retire_offset)
        self.start_total_slider.setValue(int(params.start_total * 10))
        self.cost_slider.setValue(int(params.cost * 100))
        self.cpi_slider.setValue(int(params.cpi * 10000))
        self.interest_rate_slider.setValue(int(params.interest_rate * 10000))
        self.new_savings_slider.setValue(int(params.new_savings * 100))
        self.data_source_combo.setCurrentText(
            "Asset" if params.use_asset else (
            "Interest Rate" if params.use_real_interest else "Manual (Sliders)"
            )
        )
        self.asset_code_combo.setCurrentText(params.asset_code)
        self.adptive_withdraw_rate_checkbox.setChecked(params.adptive_withdraw_rate)
        self.portfolio_widget.set_values(params.portfolio_data)

        self.compare_code_combo.setCurrentText(params.compare_code[0])
        self.compare_code_combo_b.setCurrentText(params.compare_code[1])

    def get_parameters(self) -> InvestmentParams:
        """Create a new InvestmentParams instance with current UI control values."""
        params = InvestmentParams()
        params.start_year = self.start_year_slider.value()
        params.duration = self.duration_slider.value()
        params.retire_offset = self.retire_offset_slider.value()
        params.start_total = self.start_total_slider.value() / 10  # Scale down for UI precision
        params.cost = self.cost_slider.value() / 100  # Scale down from cents to dollars
        params.cpi = self.cpi_slider.value() / 10000  # Scale down for percentage precision
        params.interest_rate = self.interest_rate_slider.value() / 10000  # Scale down for percentage precision
        params.use_asset = self.data_source_combo.currentText() == "Asset"
        params.use_real_interest = self.data_source_combo.currentText() == "Interest Rate"
        params.use_real_cpi = self.data_source_combo.currentText() != "Manual (Sliders)"
        params.new_savings = self.new_savings_slider.value() / 100  # Scale down from cents to dollars
        params.asset_code = self.asset_code_combo.currentText()
        params.adptive_withdraw_rate = self.adptive_withdraw_rate_checkbox.isChecked()
        params.portfolio_data = self.portfolio_widget.get_portfolio_data()
        params.compare_code = [self.compare_code_combo.currentText(), self.compare_code_combo_b.currentText()]

        return params

    def reset(self):
        """Reset all controls to default values."""
        self.set_parameters(InvestmentParams.get_defaults())
        self.update_labels()
        self._emit_parameter_change()
        # Disconnect signals temporarily to avoid triggering on change
        self.show_benchmark_checkbox.setChecked(True)


    def update(self, msg: str=""):
        """Update the analysis report pop-out window content."""
        self._latest_report_text = msg or ""
        # Update window content if created
        if self.report_window is not None:
            self.report_window.set_report_text(self._latest_report_text)

    def _show_report_window(self):
        if self.report_window is None:
            self.report_window = AnalysisReportWindow()
            self.report_window.set_report_text(self._latest_report_text)
        else:
            # Ensure the window has the latest text before showing
            self.report_window.set_report_text(self._latest_report_text)
        self.report_window.show_window()

class InvestmentPlotPanel(QWidget):
    """Plot panel widget containing all investment simulation plots."""
    
    def __init__(self, show_benchmark=False):
        super().__init__()
        self.setup_ui()
        self.show_benchmark = show_benchmark
        # sequence index for compare overlays to vary color/styles

    def setup_ui(self):
        layout = QGridLayout(self)
        
        # Create list of plot widgets with titles
        plot_configs = [
            ("Total Amount Over Time", "Amount ($)", "Year"),
            ("ROI", "Rate", "Year"),
            ("Withdraw Rate", "Rate", "Year"),
            ("Return", "Return ($)", "Year"),
            ("Withdrawals", "Withdraw ($)", "Year"),
            ("Accumulated ROI (Withdraw)", "Rate", "Year")
        ]
        
        # Create plot widgets from configuration
        self.plots = []
        for title, y_label, x_label in plot_configs:
            plot_widget = pg.PlotWidget(title=title)
            plot_widget.setLabel('left', y_label)
            plot_widget.setLabel('bottom', x_label)
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            self.plots.append(plot_widget)
        
        # Set individual plot references for backward compatibility
        self.plot_total = self.plots[0]
        self.plot_interest_rate = self.plots[1]
        self.plot_withdraw_rate = self.plots[2]
        self.plot_interest_total = self.plots[3]
        self.plot_withdraw = self.plots[4]
        self.plot_ratio = self.plots[5]

        self.setup_layout(layout)

    def setup_layout(self, layout):
        self.plots.clear()
        self.plots.append(self.plot_total)
        self.plots.append(self.plot_interest_rate)
        self.plots.append(self.plot_withdraw_rate)
        self.plots.append(self.plot_interest_total)
        self.plots.append(self.plot_withdraw)
        self.plots.append(self.plot_ratio)
        
        # Automatically arrange plots in grid based on list length
        num_plots = len(self.plots)
        # Calculate optimal grid dimensions (prefer wider layouts)
        cols = int(num_plots ** 0.5)
        if cols > 3:
            cols = 3  # Limit to maximum 3 columns for readability
        rows = (num_plots + cols - 1) // cols  # Ceiling division
        
        # Add plots to grid layout
        for i, plot in enumerate(self.plots):
            row = i // cols
            col = i % cols
            layout.addWidget(plot, row, col)

    def clear_all_plots(self):
        """Clear all plot widgets."""
        self.plot_total.clear()
        self.plot_interest_rate.clear()
        self.plot_withdraw_rate.clear()
        self.plot_interest_total.clear()
        self.plot_withdraw.clear()
        self.plot_ratio.clear()

    def update(self, years_result: InvestmentYearsResult=None):
        """Update all plots with new simulation results."""
        if years_result is not None:
            self.years_result = years_result
        elif self.years_result is not None:
            years_result = self.years_result
        else:
            raise ValueError("years_result is required")
        
        
        self.clear_all_plots()

        self.draw_all_plots(years_result, color="red")
        
        compare_code, compare_code_b = years_result.params.compare_code

        if compare_code and compare_code != "None":
            compare_years_result = get_compare_results(self.years_result.params, compare_code)
            self.draw_all_plots(compare_years_result, color="yellow")
        
        if compare_code_b and compare_code_b != "None":
            compare_years_result_b = get_compare_results(self.years_result.params, compare_code_b)
            self.draw_all_plots(compare_years_result_b, color="green")
        
        if self.show_benchmark and years_result is not None:
            self.draw_benchmark_plots(years_result)
        
        
    def draw_benchmark_plots(self, years_result: InvestmentYearsResult):
        years = years_result.series('year')
        self.plot_total.plot(years, years_result.get_benchmark_total(), 
                            pen=pg.mkPen(color='gray', width=1, style=Qt.DashLine), 
                            name='Inflation Adjusted Start Total')


        # Add benchmark line for real interest rate
        self.plot_interest_rate.plot(years, years_result.get_benchmark_roi(), 
                                    pen=pg.mkPen(color='gray', width=1, style=Qt.DashLine), 
                                    name='Real Interest Rate')

        # 2: benchmark Withdraw with monthly withraw
        withdraw_data = years_result.series('withdraw', lambda v: round(v, 2))
        benchmark_withdrawals = years_result.get_benchmark_withdraw()
        # Add benchmark line for withdrawal plot
        self.plot_withdraw.plot(years, benchmark_withdrawals, 
                            pen=pg.mkPen(color='gray', width=1, style=Qt.DashLine), 
                            name='Target Cost')
        self.plot_withdraw.setYRange(0, max(max(withdraw_data)*1.5, max(benchmark_withdrawals)*1.5))


        # Add text labels showing exact values on each point
        step = max(1, len(years) // 25)
        for i, (year, value) in enumerate(zip(years, withdraw_data)):
            if i % step:
                continue
            monthly_withdraw_inflation = (value+1e-16)/(benchmark_withdrawals[i]+1e-16) * years_result.params.cost / 12
            monthly_withdraw_text = f'{monthly_withdraw_inflation*100:.0f}'
            # monthly_withdraw_text += "\n\n" + f"{value / 12 * 100:.0f}"
            text_item = pg.TextItem(monthly_withdraw_text, anchor=(0.5, 1.2), color='white')
            font = text_item.textItem.font()
            font.setPointSize(6)
            text_item.textItem.setFont(font)
            text_item.setPos(year, value)
            self.plot_withdraw.addItem(text_item)


        # 3: plot interest vs principle ratio benchmark
        # Add benchmark line for interest vs principle plot
        benchmark_ratio = [years_result.params.get_real_inflation_rate_multiplier(year) - 1.0 for year in years]  # Line at 1.0 representing break-even
        self.plot_ratio.plot(years, benchmark_ratio, 
                            pen=pg.mkPen(color='gray', width=1, style=Qt.DashLine), 
                            name='Break-even')

    def draw_all_plots(self, years_result: InvestmentYearsResult=None, color="blue"):
        """Update all plots with simulation results."""
        years = years_result.series('year')
        totals = years_result.series('total')
        interest_rates = years_result.series('interest_rate', lambda v: round(v, 4))
        withdraw_rates = years_result.series('withdraw_rate', lambda v: round(v, 4))
        interest_totals = years_result.series('interest_total')
        withdrawals = years_result.series('withdraw', lambda v: round(v, 2))
        ratios = years_result.series('withdrawed_interest_vs_principle')

        self.plot_total.plot(years, totals, pen=pg.mkPen(color=color, width=2), 
                             symbol='o', symbolSize=4, symbolBrush=color)
        

        # Plot 2: Interest Rate
        self.plot_interest_rate.plot(years, interest_rates,
                                     pen=pg.mkPen(color=color, width=2),
                                     symbol='s', symbolSize=4, symbolBrush=color)


        # Plot 3: Withdraw Rate
        self.plot_withdraw_rate.plot(years, withdraw_rates,
                                     pen=pg.mkPen(color=color, width=2),
                                     symbol='x', symbolSize=4, symbolBrush=color)

        # Plot 4: Annual Interest
        self.plot_interest_total.plot(years, interest_totals, pen=pg.mkPen(color=color, width=2),
                                      symbol='t', symbolSize=4, symbolBrush=color)

        # Plot 5: Annual Withdrawals
        self.plot_withdraw.plot(years, withdrawals,
                                pen=pg.mkPen(color=color, width=2),
                                symbol='h', symbolSize=4, symbolBrush=color)
        


        # Plot 6: Interest VS Principle
        self.plot_ratio.plot(years, ratios,
                             pen=pg.mkPen(color=color, width=2),
                             symbol='d', symbolSize=4, symbolBrush=color)
        
        # Also plot interest vs principle
        # self.plot_ratio.plot(years, years_result.series('interest_vs_principle'), 
        #                     pen=pg.mkPen(color=color, width=2), 
        #                     symbol='o', symbolSize=4, symbolBrush=color)
 

class InvestmentSimulatorGui(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Investment Simulator")
        self.setGeometry(100, 100, 900, 900)
        
        # Initialize with default parameters
        self.params = InvestmentParams.get_defaults()
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Create control and plot panels
        self.plot_panel = InvestmentPlotPanel()
        self.control_panel = InvestmentControlPanel(
            self.plot_panel,
            on_parameter_change=self.update,
            initial_params=self.params,
        )
        
        main_layout.addWidget(self.control_panel)
        main_layout.addWidget(self.plot_panel)
        
        # Initial simulation run
        self.reset()

    def reset(self):
        self.control_panel.reset()
    
    def update(self):
        try:
            self.params = self.control_panel.get_parameters()
            
            year_range = f"{self.params.start_year}-{self.params.end_year}"
            self.setWindowTitle(f"Investment Simulator ({year_range})")
            
            years_result = self.run_simulation(self.params)
            self.control_panel.update(years_result.analysis_report())
            self.plot_panel.update(years_result)
        except Exception as e:
            self.control_panel.update(f"Error: {str(e)}")
            raise e
        
    def run_simulation(self, params: InvestmentParams):
        """Run the investment simulation with current parameters."""
        strategy = get_strategy(params)
        return strategy()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape or event.key() == Qt.Key_Q:
            QApplication.instance().quit()
        elif event.key() == Qt.Key_Left:
            current_year = self.control_panel.start_year_slider.value()
            self.control_panel.start_year_slider.setValue(current_year - 1)
        elif event.key() == Qt.Key_Right:
            current_year = self.control_panel.start_year_slider.value()
            self.control_panel.start_year_slider.setValue(current_year + 1)
        super().keyPressEvent(event)

    @classmethod
    def main(cls, args=None):
        if args is None:
            args = sys.argv

        app = QApplication(args)
        window = cls()
        window.show()
        app.exec_()

def main(args=None):
    """Launch the Investment Simulator GUI."""
    InvestmentSimulatorGui.main(args)


if __name__ == "__main__":
    main()
    # import cProfile
    # import pstats
    # import io
    
    # # Create profiler
    # profiler = cProfile.Profile()
    # profiler.enable()
    
    # try:
    #     InvestmentSimulatorGUI.main()
    # finally:
    #     profiler.disable()
        
    #     # Create a string buffer to capture profiler output
    #     s = io.StringIO()
    #     ps = pstats.Stats(profiler, stream=s)
    #     ps.sort_stats('cumulative')
    #     # Filter to only show functions from our code (not built-ins or libraries)
    #     ps.print_stats('invest.*\.py', 50)
        
    #     print("\n" + "="*50)
    #     print("PERFORMANCE PROFILE")
    #     print("="*50)
    #     print(s.getvalue())

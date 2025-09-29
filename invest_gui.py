import os
import sys
from functools import lru_cache, reduce

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QLabel, QSlider, 
                            QCheckBox, QPushButton, QTextEdit, QComboBox)

from invest import invest
from invest_simulator import InvestmentParams, StrategyBasic, InvestmentYearsResult
from read_data import get_change_rate_by_year, get_value_by_year, stock_data, portfolio_data, interest_data, inflation_data, inflation_rate_multiplier
from utils import USD


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


class InvestmentControlPanel(QWidget):
    """Control panel widget containing all parameter sliders and checkboxes."""
    
    def __init__(self, plot_panel, on_parameter_change=None, initial_params=None):
        super().__init__()
        self.plot_panel = plot_panel
        self.on_parameter_change = on_parameter_change
        self.portfolio_widget = PortfolioAllocationWidget(initial_params, on_change=self._on_change)
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
        self.start_year_slider.valueChanged.connect(self._on_change)
        self.start_year_label = QLabel()
        row += 1
        grid_layout.addWidget(self.start_year_slider, row, 0)
        grid_layout.addWidget(self.start_year_label, row, 1)
        row += 1
        # Duration
        self.duration_slider = QSlider(Qt.Horizontal)
        self.duration_slider.setRange(0, 99)
        self.duration_slider.valueChanged.connect(self._on_change)
        self.duration_label = QLabel()
        row += 1
        grid_layout.addWidget(self.duration_slider, row, 0)
        grid_layout.addWidget(self.duration_label, row, 1)
        row += 1
        # Retirement Offset
        self.retire_offset_slider = QSlider(Qt.Horizontal)
        self.retire_offset_slider.setRange(0, 99)
        self.retire_offset_slider.valueChanged.connect(self._on_change)
        self.retire_offset_label = QLabel()
        row += 1
        grid_layout.addWidget(self.retire_offset_slider, row, 0)
        grid_layout.addWidget(self.retire_offset_label, row, 1)
        row += 1
        # Start Total
        self.start_total_slider = QSlider(Qt.Horizontal)
        self.start_total_slider.setRange(1, int(USD(2000)))  # Scale by 10 for decimal precision
        self.start_total_slider.valueChanged.connect(self._on_change)
        self.start_total_label = QLabel()
        row += 1
        grid_layout.addWidget(self.start_total_slider, row, 0)
        grid_layout.addWidget(self.start_total_label, row, 1)
        row += 1
        # Annual Cost
        self.cost_slider = QSlider(Qt.Horizontal)
        self.cost_slider.setRange(1, int(USD(500)))  # Scale by 100 for decimal precision
        self.cost_slider.valueChanged.connect(self._on_change)
        self.cost_label = QLabel()
        row += 1
        grid_layout.addWidget(self.cost_slider, row, 0)
        grid_layout.addWidget(self.cost_label, row, 1)
        row += 1
        # CPI
        self.cpi_slider = QSlider(Qt.Horizontal)
        self.cpi_slider.setRange(0, 1000)  # Scale by 10000 for decimal precision
        self.cpi_slider.valueChanged.connect(self._on_change)
        self.cpi_label = QLabel()
        row += 1
        grid_layout.addWidget(self.cpi_slider, row, 0)
        grid_layout.addWidget(self.cpi_label, row, 1)
        row += 1
        # Interest Rate
        self.interest_rate_slider = QSlider(Qt.Horizontal)
        self.interest_rate_slider.setRange(0, 1500)  # Scale by 10000 for decimal precision
        self.interest_rate_slider.valueChanged.connect(self._on_change)
        self.interest_rate_label = QLabel()
        row += 1
        grid_layout.addWidget(self.interest_rate_slider, row, 0)
        grid_layout.addWidget(self.interest_rate_label, row, 1)
        row += 1
        # Principle Amount
        self.new_savings_slider = QSlider(Qt.Horizontal)
        self.new_savings_slider.setRange(0, int(USD(1000)))  # Scale by 100 for decimal precision
        self.new_savings_slider.valueChanged.connect(self._on_change)
        self.new_savings_label = QLabel()
        row += 1
        grid_layout.addWidget(self.new_savings_slider, row, 0)
        grid_layout.addWidget(self.new_savings_label, row, 1)
        row += 1

        # Stock Code Selection
        self.asset_code_combo = QComboBox()
        # self.asset_code_combo.addItems(["portfolio", "COKE", "BRKB"])
        asset_codes = [path.split(".")[0].strip() for path in os.listdir("data/STOCK") if path.endswith(".csv")]
        asset_codes = ["portfolio"] + [c for c in asset_codes if c]
        self.asset_code_combo.addItems(asset_codes)
        self.asset_code_combo.currentTextChanged.connect(self._on_change)
        self.asset_code_label = QLabel("Asset Code:")
        row += 1
        grid_layout.addWidget(self.asset_code_label, row, 0)
        grid_layout.addWidget(self.asset_code_combo, row, 1)
        row += 1

        layout.addWidget(grid_widget)

        # Reset and Refresh buttons
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        button_layout.setContentsMargins(0, 0, 0, 0)
        
        reset_button = QPushButton("Reset")
        # reset_button.setFixedWidth((CONTROL_PANEL_WIDTH-15)//2)
        reset_button.clicked.connect(self.reset)
        button_layout.addWidget(reset_button)

        self.portfolio_button = QPushButton("Portfolio")
        self.portfolio_button.clicked.connect(self.portfolio_widget.show_portfolio_window)
        button_layout.addWidget(self.portfolio_button)
        
        refresh_button = QPushButton("Refresh")
        # refresh_button.setFixedWidth((CONTROL_PANEL_WIDTH-15)//2)
        refresh_button.clicked.connect(self._on_change)
        button_layout.addWidget(refresh_button)
        
        layout.addWidget(button_widget)
        
        # Results display
        conclusion_widget = QWidget()
        conclusion_widget.setFixedWidth(CONTROL_PANEL_WIDTH)
        conclusion_layout = QVBoxLayout(conclusion_widget)
        conclusion_layout.setContentsMargins(0, 0, 0, 0)
        
        self.conclusion_text_widget = QTextEdit()
        self.conclusion_text_widget.setReadOnly(True)
        conclusion_layout.addWidget(self.conclusion_text_widget)
        
        layout.addWidget(conclusion_widget)

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
        self.data_source_combo.currentIndexChanged.connect(self._on_change)
        real_data_layout.addWidget(QLabel("Data Source:"), 0, 0)
        real_data_layout.addWidget(self.data_source_combo, 1, 0)
        
        self.use_real_cpi_checkbox = QCheckBox(text="American CPI")
        self.use_real_cpi_checkbox.stateChanged.connect(self._on_change)
        real_data_layout.addWidget(self.use_real_cpi_checkbox, 2, 0)
        
        self.adptive_withdraw_rate_checkbox = QCheckBox(text="Adaptive Withdraw")
        self.adptive_withdraw_rate_checkbox.stateChanged.connect(self._on_change)
        real_data_layout.addWidget(self.adptive_withdraw_rate_checkbox, 3, 0)
        
        self.show_benchmark_checkbox = QCheckBox(text="Show Benchmark")
        self.show_benchmark_checkbox.stateChanged.connect(self._on_ui_control_change)
        real_data_layout.addWidget(self.show_benchmark_checkbox, 4, 0)
        layout.addWidget(real_data_widget)

    def _on_change(self):
        """Internal callback that updates labels and triggers external callback."""
        self.update_labels()
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
        self.use_real_cpi_checkbox.setChecked(params.use_real_cpi)
        self.asset_code_combo.setCurrentText(params.asset_code)
        self.adptive_withdraw_rate_checkbox.setChecked(params.adptive_withdraw_rate)
        self.portfolio_widget.set_values(params.portfolio_data)


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
        params.use_real_cpi = self.use_real_cpi_checkbox.isChecked()
        params.new_savings = self.new_savings_slider.value() / 100  # Scale down from cents to dollars
        params.asset_code = self.asset_code_combo.currentText()
        params.adptive_withdraw_rate = self.adptive_withdraw_rate_checkbox.isChecked()
        params.portfolio_data = self.portfolio_widget.get_portfolio_data()
        return params

    def reset(self):
        """Reset all controls to default values."""
        self.set_parameters(InvestmentParams.get_defaults())
        self._on_change()
        # Disconnect signals temporarily to avoid triggering on change
        self.show_benchmark_checkbox.setChecked(True)


    def update(self, msg: str=""):
        """Update the results text display with formatted analysis."""
        self.conclusion_text_widget.setPlainText(msg)

class InvestmentPlotPanel(QWidget):
    """Plot panel widget containing all investment simulation plots."""
    
    def __init__(self, show_benchmark=False):
        super().__init__()
        self.setup_ui()
        self.show_benchmark = show_benchmark

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
        """Update all plots with simulation results."""
        if years_result is not None:
            self.years_result = years_result
        elif self.years_result is not None:
            years_result = self.years_result
        else:
            raise ValueError("years_result is required")

        self.clear_all_plots()
        
        # Helper function to extract series data
        years = years_result.series('year')
        # Plot 1: Total Amount
        self.plot_total.plot(years, years_result.series('total'), pen=pg.mkPen(color='blue', width=2), 
                            symbol='o', symbolSize=4, symbolBrush='blue')
        
        if self.show_benchmark:
            self.plot_total.plot(years, years_result.get_benchmark_total(), 
                                pen=pg.mkPen(color='gray', width=1, style=Qt.DashLine), 
                                name='Inflation Adjusted Start Total')
        
        # Plot 2: Interest Rate
        self.plot_interest_rate.plot(years, years_result.series('interest_rate', lambda v: round(v, 4)), 
                                    pen=pg.mkPen(color='green', width=2), 
                                    symbol='s', symbolSize=4, symbolBrush='green')
        
        if self.show_benchmark:
            # Add benchmark line for real interest rate
            self.plot_interest_rate.plot(years, years_result.get_benchmark_roi(), 
                                        pen=pg.mkPen(color='gray', width=1, style=Qt.DashLine), 
                                        name='Real Interest Rate')
        
        # Plot 3: Withdraw Rate
        self.plot_withdraw_rate.plot(years, years_result.series('withdraw_rate', lambda v: round(v, 4)), 
                                              pen=pg.mkPen(color='orange', width=2), 
                                              symbol='x', symbolSize=4, symbolBrush='orange')
        
        # Plot 4: Annual Interest
        self.plot_interest_total.plot(years, years_result.series('interest_total'), pen=pg.mkPen(color='purple', width=2), 
                               symbol='t', symbolSize=4, symbolBrush='purple')

        # Plot 5: Annual Withdrawals
        withdraw_data = years_result.series('withdraw', lambda v: round(v, 2))
        self.plot_withdraw.plot(years, withdraw_data, 
                               pen=pg.mkPen(color='brown', width=2), 
                               symbol='h', symbolSize=4, symbolBrush='brown')
        
        if self.show_benchmark:
            benchmark_withdrawals = years_result.get_benchmark_withdraw()
            # Add benchmark line for withdrawal plot
            self.plot_withdraw.plot(years, benchmark_withdrawals, 
                                pen=pg.mkPen(color='gray', width=1, style=Qt.DashLine), 
                                name='Target Cost')
            # Add text labels showing exact values on each point
            for i, (year, value) in enumerate(zip(years, withdraw_data)):
                text_item = pg.TextItem(f'{(value+1e-16)/(benchmark_withdrawals[i]+1e-16) * years_result.params.cost / 12:.2f}', anchor=(0.5, 1.2), color='yellow')
                font = text_item.textItem.font()
                font.setPointSize(6)
                text_item.textItem.setFont(font)
                text_item.setPos(year, value)
                self.plot_withdraw.addItem(text_item)

        # Plot 6: Interest VS Principle
        self.plot_ratio.plot(years, years_result.series('withdrawed_interest_vs_principle'), 
                            pen=pg.mkPen(color='red', width=2), 
                            symbol='d', symbolSize=4, symbolBrush='red')
        
        # Also plot interest vs principle
        self.plot_ratio.plot(years, years_result.series('interest_vs_principle'), 
                            pen=pg.mkPen(color='blue', width=2), 
                            symbol='o', symbolSize=4, symbolBrush='blue')

        if self.show_benchmark:
            # Add benchmark line for interest vs principle plot
            benchmark_ratio = [years_result.params.get_real_inflation_rate_multiplier(year) - 1.0 for year in years]  # Line at 1.0 representing break-even
            self.plot_ratio.plot(years, benchmark_ratio, 
                                pen=pg.mkPen(color='gray', width=1, style=Qt.DashLine), 
                                name='Break-even')

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
        self.control_panel = InvestmentControlPanel(self.plot_panel, on_parameter_change=self.update, initial_params=self.params)
        
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
        strategy = StrategyBasic.from_params(params)
        return strategy()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
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

if __name__ == "__main__":
    InvestmentSimulatorGui.main()
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

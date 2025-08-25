import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QLabel, QSlider, 
                            QCheckBox, QPushButton, QTextEdit)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import numpy as np
from functools import reduce
from read_data import get_change_rate_by_year, get_value_by_year, sp500_data, interest_data, inflation_data
from invest import invest
from utils import USD

"""Create GUI for adjusting investment parameters and plotting results"""

from dataclasses import dataclass

@dataclass
class InvestmentParams:
    """Data class representing investment simulation parameters."""
    start_year: int = 1995
    duration: int = 25
    retire_offset: int = 0
    start_total: float = USD(30)
    cost: float = USD(1.2)
    cpi: float = 0.00
    interest_rate: float = 0.00
    use_sp500: bool = False
    use_real_interest: bool = False
    use_real_cpi: bool = False
    new_savings: float = USD(3.0)
    
    @property
    def end_year(self) -> int:
        """Calculate the end year of the simulation."""
        return self.start_year + self.duration
    
    @property
    def retire_year(self) -> int:
        """Calculate the retirement year."""
        return self.start_year + self.retire_offset
    
    @classmethod
    def get_defaults(cls) -> 'InvestmentParams':
        """Create an instance with default values."""
        return cls()
    
    def get_interest_rate(self, year):
        """Get interest rate for a given year based on parameters."""
        if self.use_sp500:
            return get_change_rate_by_year(sp500_data(), year, default=self.interest_rate)
        elif self.use_real_interest:
            return get_value_by_year(interest_data(), year, default=self.interest_rate)
        else:
            return self.interest_rate
    
    def get_inflation_rate_multiplier(self, year):
        """Get cumulative inflation rate multiplier for a given year."""
        if self.use_real_cpi:
            return reduce(
                lambda x, y: x * y, 
                [(1.0 + get_value_by_year(inflation_data(), y, default=self.cpi)) 
                 for y in range(self.start_year, year)], 
                1
            )
        else:
            return (1 + self.cpi) ** (year - self.start_year)
    
    def get_new_savings(self, year):
        """Get new savings amount for a given year."""
        if year < self.retire_year:
            return self.new_savings * self.get_inflation_rate_multiplier(year)
        else:
            return 0
    
    def get_withdraw_rate(self, year, total):
        """Get withdrawal rate for a given year and total."""
        return (self.cost / (total + 1e-16)) * self.get_inflation_rate_multiplier(year)

class InvestmentYearsResult:
    def __init__(self, params, years_result):
        self.params = params
        self.years_result = years_result
    
    @property
    def financial_stats(self):
        """Calculate financial metrics from simulation results."""
        if not self.years_result:
            return {}
        
        final_total = self.years_result[-1]['total']
        final_withdraw_total = self.years_result[-1]['withdraw_total']
        final_interest_total = self.years_result[-1]['interest_total']
        
        # Find the year when total becomes zero or near zero
        zero_year = None
        for row in self.years_result:
            if row['withdrawed_interest_vs_principle'] < -0.999:
                zero_year = row['year']
                break
        
        # Calculate growth rate
        duration = (self.params.end_year if zero_year is None else zero_year) - self.params.start_year
        if duration > 0 and self.params.start_total > 0:
            growth_rate = (((final_total + 1e-4) / self.params.start_total) ** (1 / duration) - 1)
        else:
            growth_rate = 0.0
        
        return {
            'final_total': final_total,
            'final_withdraw_total': final_withdraw_total,
            'final_interest_total': final_interest_total,
            'zero_year': zero_year,
            'growth_rate': growth_rate,
            'inflation_rate': self.params.get_inflation_rate_multiplier(self.params.end_year)
        }
    
    @property
    def interest_rate_stats(self):
        """Calculate interest rate statistics from simulation results."""
        if not self.years_result:
            return {}
        
        interest_rates = np.array([row['interest_rate'] for row in self.years_result])
        
        return {
            'mean_rate': np.mean(interest_rates),
            'std_rate': np.std(interest_rates),
            'min_rate': np.min(interest_rates),
            'max_rate': np.max(interest_rates)
        }

    @property
    def financial_summary(self):
        """Format financial summary section of results."""
        metrics = self.financial_stats
        text = "💰 FINANCIAL SUMMARY:\n"
        text += f"  • Final:         {metrics['final_total']:>12,.2f}$\n"
        text += f"  • Withdraw Total: {metrics['final_withdraw_total']:>12,.2f}$\n"
        text += f"  • Interest Total: {metrics['final_interest_total']:>12,.2f}$\n"
        text += f"  • CGAR:  {metrics['growth_rate']:>12.2%}\n"
        text += f"  • Inflation Rate: {metrics['inflation_rate']:>12.2%}\n"
        text += "\n"
        return text
    
    @property
    def interest_rate_summary(self):
        """Format interest rate statistics section of results."""
        stats = self.interest_rate_stats
        text = "📊 INTEREST RATE STATS:\n"
        text += f"  • Mean Rate:           {stats['mean_rate']:>12.3%}\n"
        text += f"  • Standard Deviation:  {stats['std_rate']:>12.3%}\n"
        text += f"  • Min Rate:            {stats['min_rate']:>12.3%}\n"
        text += f"  • Max Rate:            {stats['max_rate']:>12.3%}\n"
        text += "\n"
        return text

    @property
    def sustainability_summary(self):
        """Format sustainability analysis section of results."""
        metrics = self.financial_stats
        text = "🔍 SUSTAINABILITY:\n"
        
        if metrics['zero_year'] is not None:
            years_lasted = metrics['zero_year'] - self.params.start_year
            retirement_years_lasted = metrics['zero_year'] - self.params.retire_year
            text += f"  • Status:              😑 DEPLETED\n"
            text += f"  • Survive Until:      {metrics['zero_year']:>12}\n"
            text += f"  • Total Years Lasted:  {years_lasted:>12}\n"
            text += f"  • Retirement Years:    {retirement_years_lasted:>12}\n"
        else:
            years_lasted = self.params.end_year - self.params.start_year
            retirement_years_lasted = self.params.end_year - self.params.retire_year
            text += f"  • Status:              😁 SUSTAINABLE\n"
            text += f"  • Survive Until:          {self.params.end_year:>12}\n"
            text += f"  • Total Years Lasted:  {years_lasted:>12}\n"
            text += f"  • Retirement Years:    {retirement_years_lasted:>12}\n"
        
        return text

    def analysis_report(self):
        """Generate a comprehensive analysis report from the simulation results."""
        results_text = ""
        results_text += self.financial_summary
        results_text += self.interest_rate_summary
        results_text += self.sustainability_summary
        return results_text
    
    def __str__(self):
        return self.analysis_report()
    
    def series(self, key, func=lambda v: v):
        return [func(row[key]) for row in self.years_result]

class InvestmentControlPanel(QWidget):
    """Control panel widget containing all parameter sliders and checkboxes."""
    
    def __init__(self, on_parameter_change=None):
        super().__init__()
        self.on_parameter_change = on_parameter_change
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
        self.start_year_slider.setRange(1970, 2100)
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
        self.start_total_slider.setRange(0, int(USD(2000)))  # Scale by 10 for decimal precision
        self.start_total_slider.valueChanged.connect(self._on_change)
        self.start_total_label = QLabel()
        row += 1
        grid_layout.addWidget(self.start_total_slider, row, 0)
        grid_layout.addWidget(self.start_total_label, row, 1)
        row += 1
        # Annual Cost
        self.cost_slider = QSlider(Qt.Horizontal)
        self.cost_slider.setRange(0, int(USD(500)))  # Scale by 100 for decimal precision
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
        
        # Real Data checkboxes
        real_data_widget = QWidget()
        real_data_layout = QGridLayout(real_data_widget)
                    
        self.use_real_interest_checkbox = QCheckBox(text="int.")
        self.use_real_interest_checkbox.stateChanged.connect(self._on_change)
        real_data_layout.addWidget(self.use_real_interest_checkbox, 0, 1)

        self.use_real_cpi_checkbox = QCheckBox(text="cpi")
        self.use_real_cpi_checkbox.stateChanged.connect(self._on_change)
        real_data_layout.addWidget(self.use_real_cpi_checkbox, 0, 2)
        
        self.use_sp500_checkbox = QCheckBox(text="sp500")
        self.use_sp500_checkbox.stateChanged.connect(self._on_change)
        real_data_layout.addWidget(self.use_sp500_checkbox, 0, 3)
        grid_layout.addWidget(real_data_widget, row, 0, 1, 2)

        layout.addWidget(grid_widget)

        # Reset button
        reset_button = QPushButton("Reset")
        reset_button.setFixedWidth(CONTROL_PANEL_WIDTH-10)
        reset_button.clicked.connect(self.reset)
        layout.addWidget(reset_button)
        
        # Results display
        conclusion_widget = QWidget()
        conclusion_widget.setFixedWidth(CONTROL_PANEL_WIDTH)
        conclusion_layout = QVBoxLayout(conclusion_widget)
        conclusion_layout.setContentsMargins(0, 0, 0, 0)
        
        self.conclusion_text_widget = QTextEdit()
        self.conclusion_text_widget.setReadOnly(True)
        conclusion_layout.addWidget(self.conclusion_text_widget)
        
        layout.addWidget(conclusion_widget)

    def _on_change(self):
        """Internal callback that updates labels and triggers external callback."""
        self.update_labels()
        if self.on_parameter_change:
            self.on_parameter_change()

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
        self.use_sp500_checkbox.setChecked(params.use_sp500)
        self.use_real_interest_checkbox.setChecked(params.use_real_interest)
        self.use_real_cpi_checkbox.setChecked(params.use_real_cpi)
        self._on_change()

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
        params.use_sp500 = self.use_sp500_checkbox.isChecked()
        params.use_real_interest = self.use_real_interest_checkbox.isChecked()
        params.use_real_cpi = self.use_real_cpi_checkbox.isChecked()
        params.new_savings = self.new_savings_slider.value() / 100  # Scale down from cents to dollars
        return params

    def reset(self):
        """Reset all controls to default values."""
        self.set_parameters(InvestmentParams.get_defaults())

    def update(self, msg: str):
        """Update the results text display with formatted analysis."""
        self.conclusion_text_widget.setPlainText(msg)

class InvestmentPlotPanel(QWidget):
    """Plot panel widget containing all investment simulation plots."""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
    
    def setup_ui(self):
        layout = QGridLayout(self)
        
        # Create list of plot widgets with titles
        plot_configs = [
            ("Total Amount Over Time", "Amount ($)", "Year"),
            ("Interest Rate", "Rate", "Year"),
            ("Withdrawed Interest Rate", "Rate", "Year"),
            ("Interest Total", "Interest ($)", "Year"),
            ("Withdrawals", "Withdraw ($)", "Year"),
            ("Withdrawed Interest VS Principle", "Rate", "Year")
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
        self.plot_withdrawed_interest_rate = self.plots[2]
        self.plot_interest = self.plots[3]
        self.plot_withdraw = self.plots[4]
        self.plot_ratio = self.plots[5]

        self.setup_layout(layout)

    def setup_layout(self, layout):
        self.plots.clear()
        self.plots.append(self.plot_total)
        # self.plots.append(self.plot_interest_rate)
        # self.plots.append(self.plot_withdrawed_interest_rate)
        # self.plots.append(self.plot_interest)
        # self.plots.append(self.plot_withdraw)
        # self.plots.append(self.plot_ratio)
        
        # Automatically arrange plots in grid based on list length
        num_plots = len(self.plots)
        # Calculate optimal grid dimensions (prefer wider layouts)
        cols = int(num_plots ** 0.5) + 1
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
        self.plot_withdrawed_interest_rate.clear()
        self.plot_interest.clear()
        self.plot_withdraw.clear()
        self.plot_ratio.clear()
    
    def update(self, years_result: InvestmentYearsResult):
        """Update all plots with simulation results."""
        self.clear_all_plots()
        
        # Helper function to extract series data
        years = years_result.series('year')
        
        # Plot 1: Total Amount
        self.plot_total.plot(years, years_result.series('total'), pen=pg.mkPen(color='blue', width=2), 
                            symbol='o', symbolSize=4, symbolBrush='blue')
        
        # Plot 2: Interest Rate
        self.plot_interest_rate.plot(years, years_result.series('interest_rate', lambda v: round(v, 4)), 
                                    pen=pg.mkPen(color='green', width=2), 
                                    symbol='s', symbolSize=4, symbolBrush='green')
        
        # Plot 3: Withdraw Rate
        self.plot_withdrawed_interest_rate.plot(years, years_result.series('withdrawed_interest_rate', lambda v: round(v, 4)), 
                                              pen=pg.mkPen(color='orange', width=2), 
                                              symbol='x', symbolSize=4, symbolBrush='orange')
        
        # Plot 4: Annual Interest
        self.plot_interest.plot(years, years_result.series('interest_total'), pen=pg.mkPen(color='purple', width=2), 
                               symbol='t', symbolSize=4, symbolBrush='purple')
        
        # Plot 5: Annual Withdrawals
        self.plot_withdraw.plot(years, years_result.series('withdraw', lambda v: round(v, 2)), 
                               pen=pg.mkPen(color='brown', width=2), 
                               symbol='h', symbolSize=4, symbolBrush='brown')
        
        # Plot 6: Interest VS Principle
        self.plot_ratio.plot(years, years_result.series('withdrawed_interest_vs_principle'), 
                            pen=pg.mkPen(color='red', width=2), 
                            symbol='d', symbolSize=4, symbolBrush='red')

class InvestmentSimulator(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Investment Simulator")
        self.setGeometry(100, 100, 1600, 800)
        
        # Initialize with default parameters
        self.params = InvestmentParams.get_defaults()
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Create control and plot panels
        self.control_panel = InvestmentControlPanel(on_parameter_change=self.update)
        self.plot_panel = InvestmentPlotPanel()
        
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
        
        return InvestmentYearsResult(params, invest(
            year=params.start_year,
            max_year=params.end_year,
            new_savings=params.get_new_savings,
            interest_rate=params.get_interest_rate,
            withdraw_rate=params.get_withdraw_rate,
            total=params.start_total,
        ))
    
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
    InvestmentSimulator.main()

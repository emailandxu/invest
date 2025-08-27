import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QLabel, QSlider, 
                            QCheckBox, QPushButton, QTextEdit)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import numpy as np
from functools import lru_cache, reduce
from read_data import get_change_rate_by_year, get_value_by_year, sp500_data, interest_data, inflation_data, inflation_rate_multiplier
from invest import invest
from utils import USD

"""Create GUI for adjusting investment parameters and plotting results"""

from dataclasses import dataclass

@dataclass
class InvestmentParams:
    """Data class representing investment simulation parameters."""
    start_year: int = 2000
    duration: int = 25
    retire_offset: int = 0
    start_total: float = USD(30)
    cost: float = USD(1.2)
    cpi: float = 0.00
    interest_rate: float = 0.00
    new_savings: float = USD(3.0)
    stock_weight: float = 1.0
    use_sp500: bool = True
    use_real_interest: bool = True
    use_real_cpi: bool = True
    adptive_withdraw_rate: bool = True

    def __hash__(self):
        """Return hash of the investment parameters for caching purposes."""
        return hash((
            self.start_year,
            self.duration,
            self.retire_offset,
            self.start_total,
            self.cost,
            self.cpi,
            self.interest_rate,
            self.use_sp500,
            self.use_real_interest,
            self.use_real_cpi,
            self.new_savings,
            self.stock_weight,
            self.adptive_withdraw_rate
        ))
    
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
    
    def sp500_interest_rate(self, year):
        """Get interest rate for a given year based on parameters."""
        return get_change_rate_by_year(sp500_data(), year, default=self.interest_rate)

    def real_interest_rate(self, year):
        """Get interest rate for a given year based on parameters."""
        return get_value_by_year(interest_data(), year, default=self.interest_rate)
    
    def get_real_inflation_rate_multiplier(self, year):
        return inflation_rate_multiplier(year, self.start_year, default=self.cpi)
    
    def get_inflation_rate_multiplier(self, year):
        """Get cumulative inflation rate multiplier for a given year."""
        if self.use_real_cpi:
            return self.get_real_inflation_rate_multiplier(year)
        else:
            return (1 + self.cpi) ** (year - self.start_year)

class StrategyBasic(InvestmentParams):
    @classmethod
    def from_params(cls, params: InvestmentParams):
        strategy = cls()
        strategy.start_year = params.start_year
        strategy.duration = params.duration
        strategy.retire_offset = params.retire_offset
        strategy.start_total = params.start_total
        strategy.cost = params.cost
        strategy.cpi = params.cpi
        strategy.interest_rate = params.interest_rate
        strategy.use_sp500 = params.use_sp500
        strategy.use_real_interest = params.use_real_interest
        strategy.use_real_cpi = params.use_real_cpi
        strategy.new_savings = params.new_savings
        strategy.stock_weight = params.stock_weight
        strategy.adptive_withdraw_rate = params.adptive_withdraw_rate
        return strategy

    def get_new_savings(self, year, total):
        """Get new savings amount for a given year."""
        if year < self.retire_year:
            return self.new_savings #* self.get_inflation_rate_multiplier(year)
        else:
            return 0

    def get_interest_rate(self, year, total):
        """Get interest rate for a given year based on parameters."""
        if self.use_sp500 and self.use_real_interest:
            stock_rate = self.sp500_interest_rate(year)
            interest_rate = self.real_interest_rate(year)
            return stock_rate * self.stock_weight + interest_rate * (1 - self.stock_weight)
        elif self.use_sp500:
            return self.sp500_interest_rate(year)
        elif self.use_real_interest:
            return self.real_interest_rate(year)
        else:
            return self.interest_rate
    
    def get_withdraw_rate(self, year, total):
        """Get withdrawal rate for a given year and total."""
        target_living_ratio = min(((self.cost+1e-16) / (max(total, 1e-16))) * self.get_inflation_rate_multiplier(year), 1.0)

        def my_log_function(x, arg_max=10):
            """When 1 < x < arg_max, the function approaching 1 is log speed, arg_max < x < inf,
            the function approaching 0 is log speed, while x < 1 is not acceptable."""
            if x > arg_max:
                return 1
            if x < 1:
                return 0
            ln_arg_max = np.log(arg_max)
            a = 1.0 / ln_arg_max
            k = np.exp(1) / ln_arg_max
            return k * np.log(x) / (x ** a)
        
        withdraw_rate = target_living_ratio

        if self.adptive_withdraw_rate:
            upper_bound = 0.0575
            center_ratio = upper_bound / 2
            if target_living_ratio > center_ratio:
                upper_bound_remains = upper_bound - center_ratio
                log_scale = my_log_function(target_living_ratio / center_ratio, arg_max=1/center_ratio) # because target_living_ratio is approching 1
                withdraw_rate = center_ratio + log_scale * upper_bound_remains
            else:
                # upper_bound = 0.04
                upper_bound_remains = upper_bound - target_living_ratio
                log_scale = my_log_function(center_ratio / target_living_ratio, arg_max=2e2*center_ratio) # because 1/target_living_ratio is approching inf
                withdraw_rate = target_living_ratio + log_scale * upper_bound_remains
        
        return withdraw_rate

class StrategyBondsPriorSell(StrategyBasic):
    @classmethod
    def from_params(cls, params: InvestmentParams):
        strategy = super().from_params(params)
        strategy.stock_total_of_year = {
            params.start_year - 1: params.start_total * params.stock_weight
        }
        strategy.bonds_total_of_year = {
            params.start_year - 1: params.start_total * (1 - params.stock_weight)
        }
        return strategy
    
    def get_stock_weight(self, year):
        try:
            return self.stock_total_of_year[year] / (self.stock_total_of_year[year] + self.bonds_total_of_year[year] + 1e-16)
        except KeyError:
            return self.stock_weight
    
    def get_interest_rate(self, year, total):
        """Get interest rate for a given year based on parameters."""
        if self.use_sp500 and self.use_real_interest:
            stock_rate = self.sp500_interest_rate(year) 
            interest_rate = self.real_interest_rate(year)

            stock_weight = self.get_stock_weight(year-1)
            self.stock_total_of_year[year] = (stock_rate+1) * total * stock_weight
            self.bonds_total_of_year[year] = (interest_rate+1) * total * (1 - stock_weight)
            
            return stock_rate * stock_weight + interest_rate * (1 - stock_weight)
        
        elif self.use_sp500:
            self.stock_total_of_year[year] = (self.sp500_interest_rate(year)+1) * total * self.get_stock_weight(year-1) 
            self.bonds_total_of_year[year] = 0
            return self.sp500_interest_rate(year)
        elif self.use_real_interest:
            self.stock_total_of_year[year] = 0
            self.bonds_total_of_year[year] = (self.real_interest_rate(year)+1) * total
            return self.real_interest_rate(year)
        else:
            self.stock_total_of_year[year] = 0
            self.bonds_total_of_year[year] = (self.interest_rate+1) * total
            return self.interest_rate
    
    def get_withdraw_rate(self, year, total):
        """Get withdrawal rate for a given year and total."""
        withdraw_rate = super().get_withdraw_rate(year, total)
        withdraw = total * withdraw_rate

        if  self.get_stock_weight(year) < self.stock_weight:
            self.bonds_total_of_year[year] -= withdraw
        else:
            self.stock_total_of_year[year] -= withdraw

        if stock_weight := self.get_stock_weight(year) > (self.stock_weight - 0.01) :
            # reblance
            bias = self.stock_total_of_year[year] * (stock_weight - (self.stock_weight - 0.01))
            self.stock_total_of_year[year] -= bias
            self.bonds_total_of_year[year] += bias


        # print(f"Year {year}: Stock weight {self.get_stock_weight(year-1):.3f} -> {self.get_stock_weight(year):.3f} (Δ{self.get_stock_weight(year) - self.get_stock_weight(year-1):+.3f}) | Stock: ${self.stock_total_of_year[year]:.0f} | Bonds: ${self.bonds_total_of_year[year]:.0f} | Withdraw: ${withdraw:.0f}")

        return withdraw_rate

    
class InvestmentYearsResult:
    def __init__(self, params, years_result):
        self.params: InvestmentParams = params
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
        principle = self.years_result[-1]['principle']
        if duration > 0 and principle > 0:
            growth_rate = (((final_total + 1e-4) / principle) ** (1 / duration) - 1)
        else:
            growth_rate = 0.0

        
        living_cost_benchmark = np.array([
            self.params.get_real_inflation_rate_multiplier(
                self.years_result[idx]['year']
            ) * self.params.cost
            for idx in range(len(self.years_result))
        ])
        
        living_cost = np.array([self.years_result[idx]['withdraw'] for idx in range(len(self.years_result))])

        return {
            'final_total': final_total,
            'final_withdraw_total': final_withdraw_total,
            'final_interest_total': final_interest_total,
            'zero_year': zero_year,
            'growth_rate': growth_rate,
            'inflation_rate': self.params.get_inflation_rate_multiplier(self.params.end_year),
            'living_cost': living_cost,
            'living_cost_benchmark': living_cost_benchmark,
            'living_cost_gap_min': np.min(living_cost / (living_cost_benchmark + 1e-4)) * self.params.cost / 12,
            'living_cost_gap_mean': np.mean(living_cost / (living_cost_benchmark + 1e-4)) * self.params.cost / 12,
            'living_cost_gap_std': np.std(living_cost / (living_cost_benchmark + 1e-4)) * self.params.cost / 12,
            'total_min': np.min(self.series('total')),
            'total_max': np.max(self.series('total')),
            'total_mean': np.mean(self.series('total')),
            'total_std': np.std(self.series('total')),
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
        text += f"  • Total Min: {metrics['total_min']:>12,.2f}$\n"
        text += f"  • Withdraw Total: {metrics['final_withdraw_total']:>12,.2f}$\n"
        text += f"  • Interest Total: {metrics['final_interest_total']:>12,.2f}$\n"
        text += f"  • CGAR:  {metrics['growth_rate']:>12.2%}\n"
        text += f"  • Inflation Rate: {metrics['inflation_rate']:>12.2%}\n"
        text += f"  • Living Cost Gap (Min):  {metrics['living_cost_gap_min']:>8.2f}$\n"
        text += f"  • Living Cost Gap (Mean): {metrics['living_cost_gap_mean']:>8.2f}$\n"
        text += f"  • Living Cost Gap (Std):  {metrics['living_cost_gap_std']:>8.2f}$\n"
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
        # results_text += self.interest_rate_summary
        results_text += self.sustainability_summary
        return results_text
    
    def __str__(self):
        return self.analysis_report()
    
    def series(self, key, func=lambda v: v):
        return [func(row[key]) for row in self.years_result]

class InvestmentControlPanel(QWidget):
    """Control panel widget containing all parameter sliders and checkboxes."""
    
    def __init__(self, plot_panel, on_parameter_change=None):
        super().__init__()
        self.plot_panel = plot_panel
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
        
        # Stock Weight
        self.stock_weight_slider = QSlider(Qt.Horizontal)
        self.stock_weight_slider.setRange(0, 100)
        self.stock_weight_slider.valueChanged.connect(self._on_change)
        self.stock_weight_label = QLabel()
        row += 1
        grid_layout.addWidget(self.stock_weight_slider, row, 0)
        grid_layout.addWidget(self.stock_weight_label, row, 1)
        row += 1

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

        # Real Data checkboxes
        real_data_widget = QWidget()
        real_data_layout = QGridLayout(real_data_widget)
        self.use_sp500_checkbox = QCheckBox(text="Standard & Poor's 500")
        self.use_sp500_checkbox.stateChanged.connect(self._on_change)
        real_data_layout.addWidget(self.use_sp500_checkbox, 0, 0)
                    
        self.use_real_interest_checkbox = QCheckBox(text="Federal Funds Rate")
        self.use_real_interest_checkbox.stateChanged.connect(self._on_change)
        real_data_layout.addWidget(self.use_real_interest_checkbox, 1, 0)
        
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
        self.stock_weight_label.setText(f"StockWeight: {self.stock_weight_slider.value() / 100:.2f}")

    def set_parameters(self, params: InvestmentParams):
        """Set all sliders and checkboxes from parameter object."""
        # Temporarily disconnect signals to avoid triggering onChange during parameter setting
        self.start_year_slider.valueChanged.disconnect()
        self.duration_slider.valueChanged.disconnect()
        self.retire_offset_slider.valueChanged.disconnect()
        self.start_total_slider.valueChanged.disconnect()
        self.cost_slider.valueChanged.disconnect()
        self.cpi_slider.valueChanged.disconnect()
        self.interest_rate_slider.valueChanged.disconnect()
        self.new_savings_slider.valueChanged.disconnect()
        self.stock_weight_slider.valueChanged.disconnect()
        self.use_sp500_checkbox.stateChanged.disconnect()
        self.use_real_interest_checkbox.stateChanged.disconnect()
        self.use_real_cpi_checkbox.stateChanged.disconnect()
        self.adptive_withdraw_rate_checkbox.stateChanged.disconnect()
        
        # Set values without triggering signals
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
        self.stock_weight_slider.setValue(int(params.stock_weight * 100))
        self.adptive_withdraw_rate_checkbox.setChecked(params.adptive_withdraw_rate)
        
        # Reconnect signals
        self.start_year_slider.valueChanged.connect(self._on_change)
        self.duration_slider.valueChanged.connect(self._on_change)
        self.retire_offset_slider.valueChanged.connect(self._on_change)
        self.start_total_slider.valueChanged.connect(self._on_change)
        self.cost_slider.valueChanged.connect(self._on_change)
        self.cpi_slider.valueChanged.connect(self._on_change)
        self.interest_rate_slider.valueChanged.connect(self._on_change)
        self.new_savings_slider.valueChanged.connect(self._on_change)
        self.stock_weight_slider.valueChanged.connect(self._on_change)
        self.use_sp500_checkbox.stateChanged.connect(self._on_change)
        self.use_real_interest_checkbox.stateChanged.connect(self._on_change)
        self.use_real_cpi_checkbox.stateChanged.connect(self._on_change)
        self.adptive_withdraw_rate_checkbox.stateChanged.connect(self._on_change)

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
        params.stock_weight = self.stock_weight_slider.value() / 100  # Scale down from cents to dollars
        params.adptive_withdraw_rate = self.adptive_withdraw_rate_checkbox.isChecked()
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
            # Bias adjustment: corrects for inflation compounding on new savings contributions
            # Each year's new savings experiences different inflation periods, so we subtract
            # the excess inflation to get an accurate baseline for comparison
            bias_benchmark_total = np.cumsum([
                years_result.series('new_savings')[idx] * (
                    years_result.params.get_real_inflation_rate_multiplier(year) - 1
                ) for idx, year in enumerate(years)
            ])
            benchmark_total = [
                years_result.series('principle')[idx] * 
                years_result.params.get_real_inflation_rate_multiplier(year) 
                for idx, year in enumerate(years)
            ]
            benchmark_total = [benchmark_total[idx] - bias_benchmark_total[idx] for idx in range(len(years))]
            self.plot_total.plot(years, benchmark_total, 
                                pen=pg.mkPen(color='gray', width=1, style=Qt.DashLine), 
                                name='Inflation Adjusted Start Total')
        
        # Plot 2: Interest Rate
        self.plot_interest_rate.plot(years, years_result.series('interest_rate', lambda v: round(v, 4)), 
                                    pen=pg.mkPen(color='green', width=2), 
                                    symbol='s', symbolSize=4, symbolBrush='green')
        
        if self.show_benchmark:
            # Add benchmark line for real interest rate
            benchmark_real_interest = [years_result.params.sp500_interest_rate(year) for year in years]
            self.plot_interest_rate.plot(years, benchmark_real_interest, 
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
            # Add benchmark line for withdrawal plot
            benchmark_withdrawals = [years_result.params.cost * years_result.params.get_real_inflation_rate_multiplier(year) for year in years]
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

class InvestmentSimulator(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Investment Simulator")
        self.setGeometry(100, 100, 900, 600)
        
        # Initialize with default parameters
        self.params = InvestmentParams.get_defaults()
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Create control and plot panels
        self.plot_panel = InvestmentPlotPanel()
        self.control_panel = InvestmentControlPanel(self.plot_panel, on_parameter_change=self.update)
        
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
        return InvestmentYearsResult(params, invest(
            year=params.start_year,
            max_year=params.end_year,
            new_savings=strategy.get_new_savings,
            interest_rate=strategy.get_interest_rate,
            withdraw_rate=strategy.get_withdraw_rate,
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
    # import cProfile
    # import pstats
    # import io
    
    # # Create profiler
    # profiler = cProfile.Profile()
    # profiler.enable()
    
    # try:
    #     InvestmentSimulator.main()
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

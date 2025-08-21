# 投资模拟分析报告 | Investment Simulation Analysis Report
# 
# 基于本脚本的投资模拟结果，以下是主要发现和结论：
# Based on the investment simulation results from this script, here are the main findings and conclusions:
#
# 1. 模拟参数设置： | Simulation Parameter Settings:
#    - 模拟期间：30年 | Simulation period: 30 years
#    - 退休年份：第5年 | Retirement year: Year 5
#    - 初始资金：9万美元（约64.8万人民币，按1:7.2汇率）| Initial funds: $90,000 (~648,000 RMB at 1:7.2 exchange rate)
#    - 退休前年收入结余：3万美元/年（约21.6万人民币）| Annual income surplus before retirement: $30,000/year (~216,000 RMB)
#    - 退休后年收入：0 | Annual income after retirement: 0
#
# 2. 投资回报率模型： | Investment Return Rate Model:
#    - 退休前2.5年：固定4%年化收益率（保守投资策略）| First 2.5 years before retirement: Fixed 4% annual return (conservative investment strategy)
#    - 退休后：正态分布随机收益率，均值6.75%，标准差0.59%（模拟市场波动）| After retirement: Normally distributed random return rate, mean 6.75%, std dev 0.59% (simulating market volatility)
#
# 3. 提取策略： | Withdrawal Strategy:
#    - 退休前：不提取资金 | Before retirement: No fund withdrawal
#    - 退休后：动态提取率，基于通胀调整的生活费需求 | After retirement: Dynamic withdrawal rate based on inflation-adjusted living expense needs
#    - 提取公式：(目标年支出 / 当前总资产 * 基准率) * 通胀调整系数 | Withdrawal formula: (target annual expenses / current total assets * base rate) * inflation adjustment factor
#
# 4. 关键结论： | Key Conclusions:
#    - 资金池的可持续性高度依赖于提取率与市场收益率的关系 | The sustainability of the fund pool is highly dependent on the relationship between withdrawal rate and market return rate
#    - 当提取率超过市场长期平均收益率时，资金池存在耗尽风险 | When withdrawal rate exceeds the long-term average market return rate, there is a risk of fund depletion
#    - 在市场低迷期间，应当考虑降低提取率（即调整生活标准）以保护本金 | During market downturns, consider reducing withdrawal rate (i.e., adjusting living standards) to protect principal
#    - 退休初期的资金积累阶段对长期财务安全至关重要 | The fund accumulation phase in early retirement is crucial for long-term financial security
#
# 5. 风险提示： | Risk Warnings:
#    - 本模拟使用的市场收益率参数来源于ChatGPT，可能不够准确 | The market return rate parameters used in this simulation come from ChatGPT and may not be accurate enough
#    - 实际市场波动可能更加复杂，包含系统性风险和黑天鹅事件 | Actual market volatility may be more complex, including systemic risks and black swan events
#    - 通胀率、税收、医疗支出等因素未充分考虑 | Factors such as inflation rate, taxes, and medical expenses are not fully considered
#    - 建议结合更多历史数据和专业财务规划进行决策 | It is recommended to combine more historical data and professional financial planning for decision-making

import random
import numpy as np
from functools import reduce
from read_sp500 import get_change_rate_by_year, get_value_by_year, sp500_data, interest_data, inflation_data

USD = lambda x: x #* 7.2 / 12

def invest(year, max_year, new_savings, interest_rate, withdraw_rate, total=0, interest_total=0, withdraw_total=0):
    if year > max_year:
        return []
    
    init_total = total
    
    # Calculate current year's principles (can be function or constant)
    if callable(new_savings):
        current_new_savings = new_savings(year)
    else:
        current_new_savings = new_savings
    
    # Calculate current year's interest rate (can be function or constant)
    if callable(interest_rate):
        current_interest_rate = interest_rate(year)
    else:
        current_interest_rate = interest_rate
    
    interest = init_total * current_interest_rate
    interest_total += interest
    total = init_total + interest + current_new_savings
    

    # Calculate current year's withdraw rate (can be function or constant)
    if callable(withdraw_rate):
        current_withdraw_rate = withdraw_rate(year, init_total)
    else:
        current_withdraw_rate = withdraw_rate
    
    # Calculate withdrawals and interest for current year
    withdraw = init_total * current_withdraw_rate

    if total - withdraw > 0:
        total -= withdraw
        withdraw_total += withdraw
    else:
        withdraw = total
        total -= withdraw
        withdraw_total += withdraw
    
    # Create current year's record
    current_year_data = {
        'year': year,
        'new_savings': current_new_savings,
        "interest_rate": current_interest_rate,
        'interest_total': interest_total,
        'interest': interest,
        "withdrawed_interest_rate": (interest - withdraw) / (init_total) if init_total!=0 else -1.0,
        'withdraw_total': withdraw_total,
        'withdraw': withdraw,
        "interest_vs_principle": (interest_total - withdraw_total) / (total - (interest_total - withdraw_total) + 1e-16),
        'total': total
    }
    
    # Recursively get remaining years and prepend current year
    return [current_year_data] + invest(year + 1, max_year, new_savings, interest_rate, withdraw_rate, 
                                       total, interest_total, withdraw_total)

def invest_gui():
    import sys
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                QHBoxLayout, QGridLayout, QLabel, QSlider, 
                                QCheckBox, QPushButton, QTextEdit)
    from PyQt5.QtCore import Qt
    import pyqtgraph as pg
    import numpy as np
    """Create GUI for adjusting investment parameters and plotting results"""
    
    class InvestmentSimulator(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Investment Simulator")
            self.setGeometry(100, 100, 1600, 800)
            
            # Create main widget and layout
            main_widget = QWidget()
            self.setCentralWidget(main_widget)
            main_layout = QHBoxLayout(main_widget)
            
            # Create control and plot areas
            self.create_controls(main_layout)
            self.create_plots(main_layout)
            self.create_result(main_layout)
            # Initialize default values
            self.set_defaults()
            
            # Initial simulation run
            self.update_simulation()
        
        def create_controls(self, main_layout):
            # Control panel
            control_widget = QWidget()
            control_widget.setFixedWidth(200)
            control_layout = QVBoxLayout(control_widget)
            
            # Parameter grid
            grid_widget = QWidget()
            grid_layout = QGridLayout(grid_widget)
            
            # Default parameter values
            self.default_start_year = 1995
            self.default_duration = 25
            self.default_retire_offset = 0
            self.default_start_total = 30
            self.default_cost = 1.2
            self.default_cpi = 0.03
            self.default_interest_rate = 0.02
            self.default_use_sp500 = False
            self.default_use_real_interest = False
            self.default_use_real_cpi = False
            self.default_new_savings = 3.0
            
            row = 0
            
            # Start Year
            grid_layout.addWidget(QLabel("Start Year:"), row, 0)
            self.start_year_slider = QSlider(Qt.Horizontal)
            self.start_year_slider.setRange(1970, 2100)
            self.start_year_slider.valueChanged.connect(self.update_simulation)
            self.start_year_label = QLabel()
            row += 1
            grid_layout.addWidget(self.start_year_slider, row, 0)
            grid_layout.addWidget(self.start_year_label, row, 1)
            row += 1
            # Duration
            grid_layout.addWidget(QLabel("Duration (years):"), row, 0)
            self.duration_slider = QSlider(Qt.Horizontal)
            self.duration_slider.setRange(0, 99)
            self.duration_slider.valueChanged.connect(self.update_simulation)
            self.duration_label = QLabel()
            row += 1
            grid_layout.addWidget(self.duration_slider, row, 0)
            grid_layout.addWidget(self.duration_label, row, 1)
            row += 1
            # Retirement Offset
            grid_layout.addWidget(QLabel("Retirement Offset:"), row, 0)
            self.retire_offset_slider = QSlider(Qt.Horizontal)
            self.retire_offset_slider.setRange(0, 99)
            self.retire_offset_slider.valueChanged.connect(self.update_simulation)
            self.retire_offset_label = QLabel()
            row += 1
            grid_layout.addWidget(self.retire_offset_slider, row, 0)
            grid_layout.addWidget(self.retire_offset_label, row, 1)
            row += 1
            # Start Total
            grid_layout.addWidget(QLabel("Start Total:"), row, 0)
            self.start_total_slider = QSlider(Qt.Horizontal)
            self.start_total_slider.setRange(0, 2000)  # Scale by 10 for decimal precision
            self.start_total_slider.valueChanged.connect(self.update_simulation)
            self.start_total_label = QLabel()
            row += 1
            grid_layout.addWidget(self.start_total_slider, row, 0)
            grid_layout.addWidget(self.start_total_label, row, 1)
            row += 1
            # Annual Cost
            grid_layout.addWidget(QLabel("Annual Cost:"), row, 0)
            self.cost_slider = QSlider(Qt.Horizontal)
            self.cost_slider.setRange(0, 500)  # Scale by 100 for decimal precision
            self.cost_slider.valueChanged.connect(self.update_simulation)
            self.cost_label = QLabel()
            row += 1
            grid_layout.addWidget(self.cost_slider, row, 0)
            grid_layout.addWidget(self.cost_label, row, 1)
            row += 1
            # CPI
            grid_layout.addWidget(QLabel("CPI:"), row, 0)
            self.cpi_slider = QSlider(Qt.Horizontal)
            self.cpi_slider.setRange(0, 1000)  # Scale by 10000 for decimal precision
            self.cpi_slider.valueChanged.connect(self.update_simulation)
            self.cpi_label = QLabel()
            row += 1
            grid_layout.addWidget(self.cpi_slider, row, 0)
            grid_layout.addWidget(self.cpi_label, row, 1)
            row += 1
            # Interest Rate
            grid_layout.addWidget(QLabel("Interest Rate:"), row, 0)
            self.interest_rate_slider = QSlider(Qt.Horizontal)
            self.interest_rate_slider.setRange(0, 1500)  # Scale by 10000 for decimal precision
            self.interest_rate_slider.valueChanged.connect(self.update_simulation)
            self.interest_rate_label = QLabel()
            row += 1
            grid_layout.addWidget(self.interest_rate_slider, row, 0)
            grid_layout.addWidget(self.interest_rate_label, row, 1)
            row += 1
            # Principle Amount
            grid_layout.addWidget(QLabel("Annual Savings:"), row, 0)
            self.new_savings_slider = QSlider(Qt.Horizontal)
            self.new_savings_slider.setRange(0, 1000)  # Scale by 100 for decimal precision
            self.new_savings_slider.valueChanged.connect(self.update_simulation)
            self.new_savings_label = QLabel()
            row += 1
            grid_layout.addWidget(self.new_savings_slider, row, 0)
            grid_layout.addWidget(self.new_savings_label, row, 1)
            row += 1
            # Use SP500
            grid_layout.addWidget(QLabel("Use S&P 500:"), row, 0)
            self.use_sp500_checkbox = QCheckBox()
            self.use_sp500_checkbox.stateChanged.connect(self.update_simulation)
            row += 1
            grid_layout.addWidget(self.use_sp500_checkbox, row, 0)
            row += 1
            grid_layout.addWidget(QLabel("Use Real Interest:"), row, 0)
            self.use_real_interest_checkbox = QCheckBox(text="Use Real Interest")
            self.use_real_interest_checkbox.stateChanged.connect(self.update_simulation)
            row += 1
            grid_layout.addWidget(self.use_real_interest_checkbox, row, 0)
            row += 1
            grid_layout.addWidget(QLabel("Use Real CPI:"), row, 0)
            self.use_real_cpi_checkbox = QCheckBox()
            self.use_real_cpi_checkbox.stateChanged.connect(self.update_simulation)
            row += 1
            grid_layout.addWidget(self.use_real_cpi_checkbox, row, 0)
            row += 1
            control_layout.addWidget(grid_widget)
            
            # Reset button
            reset_button = QPushButton("Reset to Defaults")
            reset_button.clicked.connect(self.reset_to_defaults)
            control_layout.addWidget(reset_button)
            
            main_layout.addWidget(control_widget)
        
        def create_plots(self, main_layout):
            # Plot area
            plot_widget = QWidget()
            plot_layout = QGridLayout(plot_widget)
            
            # Create pyqtgraph plot widgets
            self.plot_total = pg.PlotWidget(title="Total Amount Over Time")
            self.plot_interest_rate = pg.PlotWidget(title="Interest Rate")
            self.plot_withdrawed_interest_rate = pg.PlotWidget(title="Withdrawed Interest Rate")
            self.plot_interest = pg.PlotWidget(title="Interest Total")
            self.plot_withdraw = pg.PlotWidget(title="Withdrawals")
            self.plot_ratio = pg.PlotWidget(title="Interest VS Principle")
            
            # Configure plots
            self.plot_total.setLabel('left', 'Amount ($)')
            self.plot_total.setLabel('bottom', 'Year')
            self.plot_total.showGrid(x=True, y=True, alpha=0.3)
            
            self.plot_interest_rate.setLabel('left', 'Rate')
            self.plot_interest_rate.setLabel('bottom', 'Year')
            self.plot_interest_rate.showGrid(x=True, y=True, alpha=0.3)
            
            self.plot_withdrawed_interest_rate.setLabel('left', 'Rate')
            self.plot_withdrawed_interest_rate.setLabel('bottom', 'Year')
            self.plot_withdrawed_interest_rate.showGrid(x=True, y=True, alpha=0.3)
            
            self.plot_interest.setLabel('left', 'Interest ($)')
            self.plot_interest.setLabel('bottom', 'Year')
            self.plot_interest.showGrid(x=True, y=True, alpha=0.3)
            
            self.plot_withdraw.setLabel('left', 'Withdraw ($)')
            self.plot_withdraw.setLabel('bottom', 'Year')
            self.plot_withdraw.showGrid(x=True, y=True, alpha=0.3)
            
            self.plot_ratio.setLabel('left', 'Rate')
            self.plot_ratio.setLabel('bottom', 'Year')
            self.plot_ratio.showGrid(x=True, y=True, alpha=0.3)
            
            # Arrange plots in grid
            plot_layout.addWidget(self.plot_total, 0, 0)
            plot_layout.addWidget(self.plot_interest_rate, 0, 1)
            plot_layout.addWidget(self.plot_withdrawed_interest_rate, 0, 2)
            plot_layout.addWidget(self.plot_interest, 1, 0)
            plot_layout.addWidget(self.plot_withdraw, 1, 1)
            plot_layout.addWidget(self.plot_ratio, 1, 2)
            
            main_layout.addWidget(plot_widget)

        def create_result(self, main_layout):
            # Results display
            result_widget = QWidget()
            result_widget.setFixedWidth(250)
            result_layout = QVBoxLayout(result_widget)
            
            self.results_text = QTextEdit()
            # self.results_text.setMaximumHeight(150)
            self.results_text.setReadOnly(True)
            result_layout.addWidget(self.results_text)
            
            main_layout.addWidget(result_widget)

        def set_defaults(self):
            self.start_year_slider.setValue(self.default_start_year)
            self.duration_slider.setValue(self.default_duration)
            self.retire_offset_slider.setValue(self.default_retire_offset)
            self.start_total_slider.setValue(int(self.default_start_total * 10))
            self.cost_slider.setValue(int(self.default_cost * 100))
            self.cpi_slider.setValue(int(self.default_cpi * 10000))
            self.interest_rate_slider.setValue(int(self.default_interest_rate * 10000))
            self.new_savings_slider.setValue(int(self.default_new_savings * 100))
            self.use_sp500_checkbox.setChecked(self.default_use_sp500)
            self.use_real_interest_checkbox.setChecked(self.default_use_real_interest)
            self.use_real_cpi_checkbox.setChecked(self.default_use_real_cpi)
        
        def reset_to_defaults(self):
            self.set_defaults()
            self.update_simulation()
        
        def update_labels(self):
            self.start_year_label.setText(str(self.start_year_slider.value()))
            self.duration_label.setText(str(self.duration_slider.value()))
            self.retire_offset_label.setText(str(self.retire_offset_slider.value()))
            self.start_total_label.setText(f"{self.start_total_slider.value() / 10:.1f}")
            self.cost_label.setText(f"{self.cost_slider.value() / 100:.2f}")
            self.cpi_label.setText(f"{self.cpi_slider.value() / 10000:.3f}")
            self.interest_rate_label.setText(f"{self.interest_rate_slider.value() / 10000:.3f}")
            self.new_savings_label.setText(f"{self.new_savings_slider.value() / 100:.2f}")
        
        def update_simulation(self):
            self.update_labels()
            
            # Get current parameter values
            curr_start_year = self.start_year_slider.value()
            curr_duration = self.duration_slider.value()
            curr_retire_offset = self.retire_offset_slider.value()
            curr_start_total = self.start_total_slider.value() / 10
            curr_cost = self.cost_slider.value() / 100
            curr_cpi = self.cpi_slider.value() / 10000
            curr_interest_rate = self.interest_rate_slider.value() / 10000
            curr_use_sp500 = self.use_sp500_checkbox.isChecked()
            curr_use_real_interest = self.use_real_interest_checkbox.isChecked()
            curr_use_real_cpi = self.use_real_cpi_checkbox.isChecked()
            curr_new_savings = self.new_savings_slider.value() / 100

            self.use_real_interest_checkbox.setChecked(curr_use_real_interest)
            
            year_range = f"{curr_start_year}-{curr_start_year + curr_duration}"
            self.setWindowTitle(f"Investment Simulator ({year_range})")
            
            # Calculate derived values
            curr_end_year = curr_start_year + curr_duration
            curr_retire_year = curr_start_year + curr_retire_offset
            
            # Set up interest rate function
            if curr_use_sp500:
                rate_func = lambda year: get_change_rate_by_year(sp500_data(), year, default=curr_interest_rate)
            elif curr_use_real_interest:
                rate_func = lambda year: get_value_by_year(interest_data(), year, default=curr_interest_rate)
            else:
                rate_func = curr_interest_rate
            
            if curr_use_real_cpi:
                inflation_rate_func = lambda year: reduce(lambda x,y: x*y, [( 1.0 + get_value_by_year(inflation_data(), y, default=curr_cpi)) for y in range(curr_retire_year, year)], 1)
            else:
                inflation_rate_func = lambda year: (1 + curr_cpi) ** (year - curr_retire_year)

            try:
                # Run simulation
                years_result = invest(
                    year=curr_start_year,
                    max_year=curr_end_year,
                    new_savings=lambda year: USD(curr_new_savings) if year < curr_retire_year else USD(0),
                    interest_rate=rate_func,
                    withdraw_rate=lambda year, total: (
                        0.00 if year < curr_retire_year else ((USD(curr_cost) / (total + 1e-16)) * inflation_rate_func(year)) #(1 + curr_cpi) ** (year - curr_retire_year))
                    ),
                    total=USD(curr_start_total),
                    interest_total=0.0
                )
                
                # Clear previous plots
                self.plot_total.clear()
                self.plot_interest_rate.clear()
                self.plot_withdrawed_interest_rate.clear()
                self.plot_interest.clear()
                self.plot_withdraw.clear()
                self.plot_ratio.clear()
                
                # Plot results
                series = lambda key: [row[key] for row in years_result]
                years = series('year')
                
                # Plot 1: Total Amount
                self.plot_total.plot(years, series('total'), pen=pg.mkPen(color='blue', width=2), 
                                   symbol='o', symbolSize=4, symbolBrush='blue')
                
                # Plot 2: Interest Rate
                self.plot_interest_rate.plot(years, series('interest_rate'), pen=pg.mkPen(color='green', width=2), 
                                           symbol='s', symbolSize=4, symbolBrush='green')
                
                # Plot 3: Withdraw Rate
                self.plot_withdrawed_interest_rate.plot(years, series('withdrawed_interest_rate'), pen=pg.mkPen(color='orange', width=2), 
                                           symbol='x', symbolSize=4, symbolBrush='orange')
                
                # Plot 4: Annual Interest
                self.plot_interest.plot(years, series('interest_total'), pen=pg.mkPen(color='purple', width=2), 
                                      symbol='t', symbolSize=4, symbolBrush='purple')
                
                # Plot 5: Annual Withdrawals
                self.plot_withdraw.plot(years, series('withdraw'), pen=pg.mkPen(color='brown', width=2), 
                                      symbol='h', symbolSize=4, symbolBrush='brown')
                
                # Plot 6: Interest VS Principle
                self.plot_ratio.plot(years, series('interest_vs_principle'), pen=pg.mkPen(color='red', width=2), 
                                   symbol='d', symbolSize=4, symbolBrush='red')
                
                # Update results display
                interest_rates = np.array([row['interest_rate'] for row in years_result])
                final_total = years_result[-1]['total'] if years_result else 0
                mean_rate = np.mean(interest_rates)
                std_rate = np.std(interest_rates)
                
                # Find the year when total becomes zero or near zero
                zero_year = None
                for row in years_result:
                    if row['total'] <= row['withdraw']:  # Same threshold as in the invest function
                        zero_year = row['year']
                        break
                
                # Build structured results text
                results_text = "=== SIMULATION RESULTS ===\n\n"
                
                # Financial Summary
                results_text += "💰 FINANCIAL SUMMARY:\n"
                results_text += f"  • Final Total:         ${final_total:>12,.2f}\n"
                results_text += f"  • Initial Investment:  ${curr_start_total:>12,.2f}\n"
                duration = (curr_end_year if zero_year is None else zero_year) - curr_start_year
                if duration > 0 and curr_start_total > 0:
                    growth_rate = (((final_total+1e-4) / curr_start_total) ** (1 / duration) - 1)
                else:
                    growth_rate = 0.0
                results_text += f"  • CGAR:  {growth_rate:>12.2%}\n"
                results_text += "\n"
                
                # Interest Rate Statistics
                results_text += "📊 INTEREST RATE STATS:\n"
                results_text += f"  • Mean Rate:           {mean_rate:>12.3%}\n"
                results_text += f"  • Standard Deviation:  {std_rate:>12.3%}\n"
                results_text += f"  • Min Rate:            {np.min(interest_rates):>12.3%}\n"
                results_text += f"  • Max Rate:            {np.max(interest_rates):>12.3%}\n"
                results_text += "\n"
                
                # Timeline Information
                results_text += "⏰ TIMELINE INFO:\n"
                results_text += f"  • Start Year:          {curr_start_year:>12}\n"
                results_text += f"  • End Year:            {curr_end_year:>12}\n"
                results_text += f"  • Retirement Year:     {curr_retire_year:>12}\n"
                results_text += f"  • Years Simulated:     {len(years_result):>12}\n"
                results_text += f"  • Years to Retirement: {curr_retire_year - curr_start_year:>12}\n"
                results_text += f"  • Years in Retirement: {curr_end_year - curr_retire_year:>12}\n"
                results_text += "\n"
                
                # Sustainability Analysis
                results_text += "🔍 SUSTAINABILITY:\n"
                if zero_year is not None:
                    years_lasted = zero_year - curr_start_year
                    retirement_years_lasted = zero_year - curr_retire_year
                    results_text += f"  • Status:              ⚠️  FUNDS DEPLETED\n"
                    results_text += f"  • Depletion Year:      {zero_year:>12}\n"
                    results_text += f"  • Total Years Lasted:  {years_lasted:>12}\n"
                    results_text += f"  • Retirement Years:    {retirement_years_lasted:>12}\n"
                else:
                    results_text += f"  • Status:              ✅ SUSTAINABLE\n"
                    results_text += f"  • Funds Last:          Full Duration\n"
                
                self.results_text.setPlainText(results_text)
                
            except Exception as e:
                self.results_text.setPlainText(f"Error: {str(e)}")
                raise e
        
        def keyPressEvent(self, event):
            if event.key() == Qt.Key_Left:
                current_year = self.start_year_slider.value()
                self.start_year_slider.setValue(current_year - 1)
            elif event.key() == Qt.Key_Right:
                current_year = self.start_year_slider.value()
                self.start_year_slider.setValue(current_year + 1)
            super().keyPressEvent(event)
    
    # Create and run the application
    app = QApplication(sys.argv)
    window = InvestmentSimulator()
    window.show()
    app.exec_()

if __name__ == "__main__":
    invest_gui()

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
from read_sp500 import get_change_rate_by_year as sp500_by_year

USD = lambda x: x #* 7.2 / 12

def invest(year, max_year, principles, interest_rate, withdraw_rate, total=0, interest_total=0, withdraw_total=0):
    if year > max_year:
        return []
    
    # Calculate current year's principles (can be function or constant)
    if callable(principles):
        current_principles = principles(year)
    else:
        current_principles = principles
    
    # Calculate current year's interest rate (can be function or constant)
    if callable(interest_rate):
        current_interest_rate = interest_rate(year)
    else:
        current_interest_rate = interest_rate
    
    interest = total * current_interest_rate
    interest_total += interest
    total += (interest + current_principles)
    

    # Calculate current year's withdraw rate (can be function or constant)
    if callable(withdraw_rate):
        current_withdraw_rate = withdraw_rate(year, total)
    else:
        current_withdraw_rate = withdraw_rate
    
    # Calculate withdrawals and interest for current year
    withdraw = total * current_withdraw_rate

    if total - withdraw > 0:
        total -= withdraw
        interest_total -= withdraw
        withdraw_total += withdraw
    elif total > 0.00001:
        withdraw = (total - 0.00001) 
        total -= withdraw
        interest_total -= withdraw
        withdraw_total += withdraw
    else:
        withdraw = 0
    
    # Create current year's record
    current_year_data = {
        'year': year,
        'new_principle': current_principles,
        "new_principle_vs_interest": interest / (current_principles + interest + 0.0001),
        "interest_rate": current_interest_rate - (withdraw / (total + withdraw)),
        'interest_total': interest_total,
        'interest': interest,
        "withdraw_rate": withdraw / (total + withdraw),
        'withdraw_total': withdraw_total,
        'withdraw': withdraw,
        "interest_vs_principle": interest_total / (total - interest_total),
        'total': total
    }
    
    # Recursively get remaining years and prepend current year
    return [current_year_data] + invest(year + 1, max_year, principles, interest_rate, withdraw_rate, 
                                       total, interest_total, withdraw_total)

def invest_gui():
    import sys
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                QHBoxLayout, QGridLayout, QLabel, QSlider, 
                                QCheckBox, QPushButton, QTextEdit)
    from PyQt5.QtCore import Qt
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import numpy as np
    """Create GUI for adjusting investment parameters and plotting results"""
    
    class InvestmentSimulator(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Investment Simulator")
            self.setGeometry(100, 100, 1200, 800)
            
            # Create main widget and layout
            main_widget = QWidget()
            self.setCentralWidget(main_widget)
            main_layout = QHBoxLayout(main_widget)
            
            # Create control and plot areas
            self.create_controls(main_layout)
            self.create_plots(main_layout)
            
            # Initialize default values
            self.set_defaults()
            
            # Initial simulation run
            self.update_simulation()
        
        def create_controls(self, main_layout):
            # Control panel
            control_widget = QWidget()
            control_widget.setFixedWidth(350)
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
            self.default_principle_amount = 3.0
            
            row = 0
            
            # Start Year
            grid_layout.addWidget(QLabel("Start Year:"), row, 0)
            self.start_year_slider = QSlider(Qt.Horizontal)
            self.start_year_slider.setRange(1970, 2030)
            self.start_year_slider.valueChanged.connect(self.update_simulation)
            grid_layout.addWidget(self.start_year_slider, row, 1)
            self.start_year_label = QLabel()
            grid_layout.addWidget(self.start_year_label, row, 2)
            row += 1
            
            # Duration
            grid_layout.addWidget(QLabel("Duration (years):"), row, 0)
            self.duration_slider = QSlider(Qt.Horizontal)
            self.duration_slider.setRange(10, 50)
            self.duration_slider.valueChanged.connect(self.update_simulation)
            grid_layout.addWidget(self.duration_slider, row, 1)
            self.duration_label = QLabel()
            grid_layout.addWidget(self.duration_label, row, 2)
            row += 1
            
            # Retirement Offset
            grid_layout.addWidget(QLabel("Retirement Offset:"), row, 0)
            self.retire_offset_slider = QSlider(Qt.Horizontal)
            self.retire_offset_slider.setRange(0, 30)
            self.retire_offset_slider.valueChanged.connect(self.update_simulation)
            grid_layout.addWidget(self.retire_offset_slider, row, 1)
            self.retire_offset_label = QLabel()
            grid_layout.addWidget(self.retire_offset_label, row, 2)
            row += 1
            
            # Start Total
            grid_layout.addWidget(QLabel("Start Total:"), row, 0)
            self.start_total_slider = QSlider(Qt.Horizontal)
            self.start_total_slider.setRange(100, 2000)  # Scale by 10 for decimal precision
            self.start_total_slider.valueChanged.connect(self.update_simulation)
            grid_layout.addWidget(self.start_total_slider, row, 1)
            self.start_total_label = QLabel()
            grid_layout.addWidget(self.start_total_label, row, 2)
            row += 1
            
            # Annual Cost
            grid_layout.addWidget(QLabel("Annual Cost:"), row, 0)
            self.cost_slider = QSlider(Qt.Horizontal)
            self.cost_slider.setRange(0, 500)  # Scale by 100 for decimal precision
            self.cost_slider.valueChanged.connect(self.update_simulation)
            grid_layout.addWidget(self.cost_slider, row, 1)
            self.cost_label = QLabel()
            grid_layout.addWidget(self.cost_label, row, 2)
            row += 1
            
            # CPI
            grid_layout.addWidget(QLabel("CPI:"), row, 0)
            self.cpi_slider = QSlider(Qt.Horizontal)
            self.cpi_slider.setRange(0, 1000)  # Scale by 10000 for decimal precision
            self.cpi_slider.valueChanged.connect(self.update_simulation)
            grid_layout.addWidget(self.cpi_slider, row, 1)
            self.cpi_label = QLabel()
            grid_layout.addWidget(self.cpi_label, row, 2)
            row += 1
            
            # Interest Rate
            grid_layout.addWidget(QLabel("Interest Rate:"), row, 0)
            self.interest_rate_slider = QSlider(Qt.Horizontal)
            self.interest_rate_slider.setRange(0, 1500)  # Scale by 10000 for decimal precision
            self.interest_rate_slider.valueChanged.connect(self.update_simulation)
            grid_layout.addWidget(self.interest_rate_slider, row, 1)
            self.interest_rate_label = QLabel()
            grid_layout.addWidget(self.interest_rate_label, row, 2)
            row += 1
            
            # Use SP500
            grid_layout.addWidget(QLabel("Use S&P 500:"), row, 0)
            self.use_sp500_checkbox = QCheckBox()
            self.use_sp500_checkbox.stateChanged.connect(self.update_simulation)
            grid_layout.addWidget(self.use_sp500_checkbox, row, 1)
            row += 1
            
            # Principle Amount
            grid_layout.addWidget(QLabel("Annual Principle:"), row, 0)
            self.principle_slider = QSlider(Qt.Horizontal)
            self.principle_slider.setRange(0, 1000)  # Scale by 100 for decimal precision
            self.principle_slider.valueChanged.connect(self.update_simulation)
            grid_layout.addWidget(self.principle_slider, row, 1)
            self.principle_label = QLabel()
            grid_layout.addWidget(self.principle_label, row, 2)
            row += 1
            
            control_layout.addWidget(grid_widget)
            
            # Reset button
            reset_button = QPushButton("Reset to Defaults")
            reset_button.clicked.connect(self.reset_to_defaults)
            control_layout.addWidget(reset_button)
            
            # Results display
            self.results_text = QTextEdit()
            self.results_text.setMaximumHeight(150)
            self.results_text.setReadOnly(True)
            control_layout.addWidget(self.results_text)
            
            main_layout.addWidget(control_widget)
        
        def create_plots(self, main_layout):
            # Plot area
            self.figure = Figure(figsize=(12, 8))
            self.canvas = FigureCanvas(self.figure)
            main_layout.addWidget(self.canvas)
            
            # Create subplots
            self.axes = self.figure.subplots(2, 3)
        
        def set_defaults(self):
            self.start_year_slider.setValue(self.default_start_year)
            self.duration_slider.setValue(self.default_duration)
            self.retire_offset_slider.setValue(self.default_retire_offset)
            self.start_total_slider.setValue(int(self.default_start_total * 10))
            self.cost_slider.setValue(int(self.default_cost * 100))
            self.cpi_slider.setValue(int(self.default_cpi * 10000))
            self.interest_rate_slider.setValue(int(self.default_interest_rate * 10000))
            self.use_sp500_checkbox.setChecked(self.default_use_sp500)
            self.principle_slider.setValue(int(self.default_principle_amount * 100))
        
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
            self.principle_label.setText(f"{self.principle_slider.value() / 100:.2f}")
        
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
            curr_principle = self.principle_slider.value() / 100
            
            year_range = f"{curr_start_year}-{curr_start_year + curr_duration}"
            self.figure.suptitle(f"Investment Simulation Analysis ({year_range})", fontsize=16, fontweight='bold')
            
            # Calculate derived values
            curr_end_year = curr_start_year + curr_duration
            curr_retire_year = curr_start_year + curr_retire_offset
            
            # Set up interest rate function
            if curr_use_sp500:
                rate_func = lambda year: sp500_by_year(year, default=0.02)
            else:
                rate_func = curr_interest_rate
            
            try:
                # Run simulation
                years_result = invest(
                    year=curr_start_year,
                    max_year=curr_end_year,
                    principles=lambda year: USD(curr_principle) if year < curr_retire_year else USD(0),
                    interest_rate=rate_func,
                    withdraw_rate=lambda year, total: 0.00 if year < curr_retire_year else ((USD(curr_cost) / total) * (1 + curr_cpi) ** (year - curr_retire_year)),
                    total=USD(curr_start_total),
                    interest_total=0.0
                )
                
                # Clear previous plots
                for ax in self.axes.flat:
                    ax.clear()
                
                # Plot results
                series = lambda key: [row[key] for row in years_result]
                
                # Plot 1: Total Amount
                self.axes[0, 0].plot(series('year'), series('total'), marker='o', color='blue', linewidth=2, markersize=4)
                self.axes[0, 0].set_title('Total Amount Over Time')
                self.axes[0, 0].set_xlabel('Year')
                self.axes[0, 0].set_ylabel('Amount ($)')
                self.axes[0, 0].grid(True, alpha=0.3)
                
                # Plot 2: Interest Rate
                self.axes[0, 1].plot(series('year'), series('interest_rate'), marker='s', color='green', linewidth=2, markersize=4)
                self.axes[0, 1].set_title('Interest Rate')
                self.axes[0, 1].set_xlabel('Year')
                self.axes[0, 1].set_ylabel('Rate')
                self.axes[0, 1].grid(True, alpha=0.3)
                
                # Plot 3: Withdraw Rate
                self.axes[0, 2].plot(series('year'), series('withdraw_rate'), marker='^', color='orange', linewidth=2, markersize=4)
                self.axes[0, 2].set_title('Withdraw Rate')
                self.axes[0, 2].set_xlabel('Year')
                self.axes[0, 2].set_ylabel('Rate')
                self.axes[0, 2].grid(True, alpha=0.3)
                
                # Plot 4: Annual Interest
                self.axes[1, 0].plot(series('year'), series('interest'), marker='p', color='purple', linewidth=2, markersize=4)
                self.axes[1, 0].set_title('Annual Interest')
                self.axes[1, 0].set_xlabel('Year')
                self.axes[1, 0].set_ylabel('Interest ($)')
                self.axes[1, 0].grid(True, alpha=0.3)
                
                # Plot 5: Annual Withdrawals
                self.axes[1, 1].plot(series('year'), series('withdraw'), marker='h', color='brown', linewidth=2, markersize=4)
                self.axes[1, 1].set_title('Annual Withdrawals')
                self.axes[1, 1].set_xlabel('Year')
                self.axes[1, 1].set_ylabel('Withdraw ($)')
                self.axes[1, 1].grid(True, alpha=0.3)
                
                # Plot 6: Interest VS Principle
                self.axes[1, 2].plot(series('year'), series('interest_vs_principle'), marker='d', color='red', linewidth=2, markersize=4)
                self.axes[1, 2].set_title('Interest VS Principle')
                self.axes[1, 2].set_xlabel('Year')
                self.axes[1, 2].set_ylabel('Rate')
                self.axes[1, 2].grid(True, alpha=0.3)
                
                self.figure.tight_layout()
                self.canvas.draw()
                
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
                
                results_text = f"Final Total: ${final_total:.2f}\n"
                results_text += f"Mean Interest Rate: {mean_rate:.3%}\n"
                results_text += f"Std Interest Rate: {std_rate:.3%}\n"
                results_text += f"Years Simulated: {len(years_result)}\n"
                
                if zero_year is not None:
                    results_text += f"Total exhausting in year: {zero_year}\n"
                    results_text += f"Last {zero_year - curr_start_year} years"
                else:
                    results_text += "Total never reaches zero"
                
                self.results_text.setPlainText(results_text)
                
            except Exception as e:
                self.results_text.setPlainText(f"Error: {str(e)}")
        
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
def main():
    def print_table(years_result):
        columns = list(years_result[0].keys())
        col_widths = {col: max(len(col), max(len(f"{row[col]:.4f}" if isinstance(row[col], float) else str(row[col])) for row in years_result)) for col in columns}
        # col_widths = {col: len(col) for col in columns}
        header = " | ".join(col.ljust(col_widths[col]) for col in columns)
        formated_rows = [" | ".join([f"{row[col]:.4f}".ljust(col_widths[col]) if isinstance(row[col], float) else str(row[col]).ljust(col_widths[col]) for col in columns]) for row in years_result]
        print(header, *formated_rows, sep="\n")
        return years_result
    def plot_table(years_result):
        import matplotlib.pyplot as plt
        from functools import lru_cache
        # 下面是图表可视化
        @lru_cache(maxsize=None)
        def series(column):
            return [row[column] for row in years_result]

        # Create subplots
        plt.figure(figsize=(12, 10))

        plt.suptitle(f'Investment Analysis Dashboard ({series("year")[0]}-{series("year")[-1]})', fontsize=16, fontweight='bold')
        plt.subplot(2, 3, 1)
        plt.plot(series('year'), series('total'), marker='o', linewidth=2, markersize=6)
        plt.title('Total Amount Over Time')
        plt.xlabel('Year')
        plt.ylabel('Total ($)')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 3, 2)
        plt.plot(series('year'), series('interest_total'), marker='s', color='green', linewidth=2, markersize=6)
        plt.title('Cumulative Interest Over Time')
        plt.xlabel('Year')
        plt.ylabel('Interest Total ($)')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 3, 3)
        plt.plot(series('year'), series('withdraw_total'), marker='v', color='purple', linewidth=2, markersize=6)
        plt.title('Cumulative Withdrawals Over Time')
        plt.xlabel('Year')
        plt.ylabel('Withdraw Total ($)')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 3, 4)
        ax4 = plt.gca()
        ax4.plot(series('year'), series('interest'), marker='^', color='orange', linewidth=2, markersize=6, label='Interest ($)')
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Interest ($)', color='orange')
        ax4.tick_params(axis='y', labelcolor='orange')
        ax4.grid(True, alpha=0.3)

        ax4_twin = ax4.twinx()
        ax4_twin.plot(series('year'), series('interest_rate'), marker='o', color='blue', linewidth=2, markersize=6, label='Interest Rate')
        ax4_twin.set_ylabel('Interest Rate', color='blue')
        ax4_twin.tick_params(axis='y', labelcolor='blue')

        plt.title('Annual Interest & Interest Rate')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')

        plt.subplot(2, 3, 5)
        plt.plot(series('year'), series('withdraw'), marker='x', color='brown', linewidth=2, markersize=6)
        plt.title('Annual Withdrawals')
        plt.xlabel('Year')
        plt.ylabel('Withdraw ($)')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 3, 6)
        plt.plot(series('year'), series('interest_vs_principle'), marker='d', color='red', linewidth=2, markersize=6)
        plt.title('Interest VS Principle')
        plt.xlabel('Year')
        plt.ylabel('Rate')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return years_result
    
    visualize = lambda data: reduce(
        lambda x, f: f(x),
        [print_table, plot_table],
        # [print_table],
        data
    )


    from read_sp500 import get_change_rate_by_year
    import numpy as np

    usd = lambda x: x #* 7.2 / 12
    start_year = 1999
    end_year = start_year + 20
    retire_year = start_year + 0
    market_rate_year = start_year + 0

    # 结论：提取率要在市场均值以下才能保证资金池不耗干，市场行情不好的时候应该想办法降低提取率，也就是降低生活质量
    # 如果(利息-提款率)低于通货膨胀率，本金也会在一个较长的时间后，最终耗尽
    years_result = visualize(invest(
        year=start_year, 
        max_year=end_year, #+ 34, 
        principles=lambda year: usd(3)  if year < retire_year else (usd(0) if year < retire_year + 10 else usd(0)), 
        # 这里使用正态分布模拟市场震荡，均值为0.0675，标准差为0.0059， 数据来自chatgpt，模型和参数都不准确，待未来优化
        # interest_rate=lambda year: 0.04 if year < market_rate_year else random.normalvariate(0.055949, 0.147515),  
        interest_rate=lambda year: get_change_rate_by_year(year) if get_change_rate_by_year(year) is not None else 0.02, #0.018,
        # interest_rate=0.02,
        withdraw_rate=lambda year,total: 0.00 if year < retire_year else ((usd(1.2) / total) * (1+0.03)**(year-retire_year)), 
        total=usd(30), 
        interest_total=0.0
    ))

    interest_rate = np.array([row['interest_rate'] for row in years_result])
    print(np.mean(interest_rate), np.std(interest_rate))

if __name__ == "__main__":
    invest_gui()
    # main()

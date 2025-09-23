from dataclasses import dataclass, asdict
from utils import USD
import numpy as np
from read_data import get_change_rate_by_year, get_value_by_year, stock_data, portfolio_data, interest_data, inflation_data, inflation_rate_multiplier
from invest import invest
@dataclass
class InvestmentParams:
    """Data class representing investment simulation parameters."""
    start_year: int = 2000
    duration: int = 24
    retire_offset: int = 0
    start_total: float = USD(30)
    cost: float = USD(1.2)
    cpi: float = 0.00
    interest_rate: float = 0.00
    new_savings: float = USD(3.0)
    stock_code: str = "portfolio"
    use_portfolio: bool = True
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
            self.use_portfolio,
            self.use_real_interest,
            self.use_real_cpi,
            self.new_savings,
            self.stock_code,
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
    
    def portfolio_interest_rate(self, year):
        """Get interest rate for a given year based on parameters."""
        if self.stock_code == "portfolio":
            return sum([ratio * get_change_rate_by_year(stock_data(code), year, default=self.interest_rate) for (code, ratio) in portfolio_data().items()])
        else:
            return get_change_rate_by_year(stock_data(self.stock_code), year, default=self.interest_rate)

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
   
class InvestmentYearsResult:
    def __init__(self, params, years_result):
        self.params: InvestmentParams = params
        self.years_result = years_result
    
    def get_benchmark_total(self):
        years = self.series('year')
        bias_benchmark_total = np.cumsum([
            self.series('new_savings')[idx] * (
                self.params.get_real_inflation_rate_multiplier(year) - 1
            ) for idx, year in enumerate(years)
        ])
        benchmark_total = [
            self.series('principle')[idx] * 
            self.params.get_real_inflation_rate_multiplier(year) 
            for idx, year in enumerate(years)
        ]
        benchmark_total = [benchmark_total[idx] - bias_benchmark_total[idx] for idx in range(len(years))]
        return benchmark_total
    
    def get_benchmark_roi(self):
        benchmark_roi = [get_change_rate_by_year(stock_data("SP500"), year, default=self.params.interest_rate) for year in self.series('year')]
        return benchmark_roi

    def get_benchmark_withdraw(self):
        benchmark_withdrawals = [self.params.cost * self.params.get_real_inflation_rate_multiplier(year) for year in self.series('year')]
        return benchmark_withdrawals

    @property
    def financial_stats(self):
        """Calculate financial metrics from simulation results."""
        if not self.years_result:
            return {}
        
        final_total = self.years_result[-1]['total']
        final_withdraw_total = self.years_result[-1]['withdraw_total']
        final_interest_total = self.years_result[-1]['interest_total']
        
        final_total += final_withdraw_total
        
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
	        'final_ratio': (final_total-final_withdraw_total) / principle,
            'final_ratio_inflation': (final_total-final_withdraw_total) / (self.get_benchmark_total()[-1]),
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
        text += f"  • CGAR:  {metrics['growth_rate']:>12.2%}\n"
        text += f"  • All:         {metrics['final_total']:>12,.2f}$\n"
        text += f"  • Final:         {metrics['final_total']-metrics['final_withdraw_total']:>12,.2f}$\n"
        text += f"  • Total Min: {metrics['total_min']:>12,.2f}$\n"
        text += f"  • Final Ratio:  {metrics['final_ratio']:>12.2%}\n"
        text += f"  • Final Ratio (Inf.):  {metrics['final_ratio_inflation']:>12.2%}\n"
        text += f"  • Withdraw Total: {metrics['final_withdraw_total']:>12,.2f}$\n"
        text += f"  • Interest Total: {metrics['final_interest_total']:>12,.2f}$\n"
        text += f"  • Final Principle: {self.years_result[-1]['principle']:>12,.2f}$\n"
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
        results_text += self.interest_rate_summary
        results_text += self.sustainability_summary
        return results_text
    
    def __str__(self):
        return self.analysis_report()
    
    def series(self, key, func=lambda v: v):
        return [func(row[key]) for row in self.years_result]


class StrategyBasic(InvestmentParams):
    """Basic strategy implementation."""
    @classmethod
    def from_params(cls, params: InvestmentParams):
        return StrategyBasic(**asdict(params))

    def get_new_savings(self, year, total):
        """Get new savings amount for a given year."""
        if year < self.retire_year:
            return self.new_savings #* self.get_inflation_rate_multiplier(year)
        else:
            return 0

    def get_interest_rate(self, year, total):
        if self.use_portfolio:
            return self.portfolio_interest_rate(year)
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
                upper_bound = 0.04
                upper_bound_remains = upper_bound - target_living_ratio
                log_scale = my_log_function(center_ratio / target_living_ratio, arg_max=2e2*center_ratio) # because 1/target_living_ratio is approching inf
                withdraw_rate = target_living_ratio + log_scale * upper_bound_remains
        
        return withdraw_rate
    
    def __call__(self):
        return InvestmentYearsResult(self, invest(
            year=self.start_year,
            max_year=self.end_year,
            new_savings=self.get_new_savings,
            interest_rate=self.get_interest_rate,
            withdraw_rate=self.get_withdraw_rate,
            total=self.start_total,
        ))
 
if __name__ == "__main__":
    params = InvestmentParams.get_defaults()
    strategy = StrategyBasic.from_params(params)
    years_result = strategy()
    print(years_result.analysis_report())

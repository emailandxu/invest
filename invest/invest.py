def invest(year, max_year, new_savings, interest_rate, withdraw_rate, total=0, interest_total=0, withdraw_total=0):
    if year > max_year:
        return []

    results = []
    current_total = total
    current_interest_total = interest_total
    current_withdraw_total = withdraw_total

    for current_year in range(year, max_year + 1):
        init_total = current_total

        # Calculate current year's principles (can be function or constant)
        if callable(new_savings):
            current_new_savings = new_savings(current_year, current_total)
        else:
            current_new_savings = new_savings

        # Calculate current year's interest rate (can be function or constant)
        if callable(interest_rate):
            current_interest_rate = interest_rate(current_year, current_total)
        else:
            current_interest_rate = interest_rate

        interest = init_total * current_interest_rate
        current_interest_total += interest
        current_total = init_total + interest + current_new_savings

        # Calculate current year's withdraw rate (can be function or constant)
        if callable(withdraw_rate):
            current_withdraw_rate = withdraw_rate(current_year, current_total)
        else:
            current_withdraw_rate = withdraw_rate

        # Calculate withdrawals and interest for current year
        withdraw = current_total * current_withdraw_rate

        if current_total - withdraw > 0:
            current_total -= withdraw
            current_withdraw_total += withdraw
        else:
            withdraw = current_total
            current_withdraw_total += withdraw
            current_total = 0

        principle = current_total + current_withdraw_total - current_interest_total
        base = principle + 1e-16

        current_year_data = {
            'year': current_year,
            'new_savings': current_new_savings,
            "interest_rate": current_interest_rate,
            'interest_total': current_interest_total,
            'interest': interest,
            "withdrawed_interest_rate": (interest - withdraw) / init_total if init_total != 0 else -1.0,
            'withdraw_total': current_withdraw_total,
            'withdraw': withdraw,
            'withdraw_rate': current_withdraw_rate,
            "withdrawed_interest_vs_principle": (current_interest_total - current_withdraw_total) / base,
            "interest_vs_principle": current_interest_total / base,
            'total': current_total,
            "principle": principle,
        }

        results.append(current_year_data)

    return results

if __name__ == "__main__":
    from .utils import print_table, USD
    print_table(invest(year=1995, max_year=1995 + 25, new_savings=3, 
                      interest_rate=0.04, 
                      withdraw_rate=lambda year, total: ((0.04 * 30) / total) * (1 + 0.02) ** (year - 1995)))
    

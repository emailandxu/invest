def invest(year, max_year, new_savings, interest_rate, withdraw_rate, total=0, interest_total=0, withdraw_total=0):
    if year > max_year:
        return []
    
    init_total = total
    
    # Calculate current year's principles (can be function or constant)
    if callable(new_savings):
        current_new_savings = new_savings(year, total)
    else:
        current_new_savings = new_savings
    
    # Calculate current year's interest rate (can be function or constant)
    if callable(interest_rate):
        current_interest_rate = interest_rate(year, total)
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
        'withdraw_rate': current_withdraw_rate,
        "withdrawed_interest_vs_principle": (interest_total - withdraw_total) / (total + withdraw_total - interest_total + 1e-16),
        'total': total,
        "principle": total + withdraw_total - interest_total,
    }
    
    # Recursively get remaining years and prepend current year
    return [current_year_data] + invest(year + 1, max_year, new_savings, interest_rate, withdraw_rate, 
                                       total, interest_total, withdraw_total)

if __name__ == "__main__":
    from utils import print_table, USD
    print_table(invest(1995, 1995 + 25, 3, 0.02, 0.04))

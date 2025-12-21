import csv
import os

from dataclasses import dataclass
from datetime import date
from functools import lru_cache, reduce, wraps
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ._paths import data_path


@dataclass
class Record:
    date: str
    year: int
    month: int
    day: int
    value: Optional[float]
    change_rate: Optional[float] = None
def file_cache(filepath):
    """
    Decorator that caches function results based on file modification time and size.
    If the file hasn't changed, returns cached result. If file changes, recomputes and caches new result.
    
    Args:
        filepath (str): Path to the file to monitor for changes
    
    Returns:
        Decorator function
    """
    def decorator(func):
        cache = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if file exists
            if not os.path.exists(filepath):
                # If file doesn't exist, always call function
                return func(*args, **kwargs)
            
            # Get file modification time and size for quick check
            stat = os.stat(filepath)
            mtime = stat.st_mtime
            size = stat.st_size
            
            # Create a key from mtime and size
            cache_key = f"{mtime}_{size}"
            
            # If we have a cached result with same mtime/size, return it
            if cache_key in cache:
                return cache[cache_key]
            
            # File has changed or no cache exists, compute new result
            result = func(*args, **kwargs)
            
            # Cache the result
            cache[cache_key] = result
            
            # Clean up old cache entries (keep only the latest)
            if len(cache) > 1:
                cache.clear()
                cache[cache_key] = result
            
            return result
        
        return wrapper
    return decorator

def _parse_date_fast(date_str: str) -> Optional[Tuple[int, int, int]]:
    """Parse date strings in formats YYYY-MM-DD or MM/DD/YYYY without strptime."""
    if not date_str:
        return None

    if '-' in date_str:
        parts = date_str.split('-')
        if len(parts) != 3:
            return None
        year, month, day = parts
    elif '/' in date_str:
        parts = date_str.split('/')
        if len(parts) != 3:
            return None
        month, day, year = parts
    else:
        return None

    try:
        return int(year), int(month), int(day)
    except ValueError:
        return None


def _to_float(value: Optional[str]) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_data(csv_path: str | Path | None = None) -> Tuple[List[str], Dict[str, int], List[List[str]]]:
    """Read CSV data and return headers, header index, and raw rows."""
    path = Path(csv_path) if csv_path is not None else data_path("sp500.csv")
    try:
        with path.open('r', newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader, None)
            if headers is None:
                return [], {}, []
            headers = [header.strip() for header in headers]
            header_index = {header: idx for idx, header in enumerate(headers)}

            rows: List[List[str]] = []
            column_count = len(headers)
            for row in reader:
                if len(row) < column_count:
                    row = row + [''] * (column_count - len(row))
                rows.append(row)

        return headers, header_index, rows
    except FileNotFoundError:
        print(f"Error: Could not find file {path}")
        return [], {}, []
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return [], {}, []
        
def to_records(csv_data: Tuple[List[str], Dict[str, int], List[List[str]]]) -> List[Record]:
    """Process market data by converting dates and extracting the target month."""
    headers, index, rows = csv_data
    if not rows:
        return []

    date_idx = index.get('Date')
    value_idx = index.get('Value')

    records: List[Record] = []
    for row in rows:
        date_str = row[date_idx] if date_idx is not None and date_idx < len(row) else ''
        parsed = _parse_date_fast(date_str)
        if not parsed:
            continue
        year, month_val, day = parsed

        value = _to_float(row[value_idx]) if value_idx is not None and value_idx < len(row) else None
        record = Record(date=date_str, year=year, month=month_val, day=day, value=value)
        records.append(record)

    return records


def extract_year_data_by_mean(csv_data: Tuple[List[str], Dict[str, int], List[List[str]]]) -> List[Record]:
    """Aggregate monthly data into yearly means."""
    headers, index, rows = csv_data
    if not rows:
        return []

    date_idx = index.get('Date')
    value_idx = index.get('Value')
    if date_idx is None or value_idx is None:
        return []

    year_groups: Dict[int, List[float]] = {}
    for row in rows:
        date_str = row[date_idx] if date_idx < len(row) else ''
        parsed = _parse_date_fast(date_str)
        if not parsed:
            continue
        year, month_val, day = parsed
        value = _to_float(row[value_idx])
        if value is None:
            continue
        year_groups.setdefault(year, []).append(value)

    year_data: List[Record] = []
    for year, values in year_groups.items():
        if not values:
            continue
        mean_value = sum(values) / len(values)
        year_data.append(Record(date=f"{year}-12-31", year=year, month=12, day=31, value=mean_value))

    return sorted(year_data, key=lambda r: r.year)

def compute_annual_change_rate(year_data: List[Record]) -> List[Record]:
    """Compute annual change rate for yearly market data."""
    if not year_data:
        return []

    data = sorted(year_data, key=lambda r: (r.year*10000 + r.month*100 + r.day))
    previous_value = None
    for record in data:
        current_value = record.value
        if previous_value is None or current_value is None or previous_value == 0:
            record.change_rate = None
        else:
            record.change_rate = (current_value - previous_value) / previous_value
        if current_value is not None:
            previous_value = current_value
    return data


def filter_data_after_1960(year_data: List[Record]) -> List[Record]:
    """Filter yearly data to only include years 1960 and after."""
    if not year_data:
        return []
    return [row for row in year_data if row.year >= 1960]

def get_change_rate_by_year(data: List[Record], year: int, default=None):
    """Get the change rate for a specific year."""
    if not data:
        return default

    base_year = data[0].year
    year_index = year - base_year
    if year_index < 0 or year_index >= len(data):
        return default

    value = data[year_index].change_rate
    return default if value is None else value


def get_value_by_year(data: List[Record], year: int, default=None):
    if not data:
        return default

    base_year = data[0].year
    year_index = year - base_year
    if year_index < 0 or year_index >= len(data):
        return default
    value = data[year_index].value
    return default if value is None else value


@lru_cache(maxsize=None)
def stock_data(code="SP500", per_year=True):
    csv_path = data_path("STOCK", f"{code}.csv")
    
    # Process the data
    csv_data = read_data(csv_path=csv_path)
    records = to_records(csv_data)
    if per_year:
        year_records = {}
        records = {r.year:r for r in records}
        records = list(records.values())

    records = compute_annual_change_rate(records)
    
    return records

@lru_cache(maxsize=None)
def interest_data():
    csv_path = data_path("interest.csv")
    
    # Process the data
    csv_data = read_data(csv_path=csv_path)
    year_data = extract_year_data_by_mean(csv_data)
    year_data = compute_annual_change_rate(year_data)
    # Convert value to float, divide by 100, and add 1.0 for interest rate processing
    for row in year_data:
        if row.value is not None:
            row.value = row.value / 100.0
    
    return year_data

@lru_cache(maxsize=None)
@lru_cache(maxsize=None)
def inflation_data():
    csv_path = data_path("inflation.csv")
    
    # Process the data
    csv_data = read_data(csv_path=csv_path)
    records = to_records(csv_data)
    year_data = [r for r in records if r.month == 12]
    year_data = compute_annual_change_rate(year_data)
    for row in year_data:
        if row.value is not None:
            row.value = row.value / 100.0
    
    return year_data

@file_cache(data_path("portfolio.csv"))
def portfolio_data() -> Dict[str, float]:
    """
    Read portfolio allocation data from CSV file.
    
    Returns:
        Dict[str, float]: Dictionary mapping asset codes to their allocation ratios
    """
    csv_path = data_path("portfolio.csv")
    headers, index, rows = read_data(csv_path=csv_path)
    if not rows:
        return {}

    first_row = rows[0]
    codes = [header.strip() for header in headers]
    ratios = []
    for code in codes:
        idx = index.get(code)
        raw_value = first_row[idx] if idx is not None and idx < len(first_row) else "0"
        value = _to_float(raw_value)
        ratios.append(value if value is not None else 0.0)

    portfolio_data = dict(zip(codes, ratios))
    print(portfolio_data)
    return portfolio_data

@lru_cache()
def inflation_rate_multiplier(year, start_year, default=0.00):
    cpi = 1 + get_value_by_year(inflation_data(), year, default=default)
    if year <= start_year:
        return cpi
    else:
        return cpi * inflation_rate_multiplier(year-1, start_year, default=default)


def get_inflation_multiplier_for_date(start_date, target_date, default_rate=0.03):
    """
    Calculate cumulative inflation multiplier from start_date to target_date.
    Uses annual inflation rates with linear interpolation within years.
    
    Args:
        start_date: Start date (baseline, multiplier = 1.0)
        target_date: Target date to calculate multiplier for
        default_rate: Default annual inflation rate if data unavailable
    
    Returns:
        float: Cumulative inflation multiplier (e.g., 1.05 = 5% cumulative inflation)
    """
    from datetime import date
    
    if isinstance(start_date, str):
        start_date = date.fromisoformat(start_date)
    if isinstance(target_date, str):
        target_date = date.fromisoformat(target_date)
    
    if target_date <= start_date:
        return 1.0
    
    # Get inflation data
    inf_data = inflation_data()
    inf_by_year = {r.year: r.value for r in inf_data if r.value is not None}
    
    multiplier = 1.0
    current_year = start_date.year
    end_year = target_date.year
    
    # For each complete year between start and end
    for year in range(current_year, end_year + 1):
        annual_rate = inf_by_year.get(year, default_rate)
        
        if year == current_year == end_year:
            # Same year: partial
            days_in_year = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
            days_elapsed = (target_date - start_date).days
            partial_rate = annual_rate * (days_elapsed / days_in_year)
            multiplier *= (1 + partial_rate)
        elif year == current_year:
            # First year: from start_date to end of year
            year_end = date(year, 12, 31)
            days_in_year = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
            days_elapsed = (year_end - start_date).days
            partial_rate = annual_rate * (days_elapsed / days_in_year)
            multiplier *= (1 + partial_rate)
        elif year == end_year:
            # Last year: from start of year to target_date
            year_start = date(year, 1, 1)
            days_in_year = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
            days_elapsed = (target_date - year_start).days
            partial_rate = annual_rate * (days_elapsed / days_in_year)
            multiplier *= (1 + partial_rate)
        else:
            # Complete year
            multiplier *= (1 + annual_rate)
    
    return multiplier


def stock_data_daily(code: str = "VTI"):
    """
    Load daily stock data as Bar objects for backtesting.
    
    Returns a list of Bar dataclass with date, open, high, low, close, volume.
    Since source data only has close price, OHLC all use the close value.
    
    Args:
        code: Stock symbol (e.g., "VTI", "SP500")
    
    Returns:
        List of Bar objects sorted by date
    """
    from datetime import datetime as dt
    from .backtest import Bar
    
    csv_path = data_path("STOCK", f"{code}.csv")
    headers, index, rows = read_data(csv_path=csv_path)
    
    if not rows:
        return []
    
    date_idx = index.get('Date')
    value_idx = index.get('Value')
    
    if date_idx is None or value_idx is None:
        return []
    
    bars = []
    for row in rows:
        date_str = row[date_idx] if date_idx < len(row) else ''
        parsed = _parse_date_fast(date_str)
        if not parsed:
            continue
        
        year, month, day = parsed
        try:
            bar_date = dt(year, month, day).date()
        except ValueError:
            continue
        
        value = _to_float(row[value_idx]) if value_idx < len(row) else None
        if value is None:
            continue
        
        # Create Bar with same value for OHLC since we only have close price
        bar = Bar(
            symbol=code,
            date=bar_date,
            open=value,
            high=value,
            low=value,
            close=value,
            volume=0.0
        )
        bars.append(bar)
    
    # Sort by date
    bars.sort(key=lambda b: b.date)
    return bars


def load_dividends(code: str) -> Dict[date, float]:
    """
    Load dividend data for a stock.
    
    Returns:
        Dict mapping date to dividend per share amount.
        Empty dict if no dividend data available.
    """
    from datetime import date
    
    csv_path = data_path("STOCK", f"{code}_dividends.csv")
    if not csv_path.exists():
        # Only warn for non-CASH symbols
        if code != "CASH":
            print(f"Warning: Dividend file missing for {code}. Backtest returns may be lower than expected.")
        return {}
    
    dividends = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    d = date.fromisoformat(row['Date'])
                    div = float(row['Dividend'])
                    dividends[d] = div
                except (ValueError, KeyError):
                    continue
    except Exception:
        return {}
    
    return dividends



def load_rf_rates() -> Dict[date, float]:
    """
    Load monthly risk-free rates (FEDFUNDS) from interest.csv.
    Returns: Dict[date, rate] where rate is annualized (e.g. 0.05 for 5%)
    """
    csv_path = data_path("interest.csv")
    csv_data = read_data(csv_path=csv_path)
    records = to_records(csv_data)
    
    final_rates = {}
    for r in records:
        if r.value is not None:
            ymd = _parse_date_fast(r.date)
            if ymd:
                d = date(ymd[0], ymd[1], ymd[2])
                final_rates[d] = r.value / 100.0
                
    return final_rates


if __name__ == "__main__":
    from .utils import print_table
    import sys
    code = sys.argv[1] if len(sys.argv) > 1 else "GLDM"
    print_table(stock_data(code, per_year=True), formats={"change_rate":".2%"})
import csv
import os
import pickle
from dataclasses import dataclass
from functools import lru_cache, reduce, wraps
from typing import Dict, List, Optional, Tuple


@dataclass
class YearRecord:
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


def read_data(csv_path='data/sp500.csv') -> Tuple[List[str], Dict[str, int], List[List[str]]]:
    """Read CSV data and return headers, header index, and raw rows."""
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8-sig') as csvfile:
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
        print(f"Error: Could not find file {csv_path}")
        return [], {}, []
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return [], {}, []
def extract_year_end_data_by_month(csv_data: Tuple[List[str], Dict[str, int], List[List[str]]], month=12) -> List[YearRecord]:
    """Process market data by converting dates and extracting the target month."""
    headers, index, rows = csv_data
    if not rows:
        return []

    date_idx = index.get('Date')
    value_idx = index.get('Value')

    year_records: Dict[int, YearRecord] = {}
    for row in rows:
        date_str = row[date_idx] if date_idx is not None and date_idx < len(row) else ''
        parsed = _parse_date_fast(date_str)
        if not parsed:
            continue
        year, month_val, day = parsed
        if month_val != month:
            continue

        value = _to_float(row[value_idx]) if value_idx is not None and value_idx < len(row) else None
        year_records[year] = YearRecord(date=date_str, year=year, month=month_val, day=day, value=value)

    return sorted(year_records.values(), key=lambda r: r.year)


def extract_year_data_by_mean(csv_data: Tuple[List[str], Dict[str, int], List[List[str]]]) -> List[YearRecord]:
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

    year_data: List[YearRecord] = []
    for year, values in year_groups.items():
        if not values:
            continue
        mean_value = sum(values) / len(values)
        year_data.append(YearRecord(date=f"{year}-12-31", year=year, month=12, day=31, value=mean_value))

    return sorted(year_data, key=lambda r: r.year)

def compute_annual_change_rate(year_data: List[YearRecord]) -> List[YearRecord]:
    """Compute annual change rate for yearly market data."""
    if not year_data:
        return []

    data = sorted(year_data, key=lambda r: r.year)
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


def filter_data_after_1960(year_data: List[YearRecord]) -> List[YearRecord]:
    """Filter yearly data to only include years 1960 and after."""
    if not year_data:
        return []
    return [row for row in year_data if row.year >= 1960]

def get_change_rate_by_year(data: List[YearRecord], year: int, default=None):
    """Get the change rate for a specific year."""
    if not data:
        return default

    base_year = data[0].year
    year_index = year - base_year
    if year_index < 0 or year_index >= len(data):
        return default

    value = data[year_index].change_rate
    return default if value is None else value


def get_value_by_year(data: List[YearRecord], year: int, default=None):
    if not data:
        return default

    base_year = data[0].year
    year_index = year - base_year
    if year_index < 0 or year_index >= len(data):
        return default
    value = data[year_index].value
    return default if value is None else value


@lru_cache(maxsize=None)
def stock_data(code="SP500"):
    csv_path = "data/STOCK/{}.csv".format(code)
    pickle_path = os.path.splitext(csv_path)[0] + "_processed.pkl"
    
    # Try to load from pickle cache first
    # if os.path.exists(pickle_path):
    #     try:
    #         with open(pickle_path, 'rb') as f:
    #             return pickle.load(f)
    #     except (pickle.PickleError, EOFError, FileNotFoundError):
    #         pass  # Fall through to recompute if pickle is corrupted
    
    # Process the data
    csv_data = read_data(csv_path=csv_path)
    year_data = extract_year_end_data_by_month(csv_data, month=12)
    year_data = compute_annual_change_rate(year_data)
    # year_data = filter_data_after_1960(year_data)
    
    # Save to pickle cache
    # try:
    #     with open(pickle_path, 'wb') as f:
    #         pickle.dump(year_data, f)
    # except (pickle.PickleError, IOError):
    #     pass  # Continue even if we can't save the cache
    
    return year_data

@lru_cache(maxsize=None)
def interest_data():
    csv_path = "data/interest.csv"
    pickle_path = os.path.splitext(csv_path)[0] + "_processed.pkl"
    
    # Try to load from pickle cache first
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        except (pickle.PickleError, EOFError, FileNotFoundError):
            pass  # Fall through to recompute if pickle is corrupted
    
    # Process the data
    csv_data = read_data(csv_path=csv_path)
    year_data = extract_year_data_by_mean(csv_data)
    year_data = compute_annual_change_rate(year_data)
    # Convert value to float, divide by 100, and add 1.0 for interest rate processing
    for row in year_data:
        if row.value is not None:
            row.value = row.value / 100.0
    
    # Save to pickle cache
    try:
        with open(pickle_path, 'wb') as f:
            pickle.dump(year_data, f)
    except (pickle.PickleError, IOError):
        pass  # Continue even if we can't save the cache
    
    return year_data

@lru_cache(maxsize=None)
def inflation_data():
    csv_path = "data/inflation.csv"
    pickle_path = os.path.splitext(csv_path)[0] + "_processed.pkl"
    
    # Try to load from pickle cache first
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        except (pickle.PickleError, EOFError, FileNotFoundError):
            pass  # Fall through to recompute if pickle is corrupted
    
    # Process the data
    csv_data = read_data(csv_path=csv_path)
    year_data = extract_year_end_data_by_month(csv_data, month=12)
    year_data = compute_annual_change_rate(year_data)
    for row in year_data:
        if row.value is not None:
            row.value = row.value / 100.0
    
    # Save to pickle cache
    try:
        with open(pickle_path, 'wb') as f:
            pickle.dump(year_data, f)
    except (pickle.PickleError, IOError):
        pass  # Continue even if we can't save the cache
    
    return year_data

@file_cache("data/portfolio.csv")
def portfolio_data() -> Dict[str, float]:
    """
    Read portfolio allocation data from CSV file.
    
    Returns:
        Dict[str, float]: Dictionary mapping asset codes to their allocation ratios
    """
    csv_path = "data/portfolio.csv"
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

if __name__ == "__main__":
    # gui_data(interest_data())
    # print(interest_data())
    # print(get_change_rate_by_year(2010))
    import time
    t0 = time.time()
    print(inflation_rate_multiplier(2025, 1995))
    t1 = time.time()
    print(f"Time taken: {t1 - t0:.6f} seconds")
    
    t0 = time.time()
    print(inflation_rate_multiplier(2021, 1995))
    t1 = time.time()
    print(f"Time taken: {t1 - t0:.6f} seconds")

from functools import lru_cache, reduce
import csv
import pickle
import os

def read_data(csv_path='data/sp500.csv'):
    """
    Read S&P 500 data from CSV file.
    
    Args:
        csv_path (str): Path to the S&P 500 CSV file
        
    Returns:
        list: List of dictionaries containing S&P 500 data
    """
    try:
        data = []
        with open(csv_path, 'r', newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
        return data
    except FileNotFoundError:
        print(f"Error: Could not find file {csv_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
def extract_year_end_data_by_month(data, month=12):
    """
    Process S&P 500 data by converting dates and extracting yearly data.
    
    Args:
        data (list): Raw S&P 500 data as list of dictionaries
        
    Returns:
        list: Processed S&P 500 data with yearly records
    """
    if data is not None:
        from datetime import datetime
        
        # Convert Date column to datetime if it exists
        for row in data:
            if 'Date' in row:
                try:
                    date_obj = datetime.strptime(row['Date'], '%Y-%m-%d')
                    row['Year'] = date_obj.year
                    row['Month'] = date_obj.month
                except ValueError:
                    # Try alternative date formats
                    try:
                        date_obj = datetime.strptime(row['Date'], '%m/%d/%Y')
                        row['Year'] = date_obj.year
                        row['Month'] = date_obj.month
                    except ValueError:
                        continue
        
        # Get the specified month data of each year
        year_data = [row for row in data if row.get('Month') == month]
    else:
        year_data = None
    
    return year_data

def extract_year_data_by_mean(data):
    """
    Process S&P 500 data by converting dates and extracting yearly data using mean values.
    
    Args:
        data (list): Raw S&P 500 data as list of dictionaries
        
    Returns:
        list: Processed S&P 500 data with yearly records averaged
    """
    if data is not None:
        from datetime import datetime
        
        # Convert Date column to datetime if it exists
        for row in data:
            if 'Date' in row:
                try:
                    date_obj = datetime.strptime(row['Date'], '%Y-%m-%d')
                    row['Year'] = date_obj.year
                    row['Month'] = date_obj.month
                except ValueError:
                    # Try alternative date formats
                    try:
                        date_obj = datetime.strptime(row['Date'], '%m/%d/%Y')
                        row['Year'] = date_obj.year
                        row['Month'] = date_obj.month
                    except ValueError:
                        continue
        
        # Group data by year and calculate mean values
        year_groups = {}
        for row in data:
            year = row.get('Year')
            if year is not None:
                if year not in year_groups:
                    year_groups[year] = []
                year_groups[year].append(row)
        
        # Calculate mean values for each year
        year_data = []
        for year, rows in year_groups.items():
            if 'Value' in rows[0]:
                try:
                    values = [float(row['Value']) for row in rows if row.get('Value')]
                    if values:
                        mean_value = sum(values) / len(values)
                        year_data.append({
                            'Date': f"{year}-12-31",  # Use end of year as representative date
                            'Year': year,
                            'Month': 12,
                            'Value': str(mean_value)
                        })
                except ValueError:
                    continue
        
        # Sort by year
        year_data.sort(key=lambda x: x.get('Year', 0))
    else:
        year_data = None
    
    return year_data

def compute_annual_change_rate(year_data):
    """
    Compute annual change rate for S&P 500 data.
    
    Args:
        year_data (list): Yearly S&P 500 data as list of dictionaries
        
    Returns:
        list: List of dictionaries with annual change rates added
    """
    if year_data is None or len(year_data) == 0:
        return None
    
    # Make a copy to avoid modifying the original data
    data = [row.copy() for row in year_data]
    
    # Sort by year to ensure proper order
    data.sort(key=lambda x: x.get('Year', 0))
    
    # Calculate annual change rate
    # Using the 'Value' column for S&P 500 data
    price_column = 'Value'
    
    if not any(price_column in row for row in data):
        print("Warning: Could not find 'Value' column")
        return data
    
    # Calculate year-over-year change rate
    for i, row in enumerate(data):
        if i == 0:
            row['Change_Rate'] = None  # First row has no previous year
        else:
            try:
                current_value = float(row[price_column])
                previous_value = float(data[i-1][price_column])
                if previous_value != 0:
                    row['Change_Rate'] = (current_value - previous_value) / previous_value
                else:
                    row['Change_Rate'] = None
            except (ValueError, KeyError):
                row['Change_Rate'] = None
    
    return data

def filter_data_after_1960(year_data):
    """
    Filter S&P 500 data to only include years 1960 and after.
    
    Args:
        year_data (list): S&P 500 yearly data as list of dictionaries
        
    Returns:
        list: Filtered data with years >= 1960, or None if input is None
    """
    if year_data is None or len(year_data) == 0:
        return None
    
    # Filter data to only include years 1960 and after
    filtered_data = [row for row in year_data if row.get('Year', 0) >= 1960]
    
    return filtered_data

def get_change_rate_by_year(data, year, default=None):
    """
    Get the change rate for a specific year from S&P 500 data.
    
    Args:
        year (int): The year to get the change rate for
        
    Returns:
        float: The change rate for the given year, or None if year not found
    """    
    if data is None or len(data) == 0:
        return None
    
    # Find the row for the given year
    # year_row = None
    # for row in data:
    #     if row.get('Year') == year:
    #         year_row = row
    #         break
    # if year_row is None:
    #     return default
    year_index = year - data[0].get('Year') 
    if year_index < 0 or year_index >= len(data):
        return default
    
    year_row = data[year_index]
    
    # Return the change rate for that year
    return year_row.get('Change_Rate')

def get_value_by_year(data, year, default=None):
    if data is None or len(data) == 0:
        return None
    
    # Find the row for the given year
    # year_row = None
    # for row in data:
    #     if row.get('Year') == year:
    #         year_row = row
    #         break
    
    # if year_row is None:
    #     return default

    year_index = year - data[0].get('Year') 
    if year_index < 0 or year_index >= len(data):
        return default
    year_row = data[year_index]
    # Return the change rate for that year
    return year_row.get('Value')


@lru_cache(maxsize=None)
def sp500_data():
    csv_path = "data/sp500.csv"
    pickle_path = os.path.splitext(csv_path)[0] + "_processed.pkl"
    
    # Try to load from pickle cache first
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        except (pickle.PickleError, EOFError, FileNotFoundError):
            pass  # Fall through to recompute if pickle is corrupted
    
    # Process the data
    data = read_data(csv_path=csv_path)
    year_data = extract_year_end_data_by_month(data, month=12)
    year_data = compute_annual_change_rate(year_data)
    # year_data = filter_data_after_1960(year_data)
    
    # Save to pickle cache
    try:
        with open(pickle_path, 'wb') as f:
            pickle.dump(year_data, f)
    except (pickle.PickleError, IOError):
        pass  # Continue even if we can't save the cache
    
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
    data = read_data(csv_path=csv_path)
    # year_data = extract_year_end_data_by_month(data, month=12)
    year_data = extract_year_data_by_mean(data)
    year_data = compute_annual_change_rate(year_data)
    # Convert value to float, divide by 100, and add 1.0 for interest rate processing
    for row in year_data:
        if 'Value' in row and row['Value'] is not None:
            try:
                row['Value'] = float(row['Value']) / 100
            except (ValueError, TypeError):
                row['Value'] = None
    
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
    data = read_data(csv_path=csv_path)
    year_data = extract_year_end_data_by_month(data, month=12)
    year_data = compute_annual_change_rate(year_data)
    # Convert value to float, divide by 100, and add 1.0 for interest rate processing
    for row in year_data:
        if 'Value' in row and row['Value'] is not None:
            try:
                row['Value'] = float(row['Value']) / 100
            except (ValueError, TypeError):
                row['Value'] = None
    
    # Save to pickle cache
    try:
        with open(pickle_path, 'wb') as f:
            pickle.dump(year_data, f)
    except (pickle.PickleError, IOError):
        pass  # Continue even if we can't save the cache
    
    return year_data

@lru_cache()
def inflation_rate_multiplier(year, start_year, default=0.00):
    cpi = 1 + get_value_by_year(inflation_data(), year, default=default)
    if year <= start_year:
        return cpi
    else:
        return cpi * inflation_rate_multiplier(year-1, start_year, default=default)

def gui_data(data):
    import tkinter as tk
    from tkinter import ttk
    
    def create_gui():
        root = tk.Tk()
        root.title("S&P 500 Data Analyzer")
        root.geometry("800x600")
        
        # Get data
        data_length = len(data) if data else 0
        
        # Create main frames
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Offset slider
        ttk.Label(main_frame, text="Offset:").grid(row=0, column=0, sticky=tk.W, pady=5)
        offset_var = tk.IntVar(value=max(0, data_length-20))
        offset_scale = ttk.Scale(main_frame, from_=-data_length, to=data_length-1, variable=offset_var, orient=tk.HORIZONTAL, length=300)
        offset_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=10)
        offset_label = ttk.Label(main_frame, text=str(offset_var.get()))
        offset_label.grid(row=0, column=2, sticky=tk.W, pady=5)
        
        # Start slider
        ttk.Label(main_frame, text="Start:").grid(row=1, column=0, sticky=tk.W, pady=5)
        start_var = tk.IntVar(value=0)
        start_scale = ttk.Scale(main_frame, from_=0, to=data_length-1, variable=start_var, orient=tk.HORIZONTAL, length=300)
        start_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=10)
        start_label = ttk.Label(main_frame, text=str(start_var.get()))
        start_label.grid(row=1, column=2, sticky=tk.W, pady=5)
        
        # End slider
        ttk.Label(main_frame, text="End:").grid(row=2, column=0, sticky=tk.W, pady=5)
        end_var = tk.IntVar(value=3)
        end_scale = ttk.Scale(main_frame, from_=0, to=data_length, variable=end_var, orient=tk.HORIZONTAL, length=300)
        end_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5, padx=10)
        end_label = ttk.Label(main_frame, text=str(end_var.get()))
        end_label.grid(row=2, column=2, sticky=tk.W, pady=5)
        
        # Year range label
        year_range_label = ttk.Label(main_frame, text="", font=("Arial", 10, "bold"))
        year_range_label.grid(row=3, column=0, columnspan=3, pady=5)
        
        # Text widget for data display
        text_frame = ttk.Frame(main_frame)
        text_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        text_widget = tk.Text(text_frame, height=20, width=80, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Result label
        result_label = ttk.Label(main_frame, text="", font=("Arial", 12, "bold"))
        result_label.grid(row=5, column=0, columnspan=3, pady=10)
        
        def update_display(*args):
            if not data:
                return
                
            offset = offset_var.get()
            start = start_var.get()
            end = end_var.get()

            if start_var.get() + offset < 0:
                offset = -start
            elif end_var.get() + offset > data_length:
                offset = data_length - end

            start += offset
            end += offset

            # Update labels
            offset_label.config(text=str(offset))
            start_label.config(text=str(start))
            end_label.config(text=str(end))
            
            # Ensure end is not less than start
            if end <= start:
                end = start + 1
                end_var.set(end - offset)
            
            # Ensure indices are within bounds
            start = max(0, min(start, data_length - 1))
            end = max(start + 1, min(end, data_length))
            
            # Get subset of data
            subset_data = data[start:end]
            
            # Update year range label
            if subset_data:
                years = [row.get('Year') for row in subset_data if row.get('Year') is not None]
                if years:
                    start_year = min(years)
                    end_year = max(years)
                    unique_years = list(set(years))
                    years_str = str(len(unique_years))
                    year_range_label.config(text=f"Year Range: {start_year} - {end_year} | Years: {years_str}")
                else:
                    year_range_label.config(text="Year Range: N/A")
            else:
                year_range_label.config(text="Year Range: N/A")
            
            # Clear text widget
            text_widget.delete(1.0, tk.END)
            
            # Display data
            if subset_data:
                text_widget.insert(tk.END, "Selected Data:\n")
                text_widget.insert(tk.END, "=" * 50 + "\n")
                
                # Format data for display
                for i, row in enumerate(subset_data):
                    text_widget.insert(tk.END, f"Row {start + i}:\n")
                    for key, value in row.items():
                        text_widget.insert(tk.END, f"  {key}: {value}\n")
                    text_widget.insert(tk.END, "\n")
                
                # Calculate and display reduced result
                change_rates = [row.get('Change_Rate') for row in subset_data if row.get('Change_Rate') is not None]
                if change_rates:
                    result = reduce(lambda x, y: x * (1 + y), change_rates, 1)
                    
                    # Calculate annual return rate
                    num_years = len(change_rates)
                    if num_years > 1:
                        annual_return = (result ** (1/num_years)) - 1
                        mean_change_rate = sum(change_rates) / len(change_rates)
                        std_change_rate = (sum((x - mean_change_rate) ** 2 for x in change_rates) / len(change_rates)) ** 0.5
                        result_label.config(text=f"Cumulative Growth Factor: {result:.6f} | Annual Return: {annual_return:.4%} | Mean: {mean_change_rate:.4%} | Std: {std_change_rate:.4%}")
                    else:
                        result_label.config(text=f"Cumulative Growth Factor: {result:.6f}")
                else:
                    result_label.config(text="No valid change rate data in selection")
            else:
                text_widget.insert(tk.END, "No data in selected range")
                result_label.config(text="")
        
        # Key event handler for offset adjustment
        def on_key_press(event):
            current_offset = offset_var.get()
            if event.keysym == 'Left':
                new_offset = max(-data_length, current_offset - 1)
                offset_var.set(new_offset)
            elif event.keysym == 'Right':
                new_offset = min(data_length - 1, current_offset + 1)
                offset_var.set(new_offset)
        
        # Bind slider events
        offset_var.trace('w', update_display)
        start_var.trace('w', update_display)
        end_var.trace('w', update_display)
        
        # Bind key events
        root.bind('<Left>', on_key_press)
        root.bind('<Right>', on_key_press)
        root.focus_set()  # Ensure the root window can receive key events
        
        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        # Initial display
        update_display()
        
        root.mainloop()
    
    create_gui()

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
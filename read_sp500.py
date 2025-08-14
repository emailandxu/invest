from functools import lru_cache, reduce
import pandas as pd

def read_sp500_data(csv_path='data/sp500.csv'):
    """
    Read S&P 500 data from CSV file.
    
    Args:
        csv_path (str): Path to the S&P 500 CSV file
        
    Returns:
        pandas.DataFrame: DataFrame containing S&P 500 data
    """
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print(f"Error: Could not find file {csv_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

def extract_year_end_sp500_data(sp500_data, month=12):
    """
    Process S&P 500 data by converting dates and extracting yearly data.
    
    Args:
        sp500_data (pandas.DataFrame): Raw S&P 500 data
        
    Returns:
        pandas.DataFrame: Processed S&P 500 data with yearly records
    """
    if sp500_data is not None:
        # Convert Date column to datetime if it exists
        if 'Date' in sp500_data.columns:
            sp500_data['Date'] = pd.to_datetime(sp500_data['Date'])
            sp500_data['Year'] = sp500_data['Date'].dt.year
            sp500_data['Month'] = sp500_data['Date'].dt.month
            
            # Get the last month data of each year (December data)
            sp500_year_data = sp500_data[sp500_data['Month'] == month].copy()
        else:
            sp500_year_data = sp500_data
    else:
        sp500_year_data = None
    
    return sp500_year_data

def compute_annual_change_rate(sp500_year_data):
    """
    Compute annual change rate for S&P 500 data.
    
    Args:
        sp500_year_data (pandas.DataFrame): Yearly S&P 500 data
        
    Returns:
        pandas.DataFrame: DataFrame with annual change rates added
    """
    if sp500_year_data is None or sp500_year_data.empty:
        return None
    
    # Make a copy to avoid modifying the original data
    df = sp500_year_data.copy()
    
    # Sort by year to ensure proper order
    df = df.sort_values('Year').reset_index(drop=True)
    
    # Calculate annual change rate
    # Using the 'Value' column for S&P 500 data
    price_column = 'Value'
    
    if price_column not in df.columns:
        print("Warning: Could not find 'Value' column")
        return df
    
    # Calculate year-over-year change rate
    df['Change_Rate'] = df[price_column].pct_change()
    
    return df

def filter_sp500_data_after_1960(sp500_year_data):
    """
    Filter S&P 500 data to only include years 1960 and after.
    
    Args:
        sp500_year_data (pandas.DataFrame): S&P 500 yearly data
        
    Returns:
        pandas.DataFrame: Filtered data with years >= 1960, or None if input is None
    """
    if sp500_year_data is None or sp500_year_data.empty:
        return None
    
    # Filter data to only include years 1960 and after
    filtered_data = sp500_year_data[sp500_year_data['Year'] >= 1960].copy()
    
    # Reset index after filtering
    filtered_data = filtered_data.reset_index(drop=True)
    
    return filtered_data


def get_change_rate_by_year(year):
    """
    Get the change rate for a specific year from S&P 500 data.
    
    Args:
        year (int): The year to get the change rate for
        
    Returns:
        float: The change rate for the given year, or None if year not found
    """
    sp500_data = data()
    
    if sp500_data is None or sp500_data.empty:
        return None
    
    # Find the row for the given year
    year_row = sp500_data[sp500_data['Year'] == year]
    
    if year_row.empty:
        return None
    
    # Return the change rate for that year
    return year_row['Change_Rate'].iloc[0]


@lru_cache(maxsize=None)
def data():
    sp500_data = read_sp500_data()
    sp500_year_data = extract_year_end_sp500_data(sp500_data, month=6)
    sp500_year_data = compute_annual_change_rate(sp500_year_data)
    # sp500_year_data = filter_sp500_data_after_1960(sp500_year_data)
    return sp500_year_data



def gui_sp500_data():
    import tkinter as tk
    from tkinter import ttk
    import pandas as pd
    def create_gui():
        root = tk.Tk()
        root.title("S&P 500 Data Analyzer")
        root.geometry("800x600")
        
        # Get data
        sp500_data = data()
        data_length = len(sp500_data)
        
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
            subset_data = sp500_data.iloc[start:end]
            
            # Update year range label
            if not subset_data.empty and 'Year' in subset_data.columns:
                start_year = subset_data['Year'].min()
                end_year = subset_data['Year'].max()
                years_list = sorted(subset_data['Year'].unique())
                # years_str = ', '.join(map(str, years_list))
                years_str = str(len(years_list))
                year_range_label.config(text=f"Year Range: {start_year} - {end_year} | Years: {years_str}")
            else:
                year_range_label.config(text="Year Range: N/A")
            
            # Clear text widget
            text_widget.delete(1.0, tk.END)
            
            # Display data
            if not subset_data.empty:
                text_widget.insert(tk.END, "Selected Data:\n")
                text_widget.insert(tk.END, "=" * 50 + "\n")
                text_widget.insert(tk.END, subset_data.to_string())
                text_widget.insert(tk.END, "\n\n")
                
                # Calculate and display reduced result
                if 'Change_Rate' in subset_data.columns:
                    change_rates = subset_data['Change_Rate'].dropna()
                    if not change_rates.empty:
                        result = reduce(lambda x, y: x * (1 + y), change_rates, 1)
                        
                        # Calculate annual return rate
                        num_years = len(change_rates)
                        if num_years > 1:
                            annual_return = (result ** (1/num_years)) - 1
                            result_label.config(text=f"Cumulative Growth Factor: {result:.6f} | Annual Return: {annual_return:.4%}")
                        else:
                            result_label.config(text=f"Cumulative Growth Factor: {result:.6f}")
                    else:
                        result_label.config(text="No valid change rate data in selection")
                else:
                    result_label.config(text="Change_Rate column not found")
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
    # gui_sp500_data()
    print(get_change_rate_by_year(2010))
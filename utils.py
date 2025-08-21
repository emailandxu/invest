USD = lambda x: x * 7.2

def print_table(years_result):
    columns = list(years_result[0].keys())
    col_widths = {col: max(len(col), max(len(f"{row[col]:.4f}" if isinstance(row[col], float) else str(row[col])) for row in years_result)) for col in columns}
    # col_widths = {col: len(col) for col in columns}
    header = " | ".join(col.ljust(col_widths[col]) for col in columns)
    formated_rows = [" | ".join([f"{row[col]:.4f}".ljust(col_widths[col]) if isinstance(row[col], float) else str(row[col]).ljust(col_widths[col]) for col in columns]) for row in years_result]
    print(header, *formated_rows, sep="\n")
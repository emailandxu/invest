USD = lambda x: x * 7.1

def columns_name_of_rows(rows):
    if row := next(iter(rows), None):
        return list(row.keys())

def print_table(rows, formats=None):
    if formats is None:
        formats = {}

    if not isinstance(rows[0], dict):
        rows = [vars(row) for row in rows]

    columns = list(rows[0].keys())
    col_widths = {col: max(len(col), max(len(f"{row[col]:{formats.get(col, '.2f')}}" if isinstance(row[col], float) else str(row[col])) for row in rows)) for col in columns}
    header = " | ".join(col.ljust(col_widths[col]) for col in columns)
    formated_rows = [" | ".join([f"{row[col]:{formats.get(col, '.2f')}}".ljust(col_widths[col]) if isinstance(row[col], float) else str(row[col]).ljust(col_widths[col]) for col in columns]) for row in rows]
    print(header, *formated_rows, sep="\n")
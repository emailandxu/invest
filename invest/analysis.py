import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List

from ._paths import data_path
from .utils import USD, print_table, columns_name_of_rows

def _format_date(yyyymmdd: str) -> str:
    if len(yyyymmdd) == 8 and yyyymmdd.isdigit():
        return f"{yyyymmdd[0:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"
    return yyyymmdd

def parse_account(account_hist_xml: str):
    """Parse an IBKR Flex XML and print date vs total using print_table.

    Extracts all EquitySummaryByReportDateInBase entries, collects reportDate and total,
    sorts by date, and prints a table with totals in USD and converted via USD().
    """
    tree = ET.parse(account_hist_xml)
    root = tree.getroot()

    rows = []
    for elem in root.findall('.//EquitySummaryByReportDateInBase'):
        date_raw = elem.get('reportDate')
        total_str = elem.get('total')
        if date_raw is None or total_str is None:
            continue
        try:
            total_usd = float(total_str)
        except (TypeError, ValueError):
            continue
        rows.append({
            'Date': _format_date(date_raw),
            'Value': total_usd,
        })

    rows.sort(key=lambda r: r['Date'])

    return rows


def save_rows_to_csv(rows: List[Dict[str, Any]], csv_path: str | Path) -> None:
    """Persist rows (already enriched or raw) to a CSV file."""
    if not rows:
        return

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _coerce_value(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    try:
        return float(value)
    except ValueError:
        return value


def load_rows_from_csv(csv_path: str | Path) -> List[Dict[str, Any]]:
    """Load rows previously exported via save_rows_to_csv."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            normalized: Dict[str, Any] = {}
            for key, value in row.items():
                if value is None:
                    normalized[key] = ""
                elif key == "Date":
                    normalized[key] = value.strip()
                else:
                    normalized[key] = _coerce_value(value)
            rows.append(normalized)
    rows.sort(key=lambda r: r.get("Date", ""))
    return rows

def add_analysis(rows):
    """Given sorted rows with 'total_usd', add daily delta and change rate.

    Produces new rows with columns:
    - date
    - total_usd
    - delta_usd: day-over-day change in USD
    - change_rate_pct: percentage change vs previous day (0-100 scale)
    """
    if not rows:
        return []

    enriched = []
    prev_total_usd = None
    for r in rows:
        total_usd = r.get('Value')

        if isinstance(total_usd, (int, float)) and prev_total_usd is not None:
            delta_usd = total_usd - prev_total_usd
            if prev_total_usd != 0:
                change_rate_pct = (delta_usd / prev_total_usd)
            else:
                change_rate_pct = 0.0
        else:
            delta_usd = ''
            change_rate_pct = ''

        enriched.append({
            'date': r.get('Date'),
            'total_usd': total_usd,
            'delta_usd': delta_usd,
            'total_cny': USD(total_usd) if total_usd else total_usd,
            'delta_cny': USD(delta_usd) if delta_usd else delta_usd,
            'change_rate_pct': change_rate_pct,
        })

        if isinstance(total_usd, (int, float)):
            prev_total_usd = total_usd

    return enriched

def cli_parser():
    parser = argparse.ArgumentParser(description="Analyze IBKR reports and export to CSV.")
    parser.add_argument(
        "--code-list", 
        action="store_true", 
        help="List available stock codes in data/STOCK."
    )
    
    parser.add_argument(
        "--code",
        "-c",
        help="Stock code to download (e.g., VBTLX).",
    )
    parser.add_argument(
        "--file",
        default=str(data_path("ibkr", "primary.csv")),
        help="Path to a CSV file previously exported via this script.",
    )
    
    parser.add_argument(
        "--print",
        "-p",
        type=int,
        help="print how many rows of the table to stdout after processing.",
        default=10,
    )

    parser.add_argument(
        "--analysis",
        "-a",
        action="store_true",
        help="Do analysis.",
    )

    args = parser.parse_args()
    return args

def analysis(args=None):
    if args is None:
        args = cli_parser()
        
    if args.code_list:
        stock_dir = data_path("STOCK")
        codes = [p.stem for p in stock_dir.glob("*.csv")]
        print("Available stock codes:")
        for code in codes:
            print(f" - {code}")
        return

    if args.code:
        rows = load_rows_from_csv(data_path("STOCK", f"{args.code}.csv"))
    elif args.file.endswith('.csv'):
        rows = load_rows_from_csv(args.file)
    elif args.file.endswith('.xml'):
        rows = parse_account(args.file)
        save_rows_to_csv(rows, str(data_path("ibkr", "primary.csv")))
    else:
        raise ValueError("Input file must be .csv or .xml")
    
    original_rows = rows.copy()

    if args.analysis:
        rows = add_analysis(rows)

    if rows and args.print != 0:
        assert len(rows) >= args.print, f"Not enough rows to print, maximum {len(rows)}."
        formats = {col:".2%" for col in columns_name_of_rows(rows) if "pct" in col.lower() or "percent" in col.lower()}

        print_table(rows[-args.print:], formats)

        print("\nSummary:")
        print_table(add_analysis([original_rows[-args.print]] + [original_rows[-1]]), formats)
        
if __name__ == "__main__":
    analysis()
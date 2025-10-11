import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

from utils import USD, print_table
import xml.etree.ElementTree as ET

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
            'date': _format_date(date_raw),
            'total_usd': total_usd,
        })

    rows.sort(key=lambda r: r['date'])

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
                elif key == "date":
                    normalized[key] = value.strip()
                else:
                    normalized[key] = _coerce_value(value)
            rows.append(normalized)
    rows.sort(key=lambda r: r.get("date", ""))
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
        total_usd = r.get('total_usd')

        if isinstance(total_usd, (int, float)) and prev_total_usd is not None:
            delta_usd = total_usd - prev_total_usd
            if prev_total_usd != 0:
                change_rate_pct = (delta_usd / prev_total_usd) * 100.0
            else:
                change_rate_pct = 0.0
        else:
            delta_usd = ''
            change_rate_pct = ''

        enriched.append({
            'date': r.get('date'),
            'total_usd': total_usd,
            'delta_usd': delta_usd,
            'total_cny': USD(total_usd) if total_usd else total_usd,
            'delta_cny': USD(delta_usd) if delta_usd else delta_usd,
            'change_rate_pct': change_rate_pct,
        })

        if isinstance(total_usd, (int, float)):
            prev_total_usd = total_usd

    return enriched

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze IBKR reports and export to CSV.")
    parser.add_argument(
        "--file",
        default="data/ibkr/primary_last_365_analysis.csv",
        help="Path to a CSV file previously exported via this script.",
    )
    parser.add_argument(
        "--to-csv",
        help="Optional path to save the computed rows to a CSV file.",
    )
    parser.add_argument(
        "--to-csv-analysis",
        help="Add analysis columns (delta, change rate) to the output.",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Do not print the table to stdout after processing.",
    )

    args = parser.parse_args()

    if args.file.endswith('.csv'):
        rows = load_rows_from_csv(args.file)
    elif args.file.endswith('.xml'):
        rows = parse_account(args.file)
    else:
        raise ValueError("Input file must be .csv or .xml")

    if args.to_csv:
        save_rows_to_csv(rows, args.to_csv)

    if args.to_csv_analysis:
        analysis_rows = add_analysis(rows)
        save_rows_to_csv(analysis_rows, args.to_csv_analysis)

    if rows and not args.no_print:
        print_table(add_analysis(rows[-40:]))

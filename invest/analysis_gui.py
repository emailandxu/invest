import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pyqtgraph as pg
try:
    from pyqtgraph.graphicsItems.DateAxisItem import DateAxisItem as PGDateAxisItem
except Exception:  # fallback for older pyqtgraph
    PGDateAxisItem = None
from PySide6.QtCore import Qt, QTimer, QDate
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSlider,
    QDateEdit,
    QInputDialog,
)

from ._paths import data_path
from .read_data import read_data


# ---- Data helpers ----

def list_asset_codes() -> List[str]:
    stock_dir = data_path("STOCK")
    if not stock_dir.exists():
        return ["portfolio"]
    codes = sorted(p.stem for p in stock_dir.glob("*.csv") if not p.stem.startswith("."))
    return ["portfolio"] + [c for c in codes if c]


def _load_daily_series_from_csv(csv_path: Path, date_keys=("Date", "date"), value_keys=("Value", "total_usd", "total")) -> Tuple[np.ndarray, np.ndarray]:
    """Load a flexible Date/Value CSV into numeric arrays.

    Accepts alternate header names (case-insensitive) for date and value columns.
    Returns (timestamps_seconds, values).
    """
    headers, index, rows = read_data(csv_path)
    if not rows:
        return np.array([]), np.array([])

    # Build case-insensitive index lookup
    lower_index: Dict[str, int] = {k.lower(): v for k, v in index.items()}
    date_idx: Optional[int] = None
    value_idx: Optional[int] = None
    for k in date_keys:
        if k.lower() in lower_index:
            date_idx = lower_index[k.lower()]
            break
    for k in value_keys:
        if k.lower() in lower_index:
            value_idx = lower_index[k.lower()]
            break
    if date_idx is None or value_idx is None:
        return np.array([]), np.array([])

    xs: List[float] = []
    ys: List[float] = []
    for row in rows:
        if date_idx >= len(row) or value_idx >= len(row):
            continue
        date_str = row[date_idx]
        value_str = row[value_idx]
        try:
            year, month, day = map(int, (date_str[0:4], date_str[5:7], date_str[8:10]))
        except Exception:
            continue
        try:
            value = float(value_str)
        except Exception:
            continue
        ts = np.datetime64(f"{year:04d}-{month:02d}-{day:02d}", 's').astype('datetime64[s]').astype(np.int64)
        xs.append(int(ts))
        ys.append(value)
    if not xs:
        return np.array([]), np.array([])
    return np.array(xs, dtype=np.int64), np.array(ys, dtype=float)


def load_stock_daily_series(code: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load daily Date,Value series for a stock code from data/STOCK/<code>.csv.

    Returns two numpy arrays (timestamps_in_seconds, values).
    """
    csv_path = data_path("STOCK", f"{code}.csv")
    return _load_daily_series_from_csv(csv_path)


def load_portfolio_ibkr_series() -> Tuple[np.ndarray, np.ndarray]:
    """Load daily portfolio value from IBKR CSV at data/ibkr/primary.csv."""
    csv_path = data_path("ibkr", "primary.csv")
    return _load_daily_series_from_csv(csv_path, date_keys=("date", "Date"), value_keys=("total_usd", "Value", "total"))


def normalize_series(y: np.ndarray, base: float | None = None, scale: float = 100.0) -> np.ndarray:
    if y.size == 0:
        return y
    eps = 1e-12
    if base is None or abs(base) <= eps:
        # find first non-near-zero value
        idxs = np.where(np.abs(y) > eps)[0]
        if idxs.size == 0:
            return y
        base_val = float(y[int(idxs[0])])
    else:
        base_val = float(base)
    return y / base_val * scale


def daily_change_rate(y: np.ndarray) -> np.ndarray:
    if y.size == 0:
        return y
    # pct change, same length (leading NaN set to 0)
    dy = np.empty_like(y, dtype=float)
    dy[0] = 0.0
    prev = y[:-1]
    cur = y[1:]
    with np.errstate(divide='ignore', invalid='ignore'):
        pct = (cur - prev) / prev
    pct = np.nan_to_num(pct, nan=0.0, posinf=0.0, neginf=0.0)
    dy[1:] = pct
    return dy


def _series_year_bounds(x: np.ndarray) -> Optional[Tuple[int, int]]:
    if x.size == 0:
        return None
    secs = x.astype(np.int64)
    years = secs.astype('datetime64[s]').astype('datetime64[Y]').astype(int) + 1970
    return int(years.min()), int(years.max())

def _series_date_bounds(x: np.ndarray) -> Optional[Tuple[Tuple[int,int,int], Tuple[int,int,int]]]:
    if x.size == 0:
        return None
    secs = x.astype(np.int64)
    dmin = np.datetime64(int(secs.min()), 's').astype('datetime64[D]')
    dmax = np.datetime64(int(secs.max()), 's').astype('datetime64[D]')
    smin = str(dmin)
    smax = str(dmax)
    try:
        y1,m1,d1 = map(int, smin.split('-'))
        y2,m2,d2 = map(int, smax.split('-'))
    except Exception:
        return None
    return (y1,m1,d1), (y2,m2,d2)


# ---- UI Widgets ----

class AnalysisPlotPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._setup_ui()
        self._normalized = True

    def _setup_ui(self):
        layout = QGridLayout(self)

        # Time-aware X axis
        axis_items = {}
        if PGDateAxisItem is not None:
            axis_items = {'bottom': PGDateAxisItem(orientation='bottom')}

        self.plot_value = pg.PlotWidget(title="Value Over Time", axisItems=axis_items or None)
        self.plot_value.setLabel('left', 'Value')
        self.plot_value.showGrid(x=True, y=True, alpha=0.3)
        self.plot_value.addLegend()

        axis_items2 = {}
        if PGDateAxisItem is not None:
            axis_items2 = {'bottom': PGDateAxisItem(orientation='bottom')}
        self.plot_change = pg.PlotWidget(title="Daily Change Rate", axisItems=axis_items2 or None)
        self.plot_change.setLabel('left', 'Rate')
        self.plot_change.showGrid(x=True, y=True, alpha=0.3)
        self.plot_change.addLegend()

        layout.addWidget(self.plot_value, 0, 0)
        layout.addWidget(self.plot_change, 1, 0)

    def set_normalized(self, normalized: bool):
        self._normalized = bool(normalized)
        if self._normalized:
            self.plot_value.setLabel('left', 'Index (100 = start)')
        else:
            self.plot_value.setLabel('left', 'Value')

    def clear(self):
        self.plot_value.clear()
        self.plot_change.clear()

    def add_series(self, x: np.ndarray, y: np.ndarray, color: str, label: str, normalize: bool):
        if x.size == 0 or y.size == 0:
            # show hint only if no data at all
            if len(self.plot_value.listDataItems()) == 0:
                txt = pg.TextItem("No data", anchor=(0.5, 0.5), color='w')
                self.plot_value.addItem(txt)
            return
        y_plot = normalize_series(y) if normalize else y
        pen = pg.mkPen(color=color, width=2)
        self.plot_value.plot(x, y_plot, pen=pen, name=label)

        # Avoid flat-line auto-range collapse
        try:
            ymin = float(np.nanmin(y_plot))
            ymax = float(np.nanmax(y_plot))
            if ymin == ymax:
                pad = 1.0 if ymin == 0.0 else abs(ymin) * 0.05
                self.plot_value.setYRange(ymin - pad, ymax + pad, padding=0.0)
        except Exception:
            pass

        r = daily_change_rate(y)
        pen2 = pg.mkPen(color=color, width=1)
        self.plot_change.plot(x, r, pen=pen2, name=f"{label} : {r.std():.2%}")
        try:
            rmin = float(np.nanmin(r))
            rmax = float(np.nanmax(r))
            if rmin == rmax:
                pad = 0.01
                self.plot_change.setYRange(rmin - pad, rmax + pad, padding=0.0)
        except Exception:
            pass


class AnalysisControlPanel(QWidget):
    def __init__(self, plot_panel: AnalysisPlotPanel, on_change=None):
        super().__init__()
        self.plot_panel = plot_panel
        self.on_change = on_change
        self._update_timer = QTimer(self)
        self._update_timer.setInterval(150)
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._emit_changed)

        self._min_year = 1990
        self._max_year = 2030
        self._min_date = QDate(1990,1,1)
        self._max_date = QDate(2030,12,31)
        self._build_ui()

    def _build_ui(self):
        CONTROL_PANEL_WIDTH = 260
        self.setFixedWidth(CONTROL_PANEL_WIDTH)

        layout = QVBoxLayout(self)

        # Series selectors
        codes = list_asset_codes()
        grid = QGridLayout()
        row = 0

        self.combo_main = QComboBox(); self.combo_main.addItems(codes)
        self.combo_main.currentTextChanged.connect(self._queue_update)
        grid.addWidget(QLabel("Red (main):"), row, 0); grid.addWidget(self.combo_main, row, 1); row += 1

        # Additional comparison slots
        self._compare_specs = [
            ("Yellow:", "yellow"),
            ("Green:", "green"),
            ("Blue:", "blue"),
            ("Magenta:", "magenta"),
            ("Cyan:", "cyan"),
            ("Orange:", "orange"),
            ("Pink", "pink")        ]
        self.compare_cbs: List[QComboBox] = []
        self.compare_weights: List[float] = [0.0] * len(self._compare_specs)
        self.weight_buttons: List[QPushButton] = []
        for idx, (label_text, _color) in enumerate(self._compare_specs):
            cb = QComboBox(); cb.addItems(["None"] + codes)
            cb.currentTextChanged.connect(self._queue_update)
            self.compare_cbs.append(cb)
            grid.addWidget(QLabel(label_text), row, 0)
            grid.addWidget(cb, row, 1)

            btn = QPushButton(f"w={self.compare_weights[idx]:.2f}")
            btn.clicked.connect(lambda _=False, i=idx: self._set_weight(i))
            self.weight_buttons.append(btn)
            grid.addWidget(btn, row, 2)

            row += 1

        layout.addLayout(grid)

        # Date range pickers
        dates_grid = QGridLayout()
        dates_row = 0
        dates_grid.addWidget(QLabel("Start Date:"), dates_row, 0)
        self.edit_start = QDateEdit(calendarPopup=True)
        self.edit_start.dateChanged.connect(self._queue_update)
        dates_grid.addWidget(self.edit_start, dates_row, 1); dates_row += 1

        dates_grid.addWidget(QLabel("End Date:"), dates_row, 0)
        self.edit_end = QDateEdit(calendarPopup=True)
        self.edit_end.dateChanged.connect(self._queue_update)
        self.edit_end.setDate(QDate.currentDate())
        dates_grid.addWidget(self.edit_end, dates_row, 1); dates_row += 1
        layout.addLayout(dates_grid)

        # Options
        self.normalize_checkbox = QCheckBox("Normalize to 100")
        self.normalize_checkbox.setChecked(True)
        self.normalize_checkbox.stateChanged.connect(self._queue_update)
        layout.addWidget(self.normalize_checkbox)

        # Utilities
        btn_row = QHBoxLayout()
        btn_reload_codes = QPushButton("Reload Codes")
        btn_reload_codes.clicked.connect(self._reload_codes)
        btn_row.addWidget(btn_reload_codes)
        layout.addLayout(btn_row)

        layout.addStretch(1)

        # Initialize date bounds based on current data
        self._refresh_year_bounds()
        self._refresh_date_bounds()

    def _refresh_year_bounds(self):
        # Consider only selected series (main + active compares)
        selected = [self.combo_main.currentText()]
        for cb in getattr(self, 'compare_cbs', []):
            if cb.currentText() and cb.currentText() != 'None':
                selected.append(cb.currentText())

        min_y = None
        max_y = None
        for code in selected:
            xs, ys = (load_portfolio_ibkr_series() if code == 'portfolio' else load_stock_daily_series(code))
            b = _series_year_bounds(xs)
            if not b:
                continue
            a, z = b
            min_y = a if min_y is None else min(min_y, a)
            max_y = z if max_y is None else max(max_y, z)
        if min_y is None or max_y is None:
            # Fallback to scanning all codes if selected ones have no data
            for code in list_asset_codes():
                xs, ys = (load_portfolio_ibkr_series() if code == 'portfolio' else load_stock_daily_series(code))
                b = _series_year_bounds(xs)
                if not b:
                    continue
                a, z = b
                min_y = a if min_y is None else min(min_y, a)
                max_y = z if max_y is None else max(max_y, z)
        if min_y is None or max_y is None:
            min_y, max_y = 1990, 2030
        self._min_year, self._max_year = min_y, max_y

    def _refresh_date_bounds(self):
        # Compute precise min/max date across current selections
        selected = [self.combo_main.currentText()]
        for cb in getattr(self, 'compare_cbs', []):
            if cb.currentText() and cb.currentText() != 'None':
                selected.append(cb.currentText())

        min_d: Optional[Tuple[int,int,int]] = None
        max_d: Optional[Tuple[int,int,int]] = None
        for code in selected:
            xs, ys = (load_portfolio_ibkr_series() if code == 'portfolio' else load_stock_daily_series(code))
            b = _series_date_bounds(xs)
            if not b:
                continue
            a, z = b
            if min_d is None or a < min_d:
                min_d = a
            if max_d is None or z > max_d:
                max_d = z

        if min_d is None or max_d is None:
            min_d = (1990,1,1); max_d = (2030,12,31)
        self._min_date = QDate(*min_d)
        self._max_date = QDate(*max_d)

        # Configure date edits preserving selection when possible
        self.edit_start.setMinimumDate(self._min_date)
        self.edit_start.setMaximumDate(self._max_date)
        self.edit_end.setMinimumDate(self._min_date)
        self.edit_end.setMaximumDate(self._max_date)

        if not self.edit_start.date().isValid():
            self.edit_start.setDate(self._min_date)
        else:
            if self.edit_start.date() < self._min_date:
                self.edit_start.setDate(self._min_date)
            if self.edit_start.date() > self._max_date:
                self.edit_start.setDate(self._max_date)
        desired_end = QDate.currentDate()
        if desired_end > self._max_date:
            desired_end = self._max_date
        if not self.edit_end.date().isValid():
            self.edit_end.setDate(desired_end)
        else:
            if self.edit_end.date() > self._max_date:
                self.edit_end.setDate(self._max_date)
            elif self.edit_end.date() < self._min_date:
                # If current end is outside low bound (e.g., default far in past), move to today/max
                self.edit_end.setDate(desired_end)

    def _reload_codes(self):
        codes = list_asset_codes()
        # Save selections
        cur_main = self.combo_main.currentText()
        cur_comps = [cb.currentText() for cb in self.compare_cbs]

        self.combo_main.blockSignals(True)
        for cb in self.compare_cbs:
            cb.blockSignals(True)
        self.combo_main.clear(); self.combo_main.addItems(codes)
        for cb in self.compare_cbs:
            cb.clear(); cb.addItems(["None"] + codes)

        # Restore if possible
        def set_current(cb: QComboBox, text: str, fallback: str):
            idx = cb.findText(text)
            if idx >= 0:
                cb.setCurrentIndex(idx)
            else:
                cb.setCurrentText(fallback)

        set_current(self.combo_main, cur_main, codes[0] if codes else "portfolio")
        for cb, txt in zip(self.compare_cbs, cur_comps):
            set_current(cb, txt, "None")

        self.combo_main.blockSignals(False)
        for cb in self.compare_cbs:
            cb.blockSignals(False)
        self._refresh_year_bounds()
        self._emit_changed()

    def _queue_update(self):
        self._refresh_year_bounds()
        self._refresh_date_bounds()
        self._update_timer.start()

    def _set_weight(self, idx: int):
        try:
            current = float(self.compare_weights[idx])
        except Exception:
            current = 0.0
        val, ok = QInputDialog.getDouble(self, "Set Weight", "Enter weight (float):", current, -1e9, 1e9, 4)
        if ok:
            self.compare_weights[idx] = float(val)
            if 0 <= idx < len(self.weight_buttons):
                self.weight_buttons[idx].setText(f"w={val:.2f}")
            self._queue_update()

    def _emit_changed(self):
        if self.on_change:
            self.on_change()

    def get_settings(self):
        # Ensure start <= end
        if self.edit_end.date() < self.edit_start.date():
            self.edit_end.setDate(self.edit_start.date())
        s = self.edit_start.date(); e = self.edit_end.date()
        return {
            "main": self.combo_main.currentText(),
            "compares": [cb.currentText() for cb in self.compare_cbs if cb.currentText() and cb.currentText() != 'None'],
            "normalize": self.normalize_checkbox.isChecked(),
            "start_date": (s.year(), s.month(), s.day()),
            "end_date": (e.year(), e.month(), e.day()),
        }


class AnalysisGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analysis (Daily)")
        self.setGeometry(100, 100, 1100, 800)

        main = QWidget(); self.setCentralWidget(main)
        h = QHBoxLayout(main)

        self.plot_panel = AnalysisPlotPanel()
        self.control = AnalysisControlPanel(self.plot_panel, on_change=self.update)
        h.addWidget(self.control)
        h.addWidget(self.plot_panel, stretch=1)

        self.update()

    def _resolve_code_series(self, code: str) -> Tuple[np.ndarray, np.ndarray]:
        if code == "portfolio":
            return load_portfolio_ibkr_series()
        return load_stock_daily_series(code)

    @staticmethod
    def _filter_dates(x: np.ndarray, y: np.ndarray, start_ymd: Tuple[int,int,int], end_ymd: Tuple[int,int,int]) -> Tuple[np.ndarray, np.ndarray]:
        if x.size == 0:
            return x, y
        sy,sm,sd = start_ymd
        ey,em,ed = end_ymd
        start_ts = np.datetime64(f"{sy:04d}-{sm:02d}-{sd:02d}", 's').astype('datetime64[s]').astype(np.int64)
        # End inclusive: use next day minus 1 second
        end_next = np.datetime64(f"{ey:04d}-{em:02d}-{ed:02d}", 'D') + np.timedelta64(1,'D')
        end_ts = end_next.astype('datetime64[s]').astype(np.int64) - 1
        secs = x.astype(np.int64)
        mask = (secs >= start_ts) & (secs <= end_ts)
        if not np.any(mask):  # fallback if empty
            return x, y
        return x[mask], y[mask]

    def update(self):
        s = self.control.get_settings()
        self.plot_panel.clear()

        normalize = s["normalize"]
        self.plot_panel.set_normalized(normalize)
        start_ymd = s["start_date"]
        end_ymd = s["end_date"]

        # Main
        x, y = self._resolve_code_series(s["main"]) 
        x, y = self._filter_dates(x, y, start_ymd, end_ymd)
        self.plot_panel.add_series(x, y, color="red", label=s["main"], normalize=normalize)

        # Additional comparisons: keep color fixed per slot even if earlier slots are None
        for (label_text, color), cb in zip(self.control._compare_specs, self.control.compare_cbs):
            code = cb.currentText()
            if not code or code == "None":
                continue
            x, y = self._resolve_code_series(code)
            x, y = self._filter_dates(x, y, start_ymd, end_ymd)
            self.plot_panel.add_series(x, y, color=color, label=code, normalize=normalize)

        self._plot_weighted_composite(start_ymd, end_ymd)

    def _plot_weighted_composite(self, start_ymd: Tuple[int,int,int], end_ymd: Tuple[int,int,int]):
        series_x: List[np.ndarray] = []
        series_r: List[np.ndarray] = []
        series_w: List[float] = []
        for i, ((_, _color), cb) in enumerate(zip(self.control._compare_specs, self.control.compare_cbs)):
            code = cb.currentText()
            w = self.control.compare_weights[i] if i < len(self.control.compare_weights) else 0.0
            if not code or code == "None" or abs(w) <= 0.0:
                continue
            x, y = self._resolve_code_series(code)
            x, y = self._filter_dates(x, y, start_ymd, end_ymd)
            if x.size == 0 or y.size == 0:
                continue
            r = daily_change_rate(y)
            series_x.append(x.astype(np.int64))
            series_r.append(r.astype(float))
            series_w.append(float(w))

        if not series_x:
            return

        common_x = series_x[0]
        for x in series_x[1:]:
            common_x = np.intersect1d(common_x, x, assume_unique=False)
            if common_x.size == 0:
                return

        weighted_r = np.zeros_like(common_x, dtype=float)
        for x, r, w in zip(series_x, series_r, series_w):
            inter, idx_common, idx_in_series = np.intersect1d(common_x, x, return_indices=True)
            if inter.size != common_x.size:
                continue
            r_aligned = r[idx_in_series]
            weighted_r += w * r_aligned

        V = 100.0 * np.cumprod(1.0 + weighted_r)
        try:
            pen = pg.mkPen(color=(150, 150, 150), width=2)
            pen2 = pg.mkPen(color=(150, 150, 150), width=1)
        except Exception:
            pen = None
            pen2 = None
        self.plot_panel.plot_value.plot(common_x, V, pen=pen, name="Weighted")
        self.plot_panel.plot_change.plot(common_x, weighted_r, pen=pen2, name=f"Weighted: {weighted_r.std():.2%}")
        try:
            rmin = float(np.nanmin(weighted_r))
            rmax = float(np.nanmax(weighted_r))
            if rmin == rmax:
                pad = 0.01
                self.plot_panel.plot_change.setYRange(rmin - pad, rmax + pad, padding=0.0)
        except Exception:
            pass

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Escape, Qt.Key_Q):
            QApplication.instance().quit()
        super().keyPressEvent(event)

    @classmethod
    def main(cls, args=None):
        if args is None:
            args = sys.argv
        app = QApplication(args)
        w = cls()
        w.show()
        app.exec_()


def main(args=None):
    AnalysisGui.main(args)


if __name__ == "__main__":
    main()

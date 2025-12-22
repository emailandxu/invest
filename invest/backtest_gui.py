"""
Backtest GUI - Standalone window for running backtests.

This module provides a Qt-based GUI (PySide6) for:
- Selecting and configuring trading strategies
- Running backtests with customizable parameters  
- Visualizing equity curves and performance metrics
"""

import sys
from datetime import date, datetime
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import Qt, QDate, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QComboBox, QPushButton, QTextEdit,
    QSpinBox, QDoubleSpinBox, QDateEdit, QListWidget, QListWidgetItem,
    QAbstractItemView, QSplitter, QGroupBox, QCheckBox, QMessageBox,
    QDialog, QFormLayout, QDialogButtonBox, QToolButton,
    QTableWidget, QHeaderView, QTableWidgetItem, QLineEdit
)

from .backtest import Backtester, BacktestResult, StrategyRegistry
from ._paths import data_path

# Import strategies to ensure they are registered
from . import strategies
from dataclasses import dataclass, field

@dataclass
class PinnedStrategy:
    """Represents a pinned strategy configuration for comparison."""
    name: str
    strategy_name: str
    strategy_params: dict
    assets: List[str]
    allocation: dict
    note: str = ""
    # Optional: cache result if parameters haven't changed? 
    # For now we re-run to ensure Apple-to-Apple with global settings.


class BacktestControlPanel(QWidget):
    """Control panel for backtest configuration."""
    
    def __init__(self, on_run_backtest=None):
        super().__init__()
        self.on_run_backtest = on_run_backtest
        self.pinned_strategies: List[PinnedStrategy] = []
        self._setup_ui()
        self._load_initial_portfolios()

    def _load_initial_portfolios(self):
        """Load saved portfolios from disk."""
        from .portfolio_manager import PortfolioManager
        configs = PortfolioManager.load_all()
        
        self.pinned_strategies.clear()
        for _, config in configs.items():
            pinned = PinnedStrategy(
                name=config.name,
                strategy_name=config.strategy_name,
                strategy_params=config.strategy_params,
                assets=config.assets,
                allocation=config.allocation
            )
            self.pinned_strategies.append(pinned)
            
        self._refresh_comp_list()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # --- PORTFOLIOS (Managed in Dialog) ---
        port_group = QGroupBox("Portfolios")
        port_layout = QVBoxLayout()
        
        self.comp_list = QListWidget()
        self.comp_list.setMaximumHeight(100)
        self.comp_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.comp_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.comp_list.customContextMenuRequested.connect(self._show_context_menu)
        port_layout.addWidget(self.comp_list)
        
        btn_layout = QHBoxLayout()
        self.manage_comp_btn = QPushButton("Manage Portfolios...")
        self.manage_comp_btn.clicked.connect(self._on_manage_comparison)
        btn_layout.addWidget(self.manage_comp_btn)
        
        port_layout.addLayout(btn_layout)
        port_group.setLayout(port_layout)
        layout.addWidget(port_group)

        # --- GLOBAL SETTINGS ---
        settings_group = QGroupBox("Global Settings")
        settings_layout = QGridLayout(settings_group)
        
        # Date range
        settings_layout.addWidget(QLabel("Start Date:"), 0, 0)
        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(QDate(2010, 1, 1))
        self.start_date.setDisplayFormat("yyyy-MM-dd")
        settings_layout.addWidget(self.start_date, 0, 1)
        
        settings_layout.addWidget(QLabel("End Date:"), 1, 0)
        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setDisplayFormat("yyyy-MM-dd")
        settings_layout.addWidget(self.end_date, 1, 1)
        
        # Capital
        settings_layout.addWidget(QLabel("Capital ($):"), 2, 0)
        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(1000, 100000000)
        self.capital_spin.setValue(100000)
        self.capital_spin.setPrefix("$ ")
        self.capital_spin.setSingleStep(10000)
        settings_layout.addWidget(self.capital_spin, 2, 1)
        
        layout.addWidget(settings_group)
        
        # --- WITHDRAWAL ---
        withdrawal_group = QGroupBox("Periodic Withdrawal")
        withdrawal_layout = QGridLayout(withdrawal_group)
        
        withdrawal_layout.addWidget(QLabel("Amount:"), 0, 0)
        self.withdrawal_amount = QDoubleSpinBox()
        self.withdrawal_amount.setRange(0, 1000000)
        self.withdrawal_amount.setValue(0)
        self.withdrawal_amount.setPrefix("$ ")
        self.withdrawal_amount.setSingleStep(100)
        withdrawal_layout.addWidget(self.withdrawal_amount, 0, 1)
        
        withdrawal_layout.addWidget(QLabel("Period (Days):"), 1, 0)
        self.withdrawal_period = QSpinBox()
        self.withdrawal_period.setRange(1, 365)
        self.withdrawal_period.setValue(30)
        withdrawal_layout.addWidget(self.withdrawal_period, 1, 1)
        
        withdrawal_layout.addWidget(QLabel("Method:"), 2, 0)
        self.withdrawal_method = QComboBox()
        self.withdrawal_method.addItems([
            "Proportional",
            "Rebalance",
            "Sell Winners",
            "Sell Losers"
        ])
        withdrawal_layout.addWidget(self.withdrawal_method, 2, 1)
        
        self.adjust_inflation = QCheckBox("Adjust for US Inflation")
        self.adjust_inflation.setChecked(True)
        self.adjust_inflation.setToolTip("Withdrawal amount increases with historical US inflation")
        withdrawal_layout.addWidget(self.adjust_inflation, 3, 0, 1, 2)
        
        layout.addWidget(withdrawal_group)
        
        
        # Run button
        self.run_btn = QPushButton("▶ Run Backtest")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.run_btn.clicked.connect(self._on_run_clicked)
        layout.addWidget(self.run_btn)
        
        # Show report button
        self.report_btn = QPushButton("📊 Show Report")
        self.report_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        layout.addWidget(self.report_btn)
        
        layout.addStretch()
        
        layout.addStretch()
        
        # Loaded in _load_initial_portfolios called from __init__
    
    # --- Obsolete methods removed ---
    
    def _on_run_clicked(self):
        if self.on_run_backtest:
            self.on_run_backtest()
    
    # Removed get_selected_strategy_name, get_selected_assets
    
    def get_start_date(self) -> date:
        qdate = self.start_date.date()
        return date(qdate.year(), qdate.month(), qdate.day())
    
    def get_end_date(self) -> date:
        qdate = self.end_date.date()
        return date(qdate.year(), qdate.month(), qdate.day())
    
    def get_initial_capital(self) -> float:
        return self.capital_spin.value()
    
    
    # Removed _on_pin_current
    def _refresh_comp_list(self):
        """Update the list widget."""
        self.comp_list.clear()
        for i, p in enumerate(self.pinned_strategies):
            item = QListWidgetItem(p.name)
            # Only check the first one by default
            state = Qt.Checked if i == 0 else Qt.Unchecked
            item.setCheckState(state) 
            self.comp_list.addItem(item)
            
    def _on_manage_comparison(self):
        """Open Portfolio Manager."""
        # Using PortfolioManagerDialog directly (defined in same module)
        
        dialog = PortfolioManagerDialog(self.pinned_strategies, self)
        # We don't check for result because changes are saved to disk/session immediately via Save button in dialog
        # Or if we want to confirm?
        # Let's keep using exec and reload. If user Saves inside, it's persistent. 
        # If they Cancel, we might revert to original list?
        # But if they clicked "Save" to disk, it should be permanent.
        
        dialog.exec()
        # Always reload from what the dialog has (which is updated in-place or from disk)
        self.pinned_strategies = dialog.get_strategies()
        self._refresh_comp_list()

    def _show_context_menu(self, pos):
        menu = QMenu(self)
        
        select_all_action = QAction("Select All", self)
        select_all_action.triggered.connect(self._select_all_portfolios)
        menu.addAction(select_all_action)
        
        select_none_action = QAction("Select None", self)
        select_none_action.triggered.connect(self._select_none_portfolios)
        menu.addAction(select_none_action)
        
        menu.exec(self.comp_list.mapToGlobal(pos))
        
    def _select_all_portfolios(self):
        for i in range(self.comp_list.count()):
            self.comp_list.item(i).setCheckState(Qt.Checked)
            
    def _select_none_portfolios(self):
        for i in range(self.comp_list.count()):
            self.comp_list.item(i).setCheckState(Qt.Unchecked)
            
    def get_active_configs(self) -> List[PinnedStrategy]:
        """Return list of configurations to run (Only Checked Pinned)."""
        configs = []
        for i in range(self.comp_list.count()):
            item = self.comp_list.item(i)
            if item.checkState() == Qt.Checked:
                if i < len(self.pinned_strategies):
                    configs.append(self.pinned_strategies[i])
        return configs
    
    # --- Existing Methods ---
    
    def get_withdrawal_amount(self) -> float:
        return self.withdrawal_amount.value()
    
    def get_withdrawal_period(self) -> int:
        return self.withdrawal_period.value()
    
    def get_withdrawal_method(self) -> str:
        # Convert display name to internal name
        method_map = {
            "Proportional": "proportional",
            "Rebalance": "rebalance",
            "Sell Winners": "sell_winners",
            "Sell Losers": "sell_losers"
        }
        return method_map.get(self.withdrawal_method.currentText(), "proportional")
    
    
    def should_adjust_for_inflation(self) -> bool:
        return self.adjust_inflation.isChecked()
    
    def get_params(self) -> dict:
        """Get current global settings as a dictionary for saving."""
        return {
            'start_date': self.get_start_date().isoformat(),
            'end_date': self.get_end_date().isoformat(),
            'initial_capital': self.capital_spin.value(),
            'withdrawal_amount': self.withdrawal_amount.value(),
            'withdrawal_period': self.withdrawal_period.value(),
            'withdrawal_method': self.withdrawal_method.currentText(),
            'adjust_inflation': self.adjust_inflation.isChecked()
        }
    
    def set_params(self, params: dict) -> None:
        """Restore global settings from a dictionary."""
        if not params:
            return
            
        # Dates
        if 'start_date' in params:
            d = date.fromisoformat(params['start_date'])
            self.start_date.setDate(QDate(d.year, d.month, d.day))
        if 'end_date' in params:
            d = date.fromisoformat(params['end_date'])
            self.end_date.setDate(QDate(d.year, d.month, d.day))
        
        # Capital
        if 'initial_capital' in params:
            self.capital_spin.setValue(params['initial_capital'])
        
        if 'adjust_inflation' in params:
            self.adjust_inflation.setChecked(params['adjust_inflation'])
        
        # Withdrawal
        if 'withdrawal_amount' in params:
            self.withdrawal_amount.setValue(params['withdrawal_amount'])
        if 'withdrawal_period' in params:
            self.withdrawal_period.setValue(params['withdrawal_period'])
        if 'withdrawal_method' in params:
            idx = self.withdrawal_method.findText(params['withdrawal_method'])
            if idx >= 0:
                self.withdrawal_method.setCurrentIndex(idx)


    
    # --- PORTFOLIO MANAGEMENT METHODS ---

    # Removed _load_portfolios_to_combo, _on_portfolio_changed, _on_save_portfolio, _on_delete_portfolio


class PortfolioManagerDialog(QDialog):
    """Dialog to manage portfolios (strategies + configurations)."""
    def __init__(self, strategies: List[PinnedStrategy], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Portfolio Manager")
        self.resize(1000, 700)
        
        # Deep copy to allow cancellation
        import copy
        self.strategies = copy.deepcopy(strategies)
        self.original_strategies = strategies
        
        self.current_idx = -1
        self.is_loading = False
        
        self._setup_ui()
        self._populate_strategies_combo()
        self._load_assets()
        self._refresh_list()
        
    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        
        # --- LEFT PANEL: Portfolio List ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("<b>Portfolios</b>"))
        
        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self._on_selection_changed)
        left_layout.addWidget(self.list_widget)
        
        btn_layout = QGridLayout()
        self.add_btn = QPushButton("New")
        self.add_btn.clicked.connect(self._on_add_new)
        self.del_btn = QPushButton("Delete")
        self.del_btn.clicked.connect(self._on_delete)
        
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self._on_save_disk)
        self.save_btn.setStyleSheet("font-weight: bold; color: #4CAF50;") # Green text
        
        btn_layout.addWidget(self.add_btn, 0, 0)
        btn_layout.addWidget(self.del_btn, 0, 1)
        btn_layout.addWidget(self.save_btn, 1, 0, 1, 2)
        
        left_layout.addLayout(btn_layout)
        main_layout.addWidget(left_panel, 1)
        
        # --- RIGHT PANEL: Editor ---
        right_panel = QGroupBox("Configuration")
        self.right_layout = QVBoxLayout(right_panel)
        
        # 1. Name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.editingFinished.connect(self._save_current_edits)
        name_layout.addWidget(self.name_edit)
        self.right_layout.addLayout(name_layout)
        
        # 2. Strategy Selection
        strat_group = QGroupBox("Strategy Algorithm")
        strat_layout = QVBoxLayout(strat_group)
        self.strategy_combo = QComboBox()
        self.strategy_combo.currentTextChanged.connect(self._on_strategy_changed)
        strat_layout.addWidget(self.strategy_combo)
        self.strategy_desc = QLabel("Description...")
        self.strategy_desc.setWordWrap(True)
        self.strategy_desc.setStyleSheet("color: gray; font-size: 10px;")
        strat_layout.addWidget(self.strategy_desc)
        self.right_layout.addWidget(strat_group)
        
        # 3. Parameters (Dynamic)
        self.params_group = QGroupBox("Parameters")
        self.params_layout = QFormLayout(self.params_group)
        self.param_widgets = {}
        self.right_layout.addWidget(self.params_group)
        
        # 4. Assets & Allocation (Splitter)
        asset_alloc_splitter = QSplitter(Qt.Horizontal)
        
        # Assets List
        asset_group = QGroupBox("Select Assets")
        asset_layout = QVBoxLayout(asset_group)
        self.asset_list = QListWidget()
        self.asset_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.asset_list.itemSelectionChanged.connect(self._on_assets_changed)
        asset_layout.addWidget(self.asset_list)
        asset_alloc_splitter.addWidget(asset_group)
        
        # Allocation Table
        alloc_group = QGroupBox("Allocation (%)")
        alloc_layout = QVBoxLayout(alloc_group)
        self.alloc_table = QTableWidget()
        self.alloc_table.setColumnCount(2)
        self.alloc_table.setHorizontalHeaderLabels(["Asset", "Weight (%)"])
        self.alloc_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.alloc_table.verticalHeader().setVisible(False)
        self.alloc_table.itemChanged.connect(self._on_allocation_item_changed)
        alloc_layout.addWidget(self.alloc_table)
        asset_alloc_splitter.addWidget(alloc_group)
        
        self.right_layout.addWidget(asset_alloc_splitter, 1)
        
        main_layout.addWidget(right_panel, 2)
        
        # --- BOTTOM ---
        # Dialog buttons are usually outside the main HBox, let's restructure slightly
        # Or easier: just add a button box at bottom of left panel or overlay
        # Since we inherit QDialog, we should ideally have a bottom row
        
        # Let's verify layout. QDialog usually needs QVBoxLayout as main.
        # But here I used QHLayout for main content.
        # It's better to wrap everything in a VBox
        
    def showEvent(self, event):
        # Ugly fix for layout since I can't wrap easily in this snippet replacement
        # Use a wrapper layout in a real rewrite. 
        # For now, let's assume the calling code puts this inside a layout or we append to layout
        super().showEvent(event)

    # --- Initializers ---
    def _populate_strategies_combo(self):
        self.strategy_combo.blockSignals(True)
        self.strategy_combo.clear()
        for name in StrategyRegistry.list_strategies():
            self.strategy_combo.addItem(name)
        self.strategy_combo.blockSignals(False)
        
    def _load_assets(self):
        self.asset_list.blockSignals(True)
        self.asset_list.clear()
        stock_dir = data_path("STOCK")
        if stock_dir.exists():
            for csv_file in sorted(stock_dir.glob("*.csv")):
                if csv_file.name.startswith(".") or csv_file.name.endswith("_dividends.csv"): continue
                item = QListWidgetItem(csv_file.stem)
                self.asset_list.addItem(item)
        self.asset_list.blockSignals(False)

    # --- List Management ---
    def _refresh_list(self):
        self.list_widget.blockSignals(True)
        curr = self.list_widget.currentRow()
        self.list_widget.clear()
        for s in self.strategies:
            self.list_widget.addItem(s.name)
        
        # Restore selection or select first
        if self.strategies:
            idx = curr if curr >= 0 and curr < len(self.strategies) else 0
            self.list_widget.setCurrentRow(idx)
            self._on_selection_changed(idx) # Force update editor
        else:
             self._clear_editor()
             self.current_idx = -1
             
        self.list_widget.blockSignals(False)

    def _on_save_disk(self):
        """Save all strategies to disk."""
        from .portfolio_manager import PortfolioManager, PortfolioConfig
        
        # Convert PinnedStrategy objects to PortfolioConfig
        # Note: PinnedStrategy has (name, strategy_name, strategy_params, assets, allocation)
        # PortfolioConfig has (name, strategy_name, assets, allocation, strategy_params, note)
        
        # First load existing to preserve IDs or Notes if any? 
        # For now, we overwrite based on name.
        
        # To handle renames properly, we might need a better unique ID system.
        # But assuming names are unique enough for this simple tool.
        
        # Strategy: Clear all and rewrite? Or merge?
        # Let's rewrite the file with current list to ensure deletes are propagated.
        
        configs = {}
        for s in self.strategies:
            config = PortfolioConfig(
                name=s.name,
                strategy_name=s.strategy_name,
                assets=s.assets,
                allocation=s.allocation,
                strategy_params=s.strategy_params,
                note="" 
            )
            configs[s.name] = config
            
        # Write directly (we need access to Manager's _write_file or use save loop)
        # Manager only has save(one) or delete(one). 
        # Let's add save_all to Manager or loop.
        # Looping save() reads/writes file N times. Inefficient but safe.
        # But we also need to delete ones that are gone.
        
        # It's better to update the whole file.
        # Since PortfolioManager _write_file is semi-private, we can use it or expose explicit save_all logic
        # For now let's reuse what we have or import the class and use the method if visible.
        # Actually in Python everything is visible.
        PortfolioManager._write_file(configs)
        
        QMessageBox.information(self, "Saved", "Portfolios saved to disk.")

    def _on_selection_changed(self, row):
        if row < 0 or row >= len(self.strategies):
            self._clear_editor()
            return

        # Save previous if needed? 
        # Actually we update `strategies[current_idx]` in real-time on edit.
        
        self.current_idx = row
        self.is_loading = True
        self._load_strategy_to_editor(self.strategies[row])
        self.is_loading = False

    def _on_add_new(self):
        # Create default
        new_strat = PinnedStrategy(
            name="New Portfolio",
            strategy_name=StrategyRegistry.list_strategies()[0],
            strategy_params={}, # Defaults will be loaded
            assets=[],
            allocation={}
        )
        self.strategies.append(new_strat)
        self._refresh_list()
        # Select the new one (last)
        self.list_widget.setCurrentRow(len(self.strategies)-1)

    def _on_delete(self):
        row = self.list_widget.currentRow()
        if row >= 0:
            del self.strategies[row]
            self._refresh_list()

    # --- Editor Logic ---
    def _clear_editor(self):
        self.name_edit.clear()
        self.strategy_combo.setCurrentIndex(-1)
        self.asset_list.clearSelection()
        self.alloc_table.setRowCount(0)
        # Clear params
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.param_widgets = {}

    def _load_strategy_to_editor(self, strat: PinnedStrategy):
        self.name_edit.setText(strat.name)
        
        # Set Strategy
        idx = self.strategy_combo.findText(strat.strategy_name)
        if idx >= 0:
            self.strategy_combo.setCurrentIndex(idx)
        else:
            if self.strategy_combo.count() > 0: self.strategy_combo.setCurrentIndex(0)
            
        # Params (generated by _on_strategy_changed, then filled)
        self._update_parameter_panel() # Regenerates widgets
        
        for name, val in strat.strategy_params.items():
            if name in self.param_widgets:
                w = self.param_widgets[name]
                if isinstance(w, QCheckBox): w.setChecked(bool(val))
                elif isinstance(w, (QSpinBox, QDoubleSpinBox)): w.setValue(val)
        
        # Assets
        self.asset_list.blockSignals(True)
        self.asset_list.clearSelection()
        items = [self.asset_list.item(i) for i in range(self.asset_list.count())]
        for item in items:
            if item.text() in strat.assets:
                item.setSelected(True)
        self.asset_list.blockSignals(False)
        
        # Allocation
        self._update_allocation_table_from_data(strat.allocation)

    def _on_strategy_changed(self, text):
        if not text: return
        
        strategy_class = StrategyRegistry.get(text)
        if strategy_class:
            self.strategy_desc.setText(strategy_class.description)
            
        if self.is_loading: return # Just updating UI from load
        
        # If user changed strategy manually, regenerate params with defaults
        if self.current_idx >= 0:
             self.strategies[self.current_idx].strategy_name = text
             self.strategies[self.current_idx].strategy_params = {} # Reset params
             
        self._update_parameter_panel()
        self._save_current_edits() # Save new defaults

    def _update_parameter_panel(self):
        # Clear
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.param_widgets = {}
        
        name = self.strategy_combo.currentText()
        strategy_class = StrategyRegistry.get(name)
        if not strategy_class: return
        
        # Load current values from strategy object if available
        current_vals = {}
        if self.current_idx >= 0:
            current_vals = self.strategies[self.current_idx].strategy_params
            
        for row, param in enumerate(strategy_class.parameters):
            label = QLabel(param.label + ":")
            # QFormLayout.addRow handles the positioning
            # self.params_layout.addWidget(label, row, 0)
            
            val = current_vals.get(param.name, param.default)
            
            widget = None
            if param.param_type == 'int':
                widget = QSpinBox()
                widget.setRange(int(param.min_value or 0), int(param.max_value or 10000))
                widget.setSingleStep(int(param.step))
                widget.setValue(int(val))
                widget.valueChanged.connect(self._save_current_edits)
            elif param.param_type == 'float':
                widget = QDoubleSpinBox()
                widget.setRange(param.min_value or 0, param.max_value or 1000000)
                widget.setSingleStep(param.step)
                widget.setValue(float(val))
                widget.setDecimals(2)
                widget.valueChanged.connect(self._save_current_edits)
            elif param.param_type == 'bool':
                widget = QCheckBox()
                widget.setChecked(bool(val))
                widget.stateChanged.connect(self._save_current_edits)
            
            if widget:
                # self.params_layout.addWidget(widget, row, 1)
                self.params_layout.addRow(label, widget)
                self.param_widgets[param.name] = widget

    def _on_assets_changed(self):
        if self.is_loading or self.current_idx < 0: return
        
        selected = [item.text() for item in self.asset_list.selectedItems()]
        self.strategies[self.current_idx].assets = selected
        
        # Rebuild allocation table (preserve existing weights if possible)
        self._update_allocation_table_from_data(self.strategies[self.current_idx].allocation)
        self._save_current_edits()

    def _update_allocation_table_from_data(self, allocation: dict):
        self.alloc_table.blockSignals(True)
        selected_assets = [item.text() for item in self.asset_list.selectedItems()]
        self.alloc_table.setRowCount(len(selected_assets))
        
        # If allocation keys match selected, use them. Else distribute equal.
        # Check if assets changed
        
        # Calculate weights if missing
        existing_keys = set(allocation.keys())
        current_keys = set(selected_assets)
        
        # If new asset added, or just basic sync.
        # For simplicity: if mismatch, reset to equal weight? Or try to keep?
        # Let's just fill table.
        
        total_rows = len(selected_assets)
        for row, asset in enumerate(selected_assets):
            self.alloc_table.setItem(row, 0, QTableWidgetItem(asset))
            
            weight = allocation.get(asset, 0)
            if asset not in existing_keys and total_rows > 0:
                 # New asset logic could be complex. For now default 0 or let user fix.
                 # Or if allocation was empty, set equal.
                 if not allocation: weight = 1.0 / total_rows
            
            spin = QDoubleSpinBox()
            spin.setRange(0, 100)
            spin.setValue(weight * 100)
            spin.valueChanged.connect(self._save_current_edits)
            self.alloc_table.setCellWidget(row, 1, spin)
            
        self.alloc_table.blockSignals(False)
    
    def _on_allocation_item_changed(self, item):
         # Used if we used items instead of cell widgets
         pass

    def _save_current_edits(self):
        if self.is_loading or self.current_idx < 0: return
        
        strat = self.strategies[self.current_idx]
        strat.name = self.name_edit.text()
        
        # Params
        for name, widget in self.param_widgets.items():
            if isinstance(widget, QCheckBox):
                strat.strategy_params[name] = widget.isChecked()
            else:
                strat.strategy_params[name] = widget.value()
                
        # Allocation (Read from table)
        new_alloc = {}
        for row in range(self.alloc_table.rowCount()):
            asset = self.alloc_table.item(row, 0).text()
            spin = self.alloc_table.cellWidget(row, 1)
            if spin:
                new_alloc[asset] = spin.value() / 100.0
        strat.allocation = new_alloc
        
        # Update list name if changed
        item = self.list_widget.item(self.current_idx)
        if item and item.text() != strat.name:
            item.setText(strat.name)

    def get_strategies(self):
        return self.strategies


class ChartConfigDialog(QDialog):
    """Dialog for configuring chart parameters."""
    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Chart Settings")
        self.params = params.copy()
        self.widgets = {}
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()
        
        for name, value in self.params.items():
            if isinstance(value, int):
                widget = QSpinBox()
                widget.setRange(1, 10000)
                widget.setValue(value)
                self.widgets[name] = widget
                form.addRow(name.replace('_', ' ').title() + ":", widget)
            elif isinstance(value, float):
                widget = QDoubleSpinBox()
                widget.setRange(0.0, 10000.0)
                widget.setValue(value)
                self.widgets[name] = widget
                form.addRow(name.replace('_', ' ').title() + ":", widget)
            # Add more types as needed
            
        layout.addLayout(form)
        
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
        
    def get_params(self) -> dict:
        new_params = {}
        for name, widget in self.widgets.items():
            new_params[name] = widget.value()
        return new_params


class BaseChart(QWidget):
    """Base class for all backtest charts."""
    
    # Shared color cycle for all charts
    COLORS = ['#2196F3', '#4CAF50', '#FFC107', '#E91E63', '#9C27B0', '#00BCD4', 
              '#FF5722', '#795548', '#607D8B', '#3F51B5']

    def __init__(self, title="Chart"):
        super().__init__()
        self.title = title
        self.params = {}  # Default parameters
        self.results = {} # Store current results
        self.kwargs = {}  # Store extra args (vti_prices etc)
        self._setup_ui()
        
    def _setup_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.plot_widget = pg.PlotWidget(title=self.title, axisItems={'bottom': pg.DateAxisItem()})
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('bottom', 'Date')
        self.layout.addWidget(self.plot_widget)
        
        # Hover items
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#888', style=Qt.DashLine))
        self.plot_widget.addItem(self.vline, ignoreBounds=True)
        self.vline.setVisible(False)
        self.vline.setZValue(1000)
        
        self.label = pg.TextItem(anchor=(0, 1), fill=pg.mkBrush('#333'))
        self.plot_widget.addItem(self.label, ignoreBounds=True)
        self.label.setVisible(False)
        self.label.setZValue(1000)
        
        self.plot_widget.scene().sigMouseMoved.connect(self._on_hover)
        
    def update_chart(self, results: Dict[str, BacktestResult], **kwargs):
        self.results = results
        self.kwargs = kwargs
        self.plot_widget.clear()
        
        # Re-add persistent items
        self.plot_widget.addItem(self.vline, ignoreBounds=True)
        self.plot_widget.addItem(self.label, ignoreBounds=True)
        
        # Helper for dates (use first result)
        if results:
            first_res = next(iter(results.values()))
            if first_res.equity_curve:
                dates = [ep.date for ep in first_res.equity_curve]
                self.plot_timestamps = np.array([pd.to_datetime(d).timestamp() for d in dates])
                self.x_data = self.plot_timestamps
        
        self._plot_content(results, **kwargs)

    def _plot_content(self, results, **kwargs):
        raise NotImplementedError
        
    def _on_hover(self, pos):
        # Default implementation, can be overridden
        pass
    
    def get_params(self) -> dict:
        return self.params
        
    def set_params(self, params: dict):
        self.params.update(params)
        # Re-plot if data exists
        if hasattr(self, 'results') and self.results:
            self.update_chart(self.results, **self.kwargs)

    def set_x_link(self, target_plot):
        self.plot_widget.setXLink(target_plot)


class EquityChart(BaseChart):
    def __init__(self):
        super().__init__("Equity Curve")
        self.plot_widget.setLabel('left', 'Equity ($)')
        self.plot_widget.addLegend()

    def _plot_content(self, results, **kwargs):
        for i, (name, result) in enumerate(results.items()):
            if not result.equity_curve: continue
            
            # Choose color
            color = self.COLORS[i % len(self.COLORS)]
            
            # Choose color
            color = self.COLORS[i % len(self.COLORS)]
            
            _, equities = result.get_equity_series()
            min_len = min(len(self.x_data), len(equities))
            
            self.plot_widget.plot(
                self.x_data[:min_len], equities[:min_len],
                pen=pg.mkPen(color=color, width=2),
                name=name
            )

            # Trades (Only for Current to avoid clutter)
            if name == "Current":
                dates = [ep.date for ep in result.equity_curve]
                for trade in result.trades:
                    try:
                        idx = dates.index(trade.date)
                        if idx < len(self.x_data):
                            val = equities[idx]
                            c = '#4CAF50' if trade.side.value == 'buy' else '#F44336'
                            symbol = 't' if trade.side.value == 'buy' else 't1'
                            self.plot_widget.addItem(pg.ScatterPlotItem(
                                [self.x_data[idx]], [val], symbol=symbol, size=8, brush=pg.mkBrush(c)
                            ))
                    except (ValueError, IndexError):
                        pass


    def _on_hover(self, pos):
        if not hasattr(self, 'results') or not hasattr(self, 'x_data'): return
        if not self.plot_widget.sceneBoundingRect().contains(pos):
             self.vline.setVisible(False); self.label.setVisible(False); return
             
        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
        idx = self._get_nearest_index(mouse_point.x())
        if idx is None: return
        
        x_val = self.x_data[idx]
        date_str = pd.to_datetime(self.x_data[idx], unit='s').strftime('%Y-%m-%d')
        
        html = f'<div style="color: white; font-size: 10pt;"><b>📅 {date_str}</b><br>'
        
        for i, (name, result) in enumerate(self.results.items()):
            if not result.equity_curve or idx >= len(result.equity_curve): continue
            ep = result.equity_curve[idx]
            ret = (ep.equity / result.initial_capital - 1) * 100
            color = self.COLORS[i % len(self.COLORS)]
            html += f'<span style="color: {color};">●</span> {name}: ${ep.equity:,.0f} ({ret:+.1f}%)<br>'
            
        html += '</div>'
        self.vline.setPos(x_val); self.vline.setVisible(True)
        self.label.setHtml(html); self.label.setPos(x_val, mouse_point.y()); self.label.setVisible(True)

    def _get_nearest_index(self, x_ts):
        idx = np.searchsorted(self.x_data, x_ts)
        if idx >= len(self.x_data): idx = len(self.x_data) - 1
        if idx > 0 and abs(x_ts - self.x_data[idx-1]) < abs(x_ts - self.x_data[idx]): idx -= 1
        return idx


class DrawdownChart(BaseChart):
    def __init__(self):
        super().__init__("Drawdown")
        self.plot_widget.setLabel('left', 'Drawdown (%)')

    def _plot_content(self, results, **kwargs):
        for i, (name, result) in enumerate(results.items()):
            if not result.equity_curve: continue
            
            _, equities = result.get_equity_series()
            arr = np.array(equities)
            peak = np.maximum.accumulate(arr)
            with np.errstate(divide='ignore', invalid='ignore'):
                 dd = (arr - peak) / peak * 100
            dd = np.nan_to_num(dd)
            
            color = self.COLORS[i % len(self.COLORS)]

            min_len = min(len(self.x_data), len(dd))
            
            self.plot_widget.plot(
                self.x_data[:min_len], dd[:min_len],
                pen=pg.mkPen(color=color, width=2),
                fillLevel=0, 
                brush=pg.mkBrush(color + '40') if name=="Current" else None, 
                name=name
            )

    def _on_hover(self, pos):
        if not hasattr(self, 'results') or not hasattr(self, 'x_data'): return
        if not self.plot_widget.sceneBoundingRect().contains(pos):
             self.vline.setVisible(False); self.label.setVisible(False); return
             
        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
        idx = self._get_nearest_index(mouse_point.x())
        if idx is None: return
        
        x_val = self.x_data[idx]
        date_str = pd.to_datetime(self.x_data[idx], unit='s').strftime('%Y-%m-%d')
        
        html = f'<div style="color: white; font-size: 10pt;"><b>📅 {date_str}</b><br>'
        
        for i, (name, result) in enumerate(self.results.items()):
            if not result.equity_curve or idx >= len(result.equity_curve): continue
            
            # Simple DD calculation for tooltip
            series = [e.equity for e in result.equity_curve[:idx+1]]
            if not series: continue
            curr = series[-1]
            peak = max(series)
            color = self.COLORS[i % len(self.COLORS)]
            if peak > 0:
                dd = (curr - peak) / peak * 100
                html += f'<span style="color: {color};">●</span> {name}: {dd:.1f}%<br>'
        
        html += '</div>'
        self.vline.setPos(x_val); self.vline.setVisible(True)
        self.label.setHtml(html); self.label.setPos(x_val, mouse_point.y()); self.label.setVisible(True)

    def _get_nearest_index(self, x_ts):
        idx = np.searchsorted(self.x_data, x_ts)
        if idx >= len(self.x_data): idx = len(self.x_data) - 1
        if idx > 0 and abs(x_ts - self.x_data[idx-1]) < abs(x_ts - self.x_data[idx]): idx -= 1
        return idx


class AllocationChart(BaseChart):
    def __init__(self):
        super().__init__("Asset Allocation")
        self.plot_widget.setLabel('left', 'Allocation (%)')
        self.plot_widget.addLegend()

    def _plot_content(self, results, **kwargs):
        # NOTE: Showing allocation for multiple strategies is chaotic. 
        # Show allocation only for "Current".
        
        target_res = results.get("Current")
        if not target_res and results:
             target_res = next(iter(results.values()))
        
        # Store for tooltip
        self.displayed_result = target_res
             
        if not target_res or not target_res.equity_curve: return
        
        result = target_res
        
        all_symbols = set()
        for ep in result.equity_curve:
            all_symbols.update(ep.allocation.keys())
            
        self.symbol_colors = {}
        
        for idx, symbol in enumerate(sorted(all_symbols)):
            alloc = [ep.allocation.get(symbol, 0) * 100 for ep in result.equity_curve]
            color = self.COLORS[idx % len(self.COLORS)]
            self.symbol_colors[symbol] = color
            self.plot_widget.plot(
                self.x_data, alloc,
                pen=pg.mkPen(color=color, width=2),
                name=f"{symbol}" # Simplified name
            )

    def _on_hover(self, pos):
        if not hasattr(self, 'displayed_result') or not self.displayed_result or not hasattr(self, 'x_data'): return
        if not self.plot_widget.sceneBoundingRect().contains(pos):
             self.vline.setVisible(False); self.label.setVisible(False); return
             
        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
        idx = self._get_nearest_index(mouse_point.x())
        if idx is None: return
        
        x_val = self.x_data[idx]
        ep = self.displayed_result.equity_curve[idx]
        
        date_str = pd.to_datetime(self.x_data[idx], unit='s').strftime('%Y-%m-%d')
        lines = [f"📅 {date_str}"]
        
        # Add strategy name to header
        lines.append(f"Strategy: {self.displayed_result.strategy_name}")
        
        for symbol, pct in sorted(ep.allocation.items(), key=lambda x: x[1], reverse=True):
            if pct > 0.001:
                lines.append(f"{symbol}: {pct*100:.1f}%")
                
        self.vline.setPos(x_val); self.vline.setVisible(True)
        self.label.setText("\n".join(lines)); self.label.setPos(x_val, mouse_point.y()); self.label.setVisible(True)

    def _get_nearest_index(self, x_ts):
        idx = np.searchsorted(self.x_data, x_ts)
        if idx >= len(self.x_data): idx = len(self.x_data) - 1
        if idx > 0 and abs(x_ts - self.x_data[idx-1]) < abs(x_ts - self.x_data[idx]): idx -= 1
        return idx


class VolatilityChart(BaseChart):
    def __init__(self):
        super().__init__("Rolling Volatility")
        self.plot_widget.setLabel('left', 'Annualized Volatility (%)')
        self.params = {'window': 21}  # Default 21-day rolling window

    def _plot_content(self, results, **kwargs):
        window = int(self.params.get('window', 21))
        
        for i, (name, result) in enumerate(results.items()):
            if not result.equity_curve: continue
            
            _, equities = result.get_equity_series()
            s = pd.Series(equities)
            rets = s.pct_change()
            vol = rets.rolling(window=window).std() * np.sqrt(252) * 100
            vol_vals = vol.fillna(0).values
            
            color = self.COLORS[i % len(self.COLORS)]

            min_len = min(len(self.x_data), len(vol_vals))
            
            self.plot_widget.plot(
                self.x_data[:min_len], vol_vals[:min_len],
                pen=pg.mkPen(color=color, width=2),
                name=name
            )

    def _on_hover(self, pos):
        if not hasattr(self, 'results') or not hasattr(self, 'x_data'): return
        if not self.plot_widget.sceneBoundingRect().contains(pos):
             self.vline.setVisible(False); self.label.setVisible(False); return
             
        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
        idx = self._get_nearest_index(mouse_point.x())
        if idx is None: return
        
        x_val = self.x_data[idx]
        date_str = pd.to_datetime(self.x_data[idx], unit='s').strftime('%Y-%m-%d')
        
        html = f'<div style="color: white; font-size: 10pt;"><b>📅 {date_str}</b><br>'
        
        window = int(self.params.get('window', 21))
        for i, (name, result) in enumerate(self.results.items()):
            if not result.equity_curve or idx >= len(result.equity_curve): continue
            
            color = self.COLORS[i % len(self.COLORS)]
            if idx >= window:
                eqs = [e.equity for e in result.equity_curve[idx-window:idx+1]]
                s = pd.Series(eqs)
                val = s.pct_change().std() * np.sqrt(252) * 100
                html += f'<span style="color: {color};">●</span> {name}: {val:.1f}%<br>'
        
        html += '</div>'
        self.vline.setPos(x_val); self.vline.setVisible(True)
        self.label.setHtml(html); self.label.setPos(x_val, mouse_point.y()); self.label.setVisible(True)

    def _get_nearest_index(self, x_ts):
        idx = np.searchsorted(self.x_data, x_ts)
        if idx >= len(self.x_data): idx = len(self.x_data) - 1
        if idx > 0 and abs(x_ts - self.x_data[idx-1]) < abs(x_ts - self.x_data[idx]): idx -= 1
        return idx


class NumericTableWidgetItem(QTableWidgetItem):
    """Table item that sorts numerically."""
    def __lt__(self, other):
        try:
            return float(self.data(Qt.UserRole)) < float(other.data(Qt.UserRole))
        except (ValueError, TypeError):
            return super().__lt__(other)


class SummaryTableChart(QWidget):
    """Chart-like widget that displays a performance summary table."""
    def __init__(self):
        super().__init__()
        self.params = {}
        self.plot_widget = None # No plot widget
        self._setup_ui()
        
    def _setup_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.table = QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels([
            "Name", "Total Return", "CAGR", "Max DD", 
            "Sharpe", "Sortino", "Vol", "Trades", "Alpha (%)", "Beta"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        # Style the table to look nice in dark mode/fusion
        self.table.setStyleSheet("""
            QTableWidget {
                border: none;
                gridline-color: #444;
            }
            QHeaderView::section {
                background-color: #333;
                color: white;
                padding: 4px;
                border: 1px solid #444;
            }
        """)
        
        self.layout.addWidget(self.table)
        
    def update_chart(self, results: Dict[str, BacktestResult], **kwargs):
        # Flatten VTI results into main dict for display if passed
        display_results = results.copy()
        
        vti_withdrawal = kwargs.get('vti_withdrawal')
        if vti_withdrawal:
            display_results['VTI + Withdrawal'] = vti_withdrawal
            
        self.table.setRowCount(len(display_results))
        self.table.setSortingEnabled(False)
        
        for row, (name, res) in enumerate(display_results.items()):
            self.table.setItem(row, 0, QTableWidgetItem(name))
            
            def item(val, fmt="{:.2f}"):
                if val is None:
                    it = NumericTableWidgetItem("-")
                    it.setData(Qt.UserRole, -999999.0) # Sort Nones to bottom/top
                    return it
                it = NumericTableWidgetItem(fmt.format(val))
                it.setData(Qt.UserRole, val)
                it.setTextAlignment(Qt.AlignCenter)
                return it
                
            self.table.setItem(row, 1, item(res.total_return_pct, "{:.2f}%"))
            self.table.setItem(row, 2, item(res.cagr * 100, "{:.2f}%"))
            self.table.setItem(row, 3, item(res.max_drawdown_pct, "{:.2f}%"))
            self.table.setItem(row, 4, item(res.sharpe_ratio))
            self.table.setItem(row, 5, item(res.sortino_ratio))
            self.table.setItem(row, 6, item(res.volatility * 100, "{:.2f}%"))
            self.table.setItem(row, 7, item(len(res.trades), "{:d}"))
            
            alpha_val = res.alpha * 100 if res.alpha is not None else None
            self.table.setItem(row, 8, item(alpha_val, "{:+.2f}%"))
            self.table.setItem(row, 9, item(res.beta, "{:.2f}"))
            
        self.table.setSortingEnabled(True)

    def get_params(self):
        return self.params
        
    def set_params(self, params):
        self.params.update(params)
        
    def set_x_link(self, target):
        # Table doesn't support X-linking
        pass


class ChartContainer(QWidget):
    """Container for a single chart with controls."""
    
    # Signal: container_id
    on_close = None 
    on_type_change = None
    
    CHART_TYPES = {
        "Equity Curve": EquityChart,
        "Drawdown": DrawdownChart,
        "Asset Allocation": AllocationChart,
        "Rolling Volatility": VolatilityChart,
        "Summary Table": SummaryTableChart
    }
    
    def __init__(self, initial_type="Equity Curve", parent=None):
        super().__init__(parent)
        self.current_chart = None
        self._setup_ui(initial_type)
        
    def _setup_ui(self, initial_type):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(2)
        
        # Header
        header = QHBoxLayout()
        header.setContentsMargins(5, 5, 5, 0)
        
        self.type_combo = QComboBox()
        self.type_combo.addItems(list(self.CHART_TYPES.keys()))
        self.type_combo.setCurrentText(initial_type)
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        header.addWidget(self.type_combo)
        
        # Settings button
        self.settings_btn = QToolButton()
        self.settings_btn.setText("⚙️")
        self.settings_btn.setToolTip("Chart Settings")
        self.settings_btn.clicked.connect(self._on_settings_clicked)
        header.addWidget(self.settings_btn)
        
        # Close button
        self.close_btn = QToolButton()
        self.close_btn.setText("✕")
        self.close_btn.setStyleSheet("color: red; font-weight: bold;")
        self.close_btn.setToolTip("Remove Chart")
        self.close_btn.clicked.connect(lambda: self.on_close(self) if self.on_close else None)
        header.addWidget(self.close_btn)
        
        self.layout.addLayout(header)
        
        # Content area
        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.content_area, stretch=1)
        
        # Initialize chart
        self._set_chart(initial_type)
        
    def _set_chart(self, chart_type):
        # Clear existing
        if self.current_chart:
            self.content_layout.removeWidget(self.current_chart)
            self.current_chart.deleteLater()
            self.current_chart = None
            
        chart_class = self.CHART_TYPES.get(chart_type)
        if chart_class:
            self.current_chart = chart_class()
            self.content_layout.addWidget(self.current_chart)
            
            # Show/hide settings button based on params
            has_params = bool(self.current_chart.get_params())
            self.settings_btn.setVisible(has_params)
            
        if self.on_type_change:
            self.on_type_change()

    def _on_type_changed(self, text):
        self._set_chart(text)
        
    def _on_settings_clicked(self):
        if not self.current_chart: return
        
        params = self.current_chart.get_params()
        dialog = ChartConfigDialog(params, self)
        if dialog.exec() == QDialog.Accepted:
            new_params = dialog.get_params()
            self.current_chart.set_params(new_params)
            
    def update_chart(self, results: Dict[str, BacktestResult], **kwargs):
        if self.current_chart:
            self.current_chart.update_chart(results, **kwargs)
            
    def get_plot_widget(self):
        if self.current_chart:
            return getattr(self.current_chart, 'plot_widget', None)
        return None


class BacktestPlotPanel(QWidget):
    """Refactored panel for displaying flexible backtest charts."""
    
    def __init__(self):
        super().__init__()
        self.charts = []
        self.last_result_args = None # Cache for refreshing
        self._setup_ui()
    
    def _setup_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Toolbar
        toolbar = QHBoxLayout()
        self.add_btn = QPushButton("+ Add Chart")
        self.add_btn.setStyleSheet("background-color: #444; color: white;")
        self.add_btn.clicked.connect(self._add_chart_slot)
        toolbar.addWidget(self.add_btn)
        toolbar.addStretch()
        self.main_layout.addLayout(toolbar)
        
        # Charts Area (Splitter for resizing?)
        # A simple VBox inside a ScrollArea might be better if many charts, 
        # but for now let's just use VBox with equal stretch or QSplitter.
        # QSplitter is nice for resizing.
        self.splitter = QSplitter(Qt.Vertical)
        self.main_layout.addWidget(self.splitter)
        
        # Add default charts
        self.add_chart("Equity Curve")
        self.add_chart("Drawdown")
        self.add_chart("Asset Allocation")

    def add_chart(self, chart_type="Equity Curve"):
        container = ChartContainer(chart_type)
        container.on_close = self._remove_chart
        container.on_type_change = self._sync_x_axes
        
        self.charts.append(container)
        self.splitter.addWidget(container)
        
        # If we have data, populate it immediately
        if self.last_result_args:
            container.update_chart(*self.last_result_args[0], **self.last_result_args[1])
            
        self._sync_x_axes()
        
    def _add_chart_slot(self):
        self.add_chart()

    def _remove_chart(self, container):
        if container in self.charts:
            self.charts.remove(container)
            container.deleteLater()
            self._sync_x_axes()

    def _sync_x_axes(self):
        # Link all charts to the first one (or any one)
        if not self.charts: return
        
        first_plot = None
        # Find first valid plot
        for c in self.charts:
            p = c.get_plot_widget()
            if p:
                first_plot = p
                break
        
        if not first_plot: return
        
        for c in self.charts:
            p = c.get_plot_widget()
            if p and p != first_plot:
                p.setXLink(first_plot)

    def update_plots(self, results: Dict[str, BacktestResult]):
        """Update all active charts.
        
        Args:
            results: Dictionary mapping strategy name to BacktestResult.
        """
        self.last_result_args = (results, {})
        
        for chart in self.charts:
            chart.update_chart(results)
            
        # Re-sync in case plot widgets changed
        self._sync_x_axes()

class BacktestReportWindow(QWidget):
    """Popup window for displaying backtest report."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Backtest Report")
        self.resize(500, 700)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                background-color: #1e1e1e;
                color: #d4d4d4;
                padding: 10px;
            }
        """)
        layout.addWidget(self.results_text)
    
    def set_report(self, result: BacktestResult, benchmark: BacktestResult = None):
        """Set the report text from backtest result."""
        report = result.summary(benchmark)
        
        # Add transaction log at the bottom
        if result.transaction_log:
            report += "\n\n" + "=" * 60
            report += "\n📋 TRANSACTION LOG"
            report += "\n" + "=" * 60 + "\n"
            # Show last 100 transactions (most recent at bottom)
            recent_log = result.transaction_log
            report += "\n".join(recent_log)
        
        self.results_text.setPlainText(report)
    
    def show_report(self):
        """Show the report window."""
        self.show()
        self.raise_()
        self.activateWindow()


class BacktestWindow(QMainWindow):
    """Main backtest window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Backtest - Investment Strategy Tester")
        self.setGeometry(100, 100, 1200, 800)
        
        self._setup_ui()
        
        # Auto-run backtest on startup (delayed slightly to ensure UI is ready)
        QTimer.singleShot(100, self.run_backtest)

    
    def _setup_ui(self):
        # Main Layout (No more Tabs)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left: Control panel
        self.control_panel = BacktestControlPanel(on_run_backtest=self.run_backtest)
        self.control_panel.setFixedWidth(280)
        main_layout.addWidget(self.control_panel)
        
        # Right: Plot panel
        self.plot_panel = BacktestPlotPanel()
        main_layout.addWidget(self.plot_panel)
        
        # Report popup window
        self.report_window = BacktestReportWindow()
        
        # Connect Show Report button
        self.control_panel.report_btn.clicked.connect(self._show_report)
            
    def _show_report(self):
        """Show the report popup window."""
        self.report_window.show_report()
    
    def run_backtest(self):
        """Execute backtest for all active strategies (Current + Pinned)."""
        configs = self.control_panel.get_active_configs()
        if not configs:
            self.report_window.results_text.setPlainText(
                "Error: No valid strategy configuration found."
            )
            return

        # Common Settings (Global)
        start_date = self.control_panel.get_start_date()
        end_date = self.control_panel.get_end_date()
        capital = self.control_panel.get_initial_capital()
        
        withdrawal_amount = self.control_panel.get_withdrawal_amount()
        withdrawal_period = self.control_panel.get_withdrawal_period()
        withdrawal_method = self.control_panel.get_withdrawal_method()
        adjust_inflation = self.control_panel.should_adjust_for_inflation()
        
        results = {}
        benchmark = None # Keep track of benchmark for the report window
        
        try:
            for config in configs:
                # Add allocation to params if needed
                params = config.strategy_params.copy()
                params['allocation'] = config.allocation
                
                try:
                    strategy = StrategyRegistry.create(config.strategy_name, **params)
                except TypeError:
                    if 'allocation' in params: del params['allocation']
                    strategy = StrategyRegistry.create(config.strategy_name, **params)
                
                if not strategy:
                    print(f"Error creating strategy {config.strategy_name}")
                    continue
                    
                backtester = Backtester(
                    strategy=strategy,
                    symbols=config.assets,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=capital,
                    withdrawal_amount=withdrawal_amount,
                    withdrawal_period_days=withdrawal_period,
                    withdrawal_method=withdrawal_method,
                    target_allocation=config.allocation,
                    adjust_for_inflation=adjust_inflation
                )
                res = backtester.run()
                res.strategy_name = config.name # Override name with pinned alias
                results[config.name] = res
                

            
            # Update UI
            self.plot_panel.update_plots(results)
            
            # Report usually shows "Current" result. 
            if "Current" in results:
                self.report_window.set_report(results["Current"], benchmark)
            elif results:
                self.report_window.set_report(next(iter(results.values())), benchmark)
            
        except Exception as e:
            self.report_window.results_text.setPlainText(
                f"Error running backtest:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
        except Exception as e:
            self.report_window.results_text.setPlainText(
                f"Error running backtest:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()
    
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Escape, Qt.Key_Q):
            self.close()
        super().keyPressEvent(event)


def main():
    """Launch the Backtest GUI."""
    app = QApplication(sys.argv)
    
    # Set dark theme
    app.setStyle('Fusion')
    
    window = BacktestWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

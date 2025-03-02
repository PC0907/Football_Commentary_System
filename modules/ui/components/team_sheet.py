from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                           QLabel, QTabWidget, QTableWidget, QTableWidgetItem,
                           QHeaderView, QFileDialog, QMessageBox, QLineEdit,
                           QComboBox, QFormLayout, QWidget, QSpinBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import csv
import json

class TeamSheetDialog(QDialog):
    def __init__(self, team_data=None, theme_manager=None, parent=None):
        super().__init__(parent)
        
        self.team_data = team_data or {
            "team_name": "Default Team",
            "players": []
        }
        
        self.theme_manager = theme_manager
        
        self.init_ui()
        self.apply_theme()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Team Sheet")
        self.setMinimumSize(600, 500)
        
        layout = QVBoxLayout(self)
        
        # Team name field
        team_header_layout = QHBoxLayout()
        team_label = QLabel("Team Name:")
        team_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        team_header_layout.addWidget(team_label)
        
        self.team_name_edit = QLineEdit(self.team_data.get("team_name", ""))
        team_header_layout.addWidget(self.team_name_edit)
        
        layout.addLayout(team_header_layout)
        
        # Tab widget for different input methods
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Tab 1: Manual entry
        manual_tab = QWidget()
        self.setup_manual_tab(manual_tab)
        self.tabs.addTab(manual_tab, "Manual Entry")
        
        # Tab 2: CSV import
        csv_tab = QWidget()
        self.setup_csv_tab(csv_tab)
        self.tabs.addTab(csv_tab, "Import from CSV")
        
        # Button layout
        button_layout = QHBoxLayout()
        
        # Save button
        self.save_btn = QPushButton("Save Team Sheet")
        self.save_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.save_btn)
        
        # Cancel button
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
    
    def apply_theme(self):
        """Apply theme settings to the dialog"""
        if not self.theme_manager:
            return
        
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {self.theme_manager.bg_color};
                color: {self.theme_manager.primary_text_color};
                font-family: {self.theme_manager.font_family};
            }}
            
            QTabWidget::pane {{
                border: 1px solid {self.theme_manager.border_color};
                background-color: {self.theme_manager.bg_color};
            }}
            
            QTabBar::tab {{
                background-color: {self.theme_manager.secondary_bg_color};
                color: {self.theme_manager.primary_text_color};
                border: 1px solid {self.theme_manager.border_color};
                padding: 8px 16px;
                margin-right: 2px;
            }}
            
            QTabBar::tab:selected {{
                background-color: {self.theme_manager.accent_color};
                color: {self.theme_manager.light_text_color};
            }}
            
            QTableWidget {{
                border: 1px solid {self.theme_manager.border_color};
                background-color: {self.theme_manager.bg_color};
                color: {self.theme_manager.primary_text_color};
                gridline-color: {self.theme_manager.border_color};
            }}
            
            QTableWidget::item:selected {{
                background-color: {self.theme_manager.accent_color};
                color: {self.theme_manager.light_text_color};
            }}
            
            QHeaderView::section {{
                background-color: {self.theme_manager.secondary_bg_color};
                color: {self.theme_manager.primary_text_color};
                padding: 4px;
                border: 1px solid {self.theme_manager.border_color};
            }}
            
            QLineEdit, QComboBox, QSpinBox {{
                border: 1px solid {self.theme_manager.border_color};
                border-radius: 3px;
                padding: 5px;
                background-color: {self.theme_manager.input_bg_color};
                color: {self.theme_manager.primary_text_color};
            }}
        """)
    
    def setup_manual_tab(self, tab):
        """Set up the manual entry tab"""
        layout = QVBoxLayout(tab)
        
        # Player table
        self.player_table = QTableWidget()
        self.player_table.setColumnCount(3)
        self.player_table.setHorizontalHeaderLabels(["Number", "Name", "Position"])
        
        # Set column widths
        header = self.player_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        
        # Load existing player data
        self.load_players_to_table()
        
        layout.addWidget(self.player_table)
        
        # Controls for adding players
        form_layout = QFormLayout()
        
        # Number input
        self.number_input = QSpinBox()
        self.number_input.setRange(1, 99)
        self.number_input.setValue(1)
        form_layout.addRow("Number:", self.number_input)
        
        # Name input
        self.name_input = QLineEdit()
        form_layout.addRow("Name:", self.name_input)
        
        # Position dropdown
        self.position_input = QComboBox()
        self.position_input.addItems(["Goalkeeper", "Defender", "Midfielder", "Forward"])
        form_layout.addRow("Position:", self.position_input)
        
        layout.addLayout(form_layout)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        # Add player button
        add_btn = QPushButton("Add Player")
        add_btn.clicked.connect(self.add_player)
        button_layout.addWidget(add_btn)
        
        # Remove player button
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_player)
        button_layout.addWidget(remove_btn)
        
        # Clear all button
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_players)
        button_layout.addWidget(clear_btn)
        
        layout.addLayout(button_layout)
    
    def setup_csv_tab(self, tab):
        """Set up the CSV import tab"""
        layout = QVBoxLayout(tab)
        
        # Instructions
        instructions = QLabel(
            "Import a CSV file with the following columns:\n"
            "1. Number (1-99)\n"
            "2. Name\n"
            "3. Position (Goalkeeper, Defender, Midfielder, Forward)\n\n"
            "The first row should be a header row."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Import button
        import_btn = QPushButton("Select CSV File")
        import_btn.clicked.connect(self.import_csv)
        layout.addWidget(import_btn)
        
        # Preview table
        self.csv_preview = QTableWidget()
        self.csv_preview.setColumnCount(3)
        self.csv_preview.setHorizontalHeaderLabels(["Number", "Name", "Position"])
        
        # Set column widths
        header = self.csv_preview.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        
        layout.addWidget(self.csv_preview)
        
        # Status label
        self.csv_status = QLabel("No file selected")
        layout.addWidget(self.csv_status)
    
    def load_players_to_table(self):
        """Load existing player data into the table"""
        players = self.team_data.get("players", [])
        self.player_table.setRowCount(len(players))
        
        for row, player in enumerate(players):
            self.player_table.setItem(row, 0, QTableWidgetItem(str(player.get("number", ""))))
            self.player_table.setItem(row, 1, QTableWidgetItem(player.get("name", "")))
            self.player_table.setItem(row, 2, QTableWidgetItem(player.get("position", "")))
    
    def add_player(self):
        """Add a player to the table"""
        number = self.number_input.value()
        name = self.name_input.text().strip()
        position = self.position_input.currentText()
        
        if not name:
            QMessageBox.warning(self, "Missing Data", "Please enter a player name.")
            return
        
        # Check if the number is already used
        for row in range(self.player_table.rowCount()):
            if self.player_table.item(row, 0).text() == str(number):
                QMessageBox.warning(self, "Duplicate Number", 
                                   f"Player number {number} is already assigned.")
                return
        
        # Add the player to the table
        row = self.player_table.rowCount()
        self.player_table.setRowCount(row + 1)
        
        self.player_table.setItem(row, 0, QTableWidgetItem(str(number)))
        self.player_table.setItem(row, 1, QTableWidgetItem(name))
        self.player_table.setItem(row, 2, QTableWidgetItem(position))
        
        # Clear the input fields
        self.number_input.setValue(self.number_input.value() + 1)
        self.name_input.clear()
    
    def remove_player(self):
        """Remove selected player from the table"""
        selected_rows = set(index.row() for index in self.player_table.selectedIndexes())
        
        if not selected_rows:
            return
        
        # Remove rows in reverse order to maintain correct indices
        for row in sorted(selected_rows, reverse=True):
            self.player_table.removeRow(row)
    
    def clear_players(self):
        """Clear all players from the table"""
        confirm = QMessageBox.question(
            self, "Confirm Clear", "Are you sure you want to clear all players?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if confirm == QMessageBox.StandardButton.Yes:
            self.player_table.setRowCount(0)
    
    def import_csv(self):
        """Import player data from a CSV file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv)"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                
                # Skip header row
                next(reader, None)
                
                # Read player data
                players = []
                for row in reader:
                    if len(row) >= 3:
                        try:
                            number = int(row[0])
                            name = row[1].strip()
                            position = row[2].strip()
                            
                            if not name:
                                continue
                            
                            # Validate position
                            valid_positions = ["Goalkeeper", "Defender", "Midfielder", "Forward"]
                            if position not in valid_positions:
                                position = "Midfielder"  # Default position
                            
                            players.append({
                                "number": number,
                                "name": name,
                                "position": position
                            })
                        except ValueError:
                            continue
                
                # Update the preview table
                self.csv_preview.setRowCount(len(players))
                for row, player in enumerate(players):
                    self.csv_preview.setItem(row, 0, QTableWidgetItem(str(player["number"])))
                    self.csv_preview.setItem(row, 1, QTableWidgetItem(player["name"]))
                    self.csv_preview.setItem(row, 2, QTableWidgetItem(player["position"]))
                
                # Update the manual entry table too
                self.player_table.setRowCount(len(players))
                for row, player in enumerate(players):
                    self.player_table.setItem(row, 0, QTableWidgetItem(str(player["number"])))
                    self.player_table.setItem(row, 1, QTableWidgetItem(player["name"]))
                    self.player_table.setItem(row, 2, QTableWidgetItem(player["position"]))
                
                self.csv_status.setText(f"Imported {len(players)} players from {file_path}")
                
                # Switch to manual tab to show the imported data
                self.tabs.setCurrentIndex(0)
        
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Error importing CSV: {str(e)}")
            self.csv_status.setText("Import failed")
    
    def get_team_data(self):
        """Get the current team data from the table"""
        players = []
        
        for row in range(self.player_table.rowCount()):
            try:
                number = int(self.player_table.item(row, 0).text())
                name = self.player_table.item(row, 1).text().strip()
                position = self.player_table.item(row, 2).text().strip()
                
                if name:
                    players.append({
                        "number": number,
                        "name": name,
                        "position": position
                    })
            except (ValueError, AttributeError):
                continue
        
        # Sort players by number
        players.sort(key=lambda p: p["number"])
        
        return {
            "team_name": self.team_name_edit.text().strip() or "Default Team",
            "players": players
        }
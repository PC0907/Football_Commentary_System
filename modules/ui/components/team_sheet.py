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
            "team_a": {
                "team_name": "Team A",
                "players": []
            },
            "team_b": {
                "team_name": "Team B",
                "players": []
            }
        }
        
        self.theme_manager = theme_manager
        
        self.init_ui()
        self.apply_theme()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Team Sheet")
        self.setMinimumSize(1200, 500)
        
        layout = QVBoxLayout(self)
        
        # Team name fields
        team_header_layout = QHBoxLayout()
        
        team_a_label = QLabel("Team A Name:")
        team_a_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        team_header_layout.addWidget(team_a_label)
        
        self.team_a_name_edit = QLineEdit(self.team_data["team_a"].get("team_name", ""))
        team_header_layout.addWidget(self.team_a_name_edit)
        
        team_b_label = QLabel("Team B Name:")
        team_b_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        team_header_layout.addWidget(team_b_label)
        
        self.team_b_name_edit = QLineEdit(self.team_data["team_b"].get("team_name", ""))
        team_header_layout.addWidget(self.team_b_name_edit)
        
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
        layout = QHBoxLayout(tab)
        
        # Team A player table
        team_a_layout = QVBoxLayout()
        team_a_label = QLabel("Team A Players")
        team_a_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        team_a_layout.addWidget(team_a_label)
        
        self.team_a_table = QTableWidget()
        self.team_a_table.setColumnCount(3)
        self.team_a_table.setHorizontalHeaderLabels(["Number", "Name", "Position"])
        
        # Set column widths
        header = self.team_a_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        
        # Load existing player data
        self.load_players_to_table(self.team_a_table, self.team_data["team_a"]["players"])
        
        team_a_layout.addWidget(self.team_a_table)
        
        # Controls for adding players to Team A
        form_layout_a = QFormLayout()
        
        # Number input
        self.number_input_a = QSpinBox()
        self.number_input_a.setRange(1, 99)
        self.number_input_a.setValue(1)
        form_layout_a.addRow("Number:", self.number_input_a)
        
        # Name input
        self.name_input_a = QLineEdit()
        form_layout_a.addRow("Name:", self.name_input_a)
        
        # Position dropdown
        self.position_input_a = QComboBox()
        self.position_input_a.addItems(["Goalkeeper", "Defender", "Midfielder", "Forward"])
        form_layout_a.addRow("Position:", self.position_input_a)
        
        team_a_layout.addLayout(form_layout_a)
        
        # Button layout for Team A
        button_layout_a = QHBoxLayout()
        
        # Add player button
        add_btn_a = QPushButton("Add Player")
        add_btn_a.clicked.connect(lambda: self.add_player(self.team_a_table, self.number_input_a, self.name_input_a, self.position_input_a))
        button_layout_a.addWidget(add_btn_a)
        
        # Remove player button
        remove_btn_a = QPushButton("Remove Selected")
        remove_btn_a.clicked.connect(lambda: self.remove_player(self.team_a_table))
        button_layout_a.addWidget(remove_btn_a)
        
        # Clear all button
        clear_btn_a = QPushButton("Clear All")
        clear_btn_a.clicked.connect(lambda: self.clear_players(self.team_a_table))
        button_layout_a.addWidget(clear_btn_a)
        
        team_a_layout.addLayout(button_layout_a)
        
        layout.addLayout(team_a_layout)
        
        # Spacer
        layout.addSpacing(20)
        
        # Team B player table
        team_b_layout = QVBoxLayout()
        team_b_label = QLabel("Team B Players")
        team_b_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        team_b_layout.addWidget(team_b_label)
        
        self.team_b_table = QTableWidget()
        self.team_b_table.setColumnCount(3)
        self.team_b_table.setHorizontalHeaderLabels(["Number", "Name", "Position"])
        
        # Set column widths
        header = self.team_b_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        
        # Load existing player data
        self.load_players_to_table(self.team_b_table, self.team_data["team_b"]["players"])
        
        team_b_layout.addWidget(self.team_b_table)
        
        # Controls for adding players to Team B
        form_layout_b = QFormLayout()
        
        # Number input
        self.number_input_b = QSpinBox()
        self.number_input_b.setRange(1, 99)
        self.number_input_b.setValue(1)
        form_layout_b.addRow("Number:", self.number_input_b)
        
        # Name input
        self.name_input_b = QLineEdit()
        form_layout_b.addRow("Name:", self.name_input_b)
        
        # Position dropdown
        self.position_input_b = QComboBox()
        self.position_input_b.addItems(["Goalkeeper", "Defender", "Midfielder", "Forward"])
        form_layout_b.addRow("Position:", self.position_input_b)
        
        team_b_layout.addLayout(form_layout_b)
        
        # Button layout for Team B
        button_layout_b = QHBoxLayout()
        
        # Add player button
        add_btn_b = QPushButton("Add Player")
        add_btn_b.clicked.connect(lambda: self.add_player(self.team_b_table, self.number_input_b, self.name_input_b, self.position_input_b))
        button_layout_b.addWidget(add_btn_b)
        
        # Remove player button
        remove_btn_b = QPushButton("Remove Selected")
        remove_btn_b.clicked.connect(lambda: self.remove_player(self.team_b_table))
        button_layout_b.addWidget(remove_btn_b)
        
        # Clear all button
        clear_btn_b = QPushButton("Clear All")
        clear_btn_b.clicked.connect(lambda: self.clear_players(self.team_b_table))
        button_layout_b.addWidget(clear_btn_b)
        
        team_b_layout.addLayout(button_layout_b)
        
        layout.addLayout(team_b_layout)
    
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
    
    def load_players_to_table(self, table, players):
        """Load existing player data into the table"""
        table.setRowCount(len(players))
        
        for row, player in enumerate(players):
            table.setItem(row, 0, QTableWidgetItem(str(player.get("number", ""))))
            table.setItem(row, 1, QTableWidgetItem(player.get("name", "")))
            table.setItem(row, 2, QTableWidgetItem(player.get("position", "")))
    
    def add_player(self, table, number_input, name_input, position_input):
        """Add a player to the table"""
        number = number_input.value()
        name = name_input.text().strip()
        position = position_input.currentText()
        
        if not name:
            QMessageBox.warning(self, "Missing Data", "Please enter a player name.")
            return
        
        # Check if the number is already used
        for row in range(table.rowCount()):
            if table.item(row, 0).text() == str(number):
                QMessageBox.warning(self, "Duplicate Number", 
                                   f"Player number {number} is already assigned.")
                return
        
        # Add the player to the table
        row = table.rowCount()
        table.setRowCount(row + 1)
        
        table.setItem(row, 0, QTableWidgetItem(str(number)))
        table.setItem(row, 1, QTableWidgetItem(name))
        table.setItem(row, 2, QTableWidgetItem(position))
        
        # Clear the input fields
        number_input.setValue(number_input.value() + 1)
        name_input.clear()
    
    def remove_player(self, table):
        """Remove selected player from the table"""
        selected_rows = set(index.row() for index in table.selectedIndexes())
        
        if not selected_rows:
            return
        
        # Remove rows in reverse order to maintain correct indices
        for row in sorted(selected_rows, reverse=True):
            table.removeRow(row)
    
    def clear_players(self, table):
        """Clear all players from the table"""
        confirm = QMessageBox.question(
            self, "Confirm Clear", "Are you sure you want to clear all players?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if confirm == QMessageBox.StandardButton.Yes:
            table.setRowCount(0)
    
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
                self.team_a_table.setRowCount(len(players))
                for row, player in enumerate(players):
                    self.team_a_table.setItem(row, 0, QTableWidgetItem(str(player["number"])))
                    self.team_a_table.setItem(row, 1, QTableWidgetItem(player["name"]))
                    self.team_a_table.setItem(row, 2, QTableWidgetItem(player["position"]))
                
                self.csv_status.setText(f"Imported {len(players)} players from {file_path}")
                
                # Switch to manual tab to show the imported data
                self.tabs.setCurrentIndex(0)
        
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Error importing CSV: {str(e)}")
            self.csv_status.setText("Import failed")
    
    def get_team_data(self):
        """Get the current team data from the table"""
        players_a = []
        players_b = []
        
        for row in range(self.team_a_table.rowCount()):
            try:
                number = int(self.team_a_table.item(row, 0).text())
                name = self.team_a_table.item(row, 1).text().strip()
                position = self.team_a_table.item(row, 2).text().strip()
                
                if name:
                    players_a.append({
                        "number": number,
                        "name": name,
                        "position": position
                    })
            except (ValueError, AttributeError):
                continue
        
        for row in range(self.team_b_table.rowCount()):
            try:
                number = int(self.team_b_table.item(row, 0).text())
                name = self.team_b_table.item(row, 1).text().strip()
                position = self.team_b_table.item(row, 2).text().strip()
                
                if name:
                    players_b.append({
                        "number": number,
                        "name": name,
                        "position": position
                    })
            except (ValueError, AttributeError):
                continue
        
        # Sort players by number
        players_a.sort(key=lambda p: p["number"])
        players_b.sort(key=lambda p: p["number"])
        
        return {
            "team_a": {
                "team_name": self.team_a_name_edit.text().strip() or "Team A",
                "players": players_a
            },
            "team_b": {
                "team_name": self.team_b_name_edit.text().strip() or "Team B",
                "players": players_b
            }
        }
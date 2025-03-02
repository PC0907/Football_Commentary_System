import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                           QVBoxLayout, QHBoxLayout, QFileDialog, QProgressBar,
                           QLabel, QDialog, QTabWidget, QTableWidget, 
                           QTableWidgetItem, QMessageBox, QHeaderView, QSizePolicy,
                           QComboBox, QSpinBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QPixmap, QFont
import cv2
from pathlib import Path
import csv
import json
import time
import logging

# Import theme settings
from themes import ThemeManager
# Import components
from components.video_player import VideoPlayer
from components.team_sheet import TeamSheetDialog
from components.processor import VideoProcessor

class FootballCommentaryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Set up logging
        logging.basicConfig(filename='app.log', level=logging.INFO,
                           format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Initialize theme manager
        self.theme_manager = ThemeManager()
        
        # Setup UI
        self.init_ui()
        
        # Initialize variables
        self.input_video_path = None
        self.team_sheet_data = self.load_default_team_sheet()
        self.processor = None
        
        logging.info("Application started")
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Football Commentary System")
        self.setMinimumSize(1200, 800)
        
        # Apply theme
        self.apply_theme()
        
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Header
        header_layout = QHBoxLayout()
        app_title = QLabel("Football Commentary System")
        app_title.setFont(QFont(self.theme_manager.font_family, 24, QFont.Weight.Bold))
        app_title.setStyleSheet(f"color: {self.theme_manager.primary_text_color};")
        header_layout.addWidget(app_title)
        
        # Theme selector
        theme_label = QLabel("Theme:")
        theme_label.setFont(QFont(self.theme_manager.font_family, 12))
        header_layout.addWidget(theme_label)
        
        self.theme_selector = QComboBox()
        self.theme_selector.addItems(["Light", "Dark", "Blue", "Green"])
        self.theme_selector.setCurrentText(self.theme_manager.current_theme)
        self.theme_selector.currentTextChanged.connect(self.change_theme)
        header_layout.addWidget(self.theme_selector)
        
        header_layout.addStretch()
        main_layout.addLayout(header_layout)
        
        # Video display area
        video_layout = QHBoxLayout()
        
        # Input video player
        self.input_video = VideoPlayer("Input Video")
        video_layout.addWidget(self.input_video)
        
        # Add spacing between videos
        video_layout.addSpacing(20)
        
        # Output video player
        self.output_video = VideoPlayer("Output Video")
        video_layout.addWidget(self.output_video)
        
        main_layout.addLayout(video_layout)
        
        # Control panel
        control_panel = QWidget()
        control_panel.setObjectName("controlPanel")
        control_panel.setStyleSheet(f"""
            QWidget#controlPanel {{
                background-color: {self.theme_manager.secondary_bg_color};
                border-radius: 10px;
                padding: 10px;
            }}
        """)
        
        control_layout = QHBoxLayout(control_panel)
        control_layout.setContentsMargins(20, 20, 20, 20)
        control_layout.setSpacing(15)
        
        # Upload video button
        self.upload_video_btn = QPushButton("Upload Video")
        self.upload_video_btn.setIcon(QIcon("icons/upload_video.png"))
        self.upload_video_btn.setMinimumHeight(40)
        self.upload_video_btn.clicked.connect(self.upload_video)
        control_layout.addWidget(self.upload_video_btn)
        
        # Team sheet button
        self.team_sheet_btn = QPushButton("Team Sheet")
        self.team_sheet_btn.setIcon(QIcon("icons/team.png"))
        self.team_sheet_btn.setMinimumHeight(40)
        self.team_sheet_btn.clicked.connect(self.open_team_sheet)
        control_layout.addWidget(self.team_sheet_btn)
        
        # Process button
        self.process_btn = QPushButton("Process Video")
        self.process_btn.setIcon(QIcon("icons/process.png"))
        self.process_btn.setMinimumHeight(40)
        self.process_btn.clicked.connect(self.process_video)
        self.process_btn.setEnabled(False)
        control_layout.addWidget(self.process_btn)
        
        # Export button
        self.export_btn = QPushButton("Export Video")
        self.export_btn.setIcon(QIcon("icons/export.png"))
        self.export_btn.setMinimumHeight(40)
        self.export_btn.clicked.connect(self.export_video)
        self.export_btn.setEnabled(False)
        control_layout.addWidget(self.export_btn)
        
        main_layout.addWidget(control_panel)
        
        # Progress bar (initially hidden)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumHeight(30)
        self.progress_bar.hide()
        main_layout.addWidget(self.progress_bar)
        
        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setFont(QFont(self.theme_manager.font_family, 10))
        self.status_label.setStyleSheet(f"color: {self.theme_manager.secondary_text_color};")
        main_layout.addWidget(self.status_label)
        
        # Set the main widget
        self.setCentralWidget(main_widget)
        
    def apply_theme(self):
        """Apply the current theme to the application"""
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {self.theme_manager.bg_color};
                color: {self.theme_manager.primary_text_color};
                font-family: {self.theme_manager.font_family};
            }}
            
            QPushButton {{
                background-color: {self.theme_manager.button_color};
                color: {self.theme_manager.button_text_color};
                border-radius: 5px;
                padding: 8px 16px;
                font-size: 12px;
                font-weight: bold;
            }}
            
            QPushButton:hover {{
                background-color: {self.theme_manager.button_hover_color};
            }}
            
            QPushButton:disabled {{
                background-color: {self.theme_manager.disabled_color};
                color: {self.theme_manager.disabled_text_color};
            }}
            
            QProgressBar {{
                border: 2px solid {self.theme_manager.accent_color};
                border-radius: 5px;
                text-align: center;
                background-color: {self.theme_manager.secondary_bg_color};
            }}
            
            QProgressBar::chunk {{
                background-color: {self.theme_manager.accent_color};
                width: 10px;
            }}
            
            QComboBox {{
                border: 1px solid {self.theme_manager.border_color};
                border-radius: 3px;
                padding: 3px 8px;
                background-color: {self.theme_manager.secondary_bg_color};
                color: {self.theme_manager.primary_text_color};
            }}
        """)
    
    def change_theme(self, theme_name):
        """Change the application theme"""
        self.theme_manager.set_theme(theme_name)
        self.apply_theme()
        
        # Update components that need theme refreshing
        self.input_video.update_theme(self.theme_manager)
        self.output_video.update_theme(self.theme_manager)
        
        # Update title colors
        app_title = self.findChild(QLabel)
        if app_title:
            app_title.setStyleSheet(f"color: {self.theme_manager.primary_text_color};")
        
        # Update status label
        self.status_label.setStyleSheet(f"color: {self.theme_manager.secondary_text_color};")
        
        logging.info(f"Theme changed to {theme_name}")
    
    def upload_video(self):
        """Open file dialog to upload a video"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        
        if file_path:
            self.input_video_path = file_path
            self.status_label.setText(f"Video loaded: {os.path.basename(file_path)}")
            
            # Load the video in the input player
            self.input_video.load_video(file_path)
            
            # Enable process button if team sheet is also loaded
            if self.team_sheet_data:
                self.process_btn.setEnabled(True)
                
            logging.info(f"Video uploaded: {file_path}")
    
    def open_team_sheet(self):
        """Open the team sheet dialog"""
        dialog = TeamSheetDialog(self.team_sheet_data, self.theme_manager, self)
        if dialog.exec():
            self.team_sheet_data = dialog.get_team_data()
            self.status_label.setText("Team sheet updated")
            
            # Enable process button if video is also loaded
            if self.input_video_path:
                self.process_btn.setEnabled(True)
                
            # Save team sheet as default for future use
            self.save_default_team_sheet(self.team_sheet_data)
            
            logging.info("Team sheet updated")
    
    def process_video(self):
        """Start video processing"""
        if not self.input_video_path or not self.team_sheet_data:
            QMessageBox.warning(self, "Missing Data", 
                               "Please upload a video and team sheet first.")
            return
        
        # Show progress bar
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.status_label.setText("Processing video...")
        
        # Disable buttons during processing
        self.upload_video_btn.setEnabled(False)
        self.team_sheet_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        
        # Create and start the processor thread
        self.processor = VideoProcessor(self.input_video_path, self.team_sheet_data)
        self.processor.progress_updated.connect(self.update_progress)
        self.processor.processing_complete.connect(self.processing_finished)
        self.processor.start()
        
        logging.info("Video processing started")
    
    def update_progress(self, progress, status_msg):
        """Update the progress bar and status message"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(status_msg)
    
    def processing_finished(self, output_path, success):
        """Handle the completion of video processing"""
        # Hide progress bar
        self.progress_bar.hide()
        
        # Re-enable buttons
        self.upload_video_btn.setEnabled(True)
        self.team_sheet_btn.setEnabled(True)
        self.process_btn.setEnabled(True)
        
        if success:
            self.status_label.setText("Processing complete!")
            
            # Load the processed video in the output player
            self.output_video.load_video(output_path)
            
            # Enable export button
            self.export_btn.setEnabled(True)
            
            logging.info(f"Video processing completed: {output_path}")
        else:
            self.status_label.setText("Error during processing")
            QMessageBox.critical(self, "Processing Error", 
                                "An error occurred during video processing. Check the logs for details.")
            logging.error("Video processing failed")
    
    def export_video(self):
        """Export the processed video to a user-selected location"""
        if not hasattr(self.processor, 'output_path') or not self.processor.output_path:
            QMessageBox.warning(self, "No Output", "No processed video available for export.")
            return
        
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Processed Video", "", "Video Files (*.mp4)"
        )
        
        if save_path:
            # Copy the video to the selected location
            try:
                # Import shutil here to avoid importing it at the top level
                import shutil
                shutil.copy2(self.processor.output_path, save_path)
                self.status_label.setText(f"Video exported to: {save_path}")
                QMessageBox.information(self, "Export Successful", 
                                       f"Video has been exported successfully to:\n{save_path}")
                logging.info(f"Video exported to: {save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Error exporting video: {str(e)}")
                logging.error(f"Export error: {str(e)}")
    
    def load_default_team_sheet(self):
        """Load the default team sheet from a JSON file"""
        try:
            config_dir = Path.home() / ".football_commentary"
            config_dir.mkdir(exist_ok=True)
            team_sheet_path = config_dir / "default_team_sheet.json"
            
            if team_sheet_path.exists():
                with open(team_sheet_path, 'r') as f:
                    return json.load(f)
            else:
                # Return default team sheet if no saved one exists
                return {
                    "team_a": {
                        "team_name": "Team A",
                        "players": [
                            {"number": 1, "name": "John Smith", "position": "Goalkeeper"},
                            {"number": 2, "name": "James Brown", "position": "Defender"},
                            {"number": 3, "name": "Michael Johnson", "position": "Defender"},
                            {"number": 4, "name": "Robert Wilson", "position": "Defender"},
                            {"number": 5, "name": "David Jones", "position": "Defender"},
                            {"number": 6, "name": "Thomas Taylor", "position": "Midfielder"},
                            {"number": 7, "name": "Christopher Anderson", "position": "Midfielder"},
                            {"number": 8, "name": "Joseph Martinez", "position": "Midfielder"},
                            {"number": 9, "name": "Daniel Robinson", "position": "Forward"},
                            {"number": 10, "name": "Paul Wright", "position": "Forward"},
                            {"number": 11, "name": "Mark Walker", "position": "Forward"}
                        ]
                    },
                    "team_b": {
                        "team_name": "Team B",
                        "players": [
                            {"number": 1, "name": "John Doe", "position": "Goalkeeper"},
                            {"number": 2, "name": "James Smith", "position": "Defender"},
                            {"number": 3, "name": "Michael Brown", "position": "Defender"},
                            {"number": 4, "name": "Robert Johnson", "position": "Defender"},
                            {"number": 5, "name": "David Wilson", "position": "Defender"},
                            {"number": 6, "name": "Thomas Jones", "position": "Midfielder"},
                            {"number": 7, "name": "Christopher Taylor", "position": "Midfielder"},
                            {"number": 8, "name": "Joseph Anderson", "position": "Midfielder"},
                            {"number": 9, "name": "Daniel Martinez", "position": "Forward"},
                            {"number": 10, "name": "Paul Robinson", "position": "Forward"},
                            {"number": 11, "name": "Mark Wright", "position": "Forward"}
                        ]
                    }
                }
        except Exception as e:
            logging.error(f"Error loading default team sheet: {str(e)}")
            return None
    
    def save_default_team_sheet(self, team_data):
        """Save the current team sheet as the default"""
        try:
            config_dir = Path.home() / ".football_commentary"
            config_dir.mkdir(exist_ok=True)
            team_sheet_path = config_dir / "default_team_sheet.json"
            
            with open(team_sheet_path, 'w') as f:
                json.dump(team_data, f, indent=4)
                
            logging.info("Default team sheet saved")
        except Exception as e:
            logging.error(f"Error saving default team sheet: {str(e)}")
    
    def closeEvent(self, event):
        """Handle application close event"""
        # Clean up resources, stop any running threads
        if self.processor and self.processor.isRunning():
            self.processor.terminate()
            self.processor.wait()
        
        logging.info("Application closed")
        event.accept()

def main():
    # Create directory for icons if it doesn't exist
    icons_dir = Path("icons")
    icons_dir.mkdir(exist_ok=True)
    
    # You would need to add actual icon files to this directory
    
    app = QApplication(sys.argv)
    window = FootballCommentaryApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
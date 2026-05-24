import sys
import os
import json
import time

# Attempt to import PyQt5, exit if not installed
try:
    from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QProgressBar, QFrame, QCheckBox
    from PyQt5.QtGui import QPixmap, QFont
    from PyQt5.QtCore import Qt
except ModuleNotFoundError:
    print("Error: PyQt5 is not installed. Please install it using 'pip install PyQt5'")
    sys.exit(1)

from image_loader import ImageLoader
from csv_handler import CSVHandler
from augmentor import Augmentor

class ImageAnnotator(QWidget):
    def __init__(self):
        """Initializes the Image Annotator GUI."""
        super().__init__()
        self.initUI()
        self.image_loader = ImageLoader()
        self.csv_handler = CSVHandler()
        self.augmentor = Augmentor()
        self.output_folder = ""
        self.label_text = ""
        self.augmented_mode = False  # Toggle for augmentation mode
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self.session_data = {
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "images_annotated": 0,
            "output_labels": []
        }

    def initUI(self):
        """Sets up the GUI layout and widgets."""
        self.setWindowTitle("Football Jersey Number Annotator")
        self.setGeometry(200, 100, 1000, 700)
        self.setStyleSheet("background-color: #2C2F33; color: #FFFFFF;")

        # Image display area
        self.image_label = QLabel("No Image Loaded")
        self.image_label.setFixedSize(500, 500)
        self.image_label.setStyleSheet("border: 2px solid #7289DA; padding: 5px;")

        # Number display
        self.number_display = QLabel("Enter Number")
        self.number_display.setFont(QFont("Arial", 40, QFont.Bold))
        self.number_display.setAlignment(Qt.AlignCenter)
        self.number_display.setStyleSheet("border: 2px solid #99AAB5; background-color: #23272A; padding: 10px;")
        self.number_display.setFixedSize(250, 120)

        # Buttons
        button_style = "padding: 10px; font-size: 14px; border-radius: 5px; background-color: #7289DA; color: white;"
        self.prev_btn = QPushButton("← Previous")
        self.next_btn = QPushButton("Next →")
        self.load_folder_btn = QPushButton("📂 Load Folder")
        self.select_output_btn = QPushButton("📁 Select Output Folder")
        self.label_btn = QPushButton("✔ Label (Enter)")
        self.save_session_btn = QPushButton("💾 Save Session")

        for btn in [self.prev_btn, self.next_btn, self.load_folder_btn, self.select_output_btn, self.label_btn, self.save_session_btn]:
            btn.setStyleSheet(button_style)

        self.progress = QProgressBar()
        self.augment_mode_toggle = QCheckBox("Enable Augmentation")
        self.augment_mode_toggle.setStyleSheet("font-size: 16px;")

        # Button connections
        self.prev_btn.clicked.connect(self.show_prev_image)
        self.next_btn.clicked.connect(self.show_next_image)
        self.load_folder_btn.clicked.connect(self.load_folder)
        self.select_output_btn.clicked.connect(self.select_output_folder)
        self.label_btn.clicked.connect(self.save_annotation)
        self.save_session_btn.clicked.connect(self.save_session)
        self.augment_mode_toggle.stateChanged.connect(self.toggle_augmentation)

        # Layout
        layout = QVBoxLayout()
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.next_btn)

        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.number_display, alignment=Qt.AlignCenter)
        layout.addWidget(self.load_folder_btn, alignment=Qt.AlignCenter)
        layout.addWidget(self.select_output_btn, alignment=Qt.AlignCenter)
        layout.addWidget(self.label_btn, alignment=Qt.AlignCenter)
        layout.addWidget(self.augment_mode_toggle, alignment=Qt.AlignCenter)
        layout.addWidget(self.save_session_btn, alignment=Qt.AlignCenter)
        layout.addLayout(btn_layout)
        layout.addWidget(self.progress)

        self.setLayout(layout)

    def toggle_augmentation(self, state):
        self.augmented_mode = state == Qt.Checked

    def save_session(self):
        self.session_data["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        session_file = os.path.join(self.output_folder, "session_data.json")
        with open(session_file, "w") as file:
            json.dump(self.session_data, file, indent=4)
        print("Session saved.")

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            self.image_loader.load_images(folder, self.output_folder, self)
            self.show_image()

    def select_output_folder(self):
        """
        Opens a file dialog for selecting the output folder.
        Ensures the folder is set before any annotation.
        """
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = folder
            print(f"Output folder set to: {self.output_folder}")

    def show_prev_image(self):
        self.image_loader.prev_image()
        self.show_image()

    def show_next_image(self):
        self.image_loader.next_image()
        self.show_image()

    def show_image(self):
        image_path = self.image_loader.get_current_image()
        if image_path:
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap.scaled(500, 500, Qt.KeepAspectRatio))

    def save_annotation(self):
        """
        Saves the entered label for the current image and applies augmentations if enabled.
        Ensures the output folder exists before saving.
        """
        label = self.label_text.strip()
        if not label or not self.output_folder:
            print("Error: No label entered or output folder not set.")
            return

        image_path = self.image_loader.get_current_image()
        if not image_path:
            print("Error: No image loaded.")
            return

        print(f"Saving annotation for {image_path} in {self.output_folder}")

        self.csv_handler.save_annotation(image_path, label, self.output_folder, self.session_id)
        self.session_data["images_annotated"] += 1
        self.session_data["output_labels"].append({"image": image_path, "label": label})

        if self.augmented_mode:
            print("Augmented mode is ON: Saving augmented images")
            aug_images = self.augmentor.augment_image(image_path, self.output_folder, True)
            for img in aug_images:
                self.csv_handler.save_annotation(img, label, self.output_folder, self.session_id)

        self.label_text = ""
        self.show_next_image()
        
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Left:
            self.show_prev_image()
        elif key == Qt.Key_Right:
            self.show_next_image()
        elif key == Qt.Key_Return or key == Qt.Key_Enter:
            self.save_annotation()
        elif Qt.Key_0 <= key <= Qt.Key_9:
            self.label_text += chr(key)
            self.number_display.setText(self.label_text)
        elif key == Qt.Key_Backspace:
            self.label_text = self.label_text[:-1]
            self.number_display.setText(self.label_text if self.label_text else "Enter Number")
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageAnnotator()
    window.show()
    sys.exit(app.exec_())

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QSlider, 
                           QPushButton, QHBoxLayout, QSizePolicy,
                           QFrame)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon
import cv2
import numpy as np
import time

class VideoPlayer(QWidget):
    def __init__(self, title):
        super().__init__()
        
        self.title = title
        self.video_path = None
        self.cap = None
        self.current_frame = 0
        self.frame_count = 0
        self.fps = 0
        self.playing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface for the video player"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        self.title_label = QLabel(self.title)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(self.title_label)
        
        # Video display
        self.video_frame = QLabel()
        self.video_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_frame.setMinimumSize(480, 360)
        self.video_frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.video_frame.setStyleSheet("""
            background-color: #222;
            border-radius: 8px;
        """)
        
        # Set placeholder image
        placeholder = QPixmap(480, 360)
        placeholder.fill(Qt.GlobalColor.transparent)
        self.video_frame.setPixmap(placeholder)
        
        layout.addWidget(self.video_frame)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        # Play/Pause button
        self.play_button = QPushButton()
        self.play_button.setIcon(QIcon("icons/play.png"))
        self.play_button.setFixedSize(36, 36)
        self.play_button.clicked.connect(self.toggle_play)
        self.play_button.setEnabled(False)
        controls_layout.addWidget(self.play_button)
        
        # Timeline slider
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setEnabled(False)
        self.timeline.valueChanged.connect(self.slider_moved)
        controls_layout.addWidget(self.timeline)
        
        # Time label
        self.time_label = QLabel("00:00 / 00:00")
        controls_layout.addWidget(self.time_label)
        
        layout.addLayout(controls_layout)
    
    def update_theme(self, theme_manager):
        """Update the component with new theme settings"""
        self.title_label.setFont(QFont(theme_manager.font_family, 14, QFont.Weight.Bold))
        self.title_label.setStyleSheet(f"color: {theme_manager.primary_text_color};")
        
        self.video_frame.setStyleSheet(f"""
            background-color: {theme_manager.secondary_bg_color};
            border-radius: 8px;
        """)
        
        self.time_label.setStyleSheet(f"color: {theme_manager.secondary_text_color};")
    
    def load_video(self, video_path):
        """Load a video file into the player"""
        self.video_path = video_path
        
        # Close any previously opened video
        if self.cap is not None:
            self.cap.release()
        
        # Open the video file
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        # Get video properties
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.timeline.setRange(0, self.frame_count - 1)
        self.current_frame = 0
        
        # Calculate video duration
        duration = self.frame_count / self.fps if self.fps > 0 else 0
        self.time_label.setText(f"00:00 / {self.format_time(duration)}")
        
        # Load and display the first frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        
        if ret:
            self.display_frame(frame)
        
        # Enable controls
        self.play_button.setEnabled(True)
        self.timeline.setEnabled(True)
    
    def toggle_play(self):
        """Toggle video play/pause state"""
        if self.playing:
            self.pause()
        else:
            self.play()
    
    def play(self):
        """Start video playback"""
        if self.cap is None or not self.cap.isOpened():
            return
        
        # If we reached the end, start from beginning
        if self.current_frame >= self.frame_count - 1:
            self.current_frame = 0
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        self.playing = True
        self.play_button.setIcon(QIcon("icons/pause.png"))
        
        # Start the timer to update frames
        interval = int(1000 / self.fps) if self.fps > 0 else 33  # Default to 30fps
        self.timer.start(interval)
    
    def pause(self):
        """Pause video playback"""
        self.playing = False
        self.timer.stop()
        self.play_button.setIcon(QIcon("icons/play.png"))
    
    def update_frame(self):
        """Update the video frame during playback"""
        if self.cap is None or not self.cap.isOpened():
            return
        
        # Read the next frame
        ret, frame = self.cap.read()
        
        if not ret:
            # End of video
            self.pause()
            return
        
        # Display the frame
        self.display_frame(frame)
        
        # Update current frame counter and slider
        self.current_frame += 1
        self.timeline.blockSignals(True)
        self.timeline.setValue(self.current_frame)
        self.timeline.blockSignals(False)
        
        # Update time display
        current_time = self.current_frame / self.fps if self.fps > 0 else 0
        duration = self.frame_count / self.fps if self.fps > 0 else 0
        self.time_label.setText(f"{self.format_time(current_time)} / {self.format_time(duration)}")
        
        # Check if we've reached the end
        if self.current_frame >= self.frame_count - 1:
            self.pause()
    
    def slider_moved(self, position):
        """Handle timeline slider movement"""
        if self.cap is None or not self.cap.isOpened():
            return
        
        # Update current frame position
        self.current_frame = position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        
        # Read and display the frame at the new position
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame)
        
        # Update time display
        current_time = position / self.fps if self.fps > 0 else 0
        duration = self.frame_count / self.fps if self.fps > 0 else 0
        self.time_label.setText(f"{self.format_time(current_time)} / {self.format_time(duration)}")
    
    def display_frame(self, frame):
        """Convert and display a video frame"""
        # Convert the frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get the current size of the video frame widget
        frame_width = self.video_frame.width()
        frame_height = self.video_frame.height()
        
        # Resize the frame to fit the widget while maintaining aspect ratio
        h, w, ch = rgb_frame.shape
        aspect_ratio = w / h
        
        if frame_width / frame_height > aspect_ratio:
            # Widget is wider than the frame
            new_height = frame_height
            new_width = int(new_height * aspect_ratio)
        else:
            # Widget is taller than the frame
            new_width = frame_width
            new_height = int(new_width / aspect_ratio)
        
        # Resize the frame
        resized_frame = cv2.resize(rgb_frame, (new_width, new_height), 
                                 interpolation=cv2.INTER_AREA)
        
        # Convert the frame to QImage and then to QPixmap
        h, w, ch = resized_frame.shape
        bytes_per_line = ch * w
        image = QImage(resized_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        
        # Display the frame
        self.video_frame.setPixmap(pixmap)
    
    def format_time(self, seconds):
        """Format time in seconds to MM:SS format"""
        minutes = int(seconds / 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def closeEvent(self, event):
        """Handle component close event"""
        # Clean up resources
        if self.cap is not None:
            self.cap.release()
        
        if self.timer.isActive():
            self.timer.stop()
        
        event.accept()
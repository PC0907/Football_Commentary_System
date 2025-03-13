import cv2
import os
from typing import Generator, Tuple
from utils.file_utils import create_directory

class VideoProcessor:
    def __init__(self, video_path: str, output_dir: str = "outputs"):
        """
        Initialize video processor.

        Args:
            video_path: Path to input video file.
            output_dir: Directory to save processed outputs.
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.cap = None
        self.writer = None
        self.frame_count = 0
        self.fps = 0
        self.width = 0
        self.height = 0

    def __enter__(self):
        """Context manager entry: Open video and initialize resources."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")

        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = 0

        # Create output directory
        create_directory(self.output_dir)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: Release resources."""
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()

    def get_video_properties(self) -> dict:
        """Return video metadata."""
        return {
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "total_frames": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }

    def frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Generator to yield frames from the video.

        Yields:
            Tuple of (frame_number, frame)
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame_count += 1
            yield self.frame_count, frame

    def initialize_writer(self, output_name: str, fps: int = None):
        """
        Initialize video writer for saving processed frames.

        Args:
            output_name: Name of output video file.
            fps: Frames per second (defaults to input video FPS).
        """
        if fps is None:
            fps = self.fps

        output_path = os.path.join(self.output_dir, output_name)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            output_path, fourcc, fps, (self.width, self.height)
        return self.writer

    def write_frame(self, frame: np.ndarray):
        """Write a frame to the output video."""
        if self.writer is None:
            raise RuntimeError("Video writer not initialized. Call initialize_writer() first.")
        self.writer.write(frame)

    def save_frame(self, frame: np.ndarray, name: str):
        """
        Save a single frame as an image.

        Args:
            frame: Frame to save.
            name: Name of the output image file.
        """
        output_path = os.path.join(self.output_dir, name)
        cv2.imwrite(output_path, frame)

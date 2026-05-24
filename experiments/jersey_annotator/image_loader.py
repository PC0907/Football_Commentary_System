import os
from csv_handler import CSVHandler
from PyQt5.QtWidgets import QMessageBox

class ImageLoader:
    def __init__(self):
        self.image_list = []
        self.index = 0
        self.annotations = {}

    def load_images(self, folder, output_folder, parent_widget):
        """
        Loads images from a directory and resumes from last unannotated image.
        If a previous session exists, alerts the user via a dialog box.
        """
        self.image_list = sorted([os.path.join(folder, img) 
                                  for img in os.listdir(folder) 
                                  if img.lower().endswith(('.png', '.jpg', '.jpeg'))])

        # Load previous annotations
        csv_handler = CSVHandler()
        self.annotations = csv_handler.load_existing_annotations(output_folder)

        last_unannotated_index = self.find_resume_index()
        
        # Notify user if previous session exists
        if last_unannotated_index > 0:
            QMessageBox.information(parent_widget, "Previous Session Found",
                                    f"Resuming from image {last_unannotated_index + 1} of {len(self.image_list)}.")

        self.index = last_unannotated_index  # Start from the first unannotated image

    def find_resume_index(self):
        """
        Finds the first unannotated image index to resume from.
        """
        for i, image_path in enumerate(self.image_list):
            if os.path.basename(image_path) not in self.annotations:
                return i
        return len(self.image_list)  # All images annotated

    def get_current_image(self):
        """
        Returns the current image path.
        """
        if self.image_list and self.index < len(self.image_list):
            return self.image_list[self.index]
        return None

    def next_image(self):
        """
        Moves to the next image.
        """
        if self.index < len(self.image_list) - 1:
            self.index += 1

    def prev_image(self):
        """
        Moves to the previous image.
        """
        if self.index > 0:
            self.index -= 1

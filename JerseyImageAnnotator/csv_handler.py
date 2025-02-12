import os
import csv
import time

class CSVHandler:
    def __init__(self):
        self.file_name = "annotations.csv"

    def save_annotation(self, image_path, label, output_folder, session_id):
        """
        Appends a new annotation to the CSV file.
        Ensures existing annotations are preserved.
        Includes session tracking and timestamps.
        """
        csv_path = os.path.join(output_folder, self.file_name)
        exists = os.path.isfile(csv_path)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        with open(csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            if not exists:
                writer.writerow(["image_name", "label", "session_id", "timestamp"])  # Add header if new file
            writer.writerow([os.path.basename(image_path), label, session_id, timestamp])

    def load_existing_annotations(self, output_folder):
        """
        Loads existing annotations from a CSV file.
        Returns a dictionary {image_name: (label, session_id, timestamp)}.
        """
        csv_path = os.path.join(output_folder, self.file_name)
        annotations = {}

        if os.path.isfile(csv_path):
            with open(csv_path, 'r') as file:
                reader = csv.reader(file)
                next(reader, None)  # Skip header if present
                for row in reader:
                    if len(row) == 4:
                        annotations[row[0]] = (row[1], row[2], row[3])  # Store label, session_id, timestamp
        return annotations

import os
import re
from PIL import Image

# Define the current folder as the working directory
folder_path = os.getcwd()  # Gets the current folder path
output_format = "jpg"

# Regular expression to match existing image names in the format: IMG_test_XXXXXXXX.jpg
pattern = re.compile(r"IMG_test_(\d{8})\.jpg$", re.IGNORECASE)

# Get all image files in the current folder (including .webp)
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpeg', 'jpg', 'bmp', 'gif', 'tiff', 'webp'))]

# Separate files into already formatted and unformatted
existing_images = []
unformatted_images = []

for file_name in image_files:
    match = pattern.match(file_name)
    if match:
        existing_images.append(int(match.group(1)))  # Store only the numeric part
    else:
        unformatted_images.append(file_name)  # Needs renaming or conversion

# Sort existing images numerically
existing_images.sort()
'''
# Find the first missing number in the sequence
missing_numbers = []
expected_number = 1  # Start checking from 1

for num in existing_images:
    while expected_number < num:
        missing_numbers.append(expected_number)
        expected_number += 1
    expected_number += 1  # Skip to the next expected number
'''
# Process unformatted images (convert and rename)
max_index = max(existing_images) if existing_images else 0

for file_name in unformatted_images:
    file_path = os.path.join(folder_path, file_name)

    # Generate a new sequential filename
    max_index += 1
    new_file_name = f"IMG_test_{max_index:08d}.jpg"
    new_file_path = os.path.join(folder_path, new_file_name)

    # Convert and save the image as JPG
    with Image.open(file_path) as img:
        img = img.convert("RGB")  # Ensure compatibility
        img.save(new_file_path, format="JPEG", quality=95)

    print(f"Converted: {file_name} → {new_file_name}")

    # Add newly named file to existing list for future numbering
    existing_images.append(max_index)
'''
# If there are missing numbers, move the last image to replace the first missing slot
if missing_numbers:
    for missing_number in missing_numbers:
        last_number = max(existing_images)
        if last_number == missing_number:  # Prevent replacing the same image
            continue

        last_image_name = f"IMG_test_{last_number:08d}.jpg"
        missing_image_name = f"IMG_test_{missing_number:08d}.jpg"

        last_image_path = os.path.join(folder_path, last_image_name)
        missing_image_path = os.path.join(folder_path, missing_image_name)

        # Move the last image to the missing slot
        if os.path.exists(last_image_path):
            os.rename(last_image_path, missing_image_path)
            print(f"Moved: {last_image_name} → {missing_image_name}")

        # Update lists to reflect changes
        existing_images.remove(last_number)
        existing_images.append(missing_number)
'''
print("Processing complete!")


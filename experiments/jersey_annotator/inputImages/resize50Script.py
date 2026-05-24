import os
from PIL import Image

def downscale_all_images(directory, output_folder="resizedImages", size=(50, 50)):
    # Get absolute paths
    directory = os.path.abspath(directory)
    output_path = os.path.join(directory, output_folder)

    # Create 'resizedImages' folder if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Get list of all .jpg and .jpeg files (case-insensitive)
    images = [f for f in os.listdir(directory) if f.lower().endswith((".jpg", ".jpeg"))]

    if not images:
        print(f"No JPG images found in {directory}")
        return

    print(f"Found {len(images)} JPG images. Processing...\n")

    # Process each image
    for img_name in images:
        img_path = os.path.join(directory, img_name)
        
        # Open the image
        img = Image.open(img_path)
        
        # Resize the image (LANCZOS for high quality)
        img_resized = img.resize(size, Image.LANCZOS)
        
        # Save to the 'resizedImages' folder with the same filename
        resized_img_path = os.path.join(output_path, img_name)
        img_resized.save(resized_img_path)
        
        print(f"✅ Resized and saved: {resized_img_path}")

# Run the function on the current directory
current_directory = os.getcwd()
print(f"🔍 Searching for JPG images in: {current_directory}")
downscale_all_images(current_directory)

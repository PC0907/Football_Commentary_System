import fire
import os
import lmdb
import cv2
import numpy as np
import pandas as pd
import glob

def checkImageIsValid(imageBin):
    """Validate image binary data"""
    if imageBin is None:
        return False
    try:
        imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        return img is not None and img.size > 0
    except Exception as e:
        print(f"Image validation error: {str(e)}")
        return False

def writeCache(env, cache):
    """Write batch to LMDB"""
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def createDataset(inputPath, csvFile, outputPath, 
                 imageCol='image_name', labelCol='label', 
                 checkValid=True, logInterval=1000):
    """
    Create LMDB dataset with metadata tracking
    
    Args:
        inputPath   : Root directory containing images
        csvFile     : CSV with image names and labels
        outputPath  : LMDB output directory
        imageCol    : CSV column for image filenames
        labelCol    : CSV column for labels
        checkValid  : Verify image validity
        logInterval : Progress logging interval
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    
    # Read CSV and prepare data
    df = pd.read_csv(csvFile)
    total_samples = len(df)
    print(f"Processing {total_samples} CSV entries")
    
    # Build image path index
    search_pattern = os.path.join(inputPath, "**", "*.jpg")
    image_paths = glob.glob(search_pattern, recursive=True)
    path_index = {os.path.basename(p).lower(): p for p in image_paths}
    print(f"Found {len(image_paths)} images in {inputPath}")

    # Error logging
    error_log = []
    missing_images = []

    for csv_idx, row in df.iterrows():
        base_name = row[imageCol].strip()
        label = str(row[labelCol]).strip()
        
        # Clean base name for matching
        clean_name = base_name.lower().replace('.jpg', '')
        
        # Find best match considering possible augmentations
        best_match = None
        for filename in path_index:
            if filename.startswith(clean_name):
                best_match = path_index[filename]
                break  # Take first match
        
        if not best_match:
            missing_images.append(base_name)
            error_log.append(f"Missing: {base_name} (CSV row {csv_idx})")
            continue
            
        # Read image data
        try:
            with open(best_match, 'rb') as f:
                imageBin = f.read()
                
            if checkValid and not checkImageIsValid(imageBin):
                error_log.append(f"Invalid: {best_match} (CSV row {csv_idx})")
                continue
                
        except Exception as e:
            error_log.append(f"Read Error: {best_match} | {str(e)}")
            continue

        # Store data with metadata
        imageKey = f'image-{cnt:09d}'.encode()
        labelKey = f'label-{cnt:09d}'.encode()
        metaKey = f'meta-{cnt:09d}'.encode()
        
        cache.update({
            imageKey: imageBin,
            labelKey: label.encode(),
            metaKey: f"{csv_idx},{base_name}".encode()
        })

        # Batch writing
        if cnt % logInterval == 0:
            writeCache(env, cache)
            cache = {}
            print(f"Processed {cnt}/{total_samples} | "
                  f"Missing: {len(missing_images)} | "
                  f"Errors: {len(error_log)-len(missing_images)}")

        cnt += 1

    # Final write
    cache['num-samples'.encode()] = str(cnt-1).encode()
    writeCache(env, cache)
    
    # Save error logs
    with open(os.path.join(outputPath, 'creation_errors.log'), 'w') as f:
        f.write("\n".join(error_log))
        
    print(f"\nDataset creation complete")
    print(f"Successfully stored: {cnt-1} samples")
    print(f"Missing images: {len(missing_images)}")
    print(f"Other errors: {len(error_log)-len(missing_images)}")
    print(f"Error log saved to: {os.path.join(outputPath, 'creation_errors.log')}")

if __name__ == '__main__':
    fire.Fire(createDataset)

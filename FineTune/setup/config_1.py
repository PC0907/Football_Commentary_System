import fire
import os
import lmdb
import cv2
import numpy as np
import pandas as pd
import glob

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    try:
        imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        imgH, imgW = img.shape[0], img.shape[1]
        return imgH * imgW > 0
    except Exception as e:
        print(f"Image validation error: {e}")
        return False

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def createDataset(inputPath, csvFile, outputPath, imageCol='image_name', labelCol='label', checkValid=True):
    """
    Updated version that:
    1. Handles augmented filenames (_augX suffixes)
    2. Searches nested directories recursively
    3. Matches partial filenames from CSV
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    
    # Read CSV and prepare image mapping
    df = pd.read_csv(csvFile)
    nSamples = len(df)
    
    # Build image path cache (filename -> full paths)
    search_pattern = os.path.join(inputPath, "**", "*.jpg")
    all_image_paths = glob.glob(search_pattern, recursive=True)
    image_map = {}
    for path in all_image_paths:
        filename = os.path.basename(path)
        if filename not in image_map:
            image_map[filename] = []
        image_map[filename].append(path)
    
    print(f"Found {len(all_image_paths)} images in {inputPath} (including subdirectories)")
    
    # Process CSV entries
    for i, row in df.iterrows():
        base_name = row[imageCol].strip()
        label = str(row[labelCol]).strip()
        
        # Remove .jpg extension if present in CSV
        if base_name.lower().endswith('.jpg'):
            base_name = base_name[:-4]
        
        # Find matching images (handles _augX versions)
        matches = []
        for img_file in image_map.keys():
            if img_file.startswith(base_name):
                matches.extend(image_map[img_file])
        
        if not matches:
            print(f'No matches found for {base_name}')
            continue
            
        # Use first match (or implement your selection logic)
        imagePath = matches[0]
        
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
            
        if checkValid:
            if not checkImageIsValid(imageBin):
                print(f'Skipping invalid image: {imagePath}')
                continue
                
        # Store in LMDB
        imageKey = f'image-{cnt:09d}'.encode()
        labelKey = f'label-{cnt:09d}'.encode()
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        
        # Write batch
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print(f'Processed {cnt}/{nSamples} samples')
            
        cnt += 1
        
    # Final write
    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print(f'Successfully created dataset with {nSamples} samples')
    print(f'Original CSV had {len(df)} entries, matched {nSamples} images')

if __name__ == '__main__':
    fire.Fire(createDataset)

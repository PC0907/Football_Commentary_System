import fire
import os
import lmdb
import cv2
import numpy as np
import pandas as pd

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def createDataset(inputPath, csvFile, outputPath, imageCol='image_name', labelCol='label', checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        csvFile    : CSV file with image names and labels
        imageCol   : Name of the column containing image filenames
        labelCol   : Name of the column containing labels
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    
    # Read CSV file
    df = pd.read_csv(csvFile)
    nSamples = len(df)
    
    # Check if we need to look in a nested output_images directory
    nested_dir = os.path.join(inputPath, 'output_images')
    use_nested = os.path.isdir(nested_dir)
    
    for i, row in df.iterrows():
        img_name = row[imageCol]
        label = str(row[labelCol])
        
        # Try multiple possible paths
        potential_paths = [
            os.path.join(inputPath, img_name),  # Direct path
            os.path.join(inputPath, 'output_images', img_name)  # Nested path
        ]
        
        imagePath = None
        for path in potential_paths:
            if os.path.exists(path):
                imagePath = path
                break
                
        if imagePath is None:
            print(f'Could not find image {img_name} in any of the expected locations')
            continue
            
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
            
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except Exception as e:
                print(f'Error occurred with image {imagePath}: {e}')
                with open(os.path.join(outputPath, 'error_image_log.txt'), 'a') as log:
                    log.write(f'Error with image {imagePath}: {e}\n')
                continue
                
        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
            
        cnt += 1
        
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

if __name__ == '__main__':
    fire.Fire(createDataset)

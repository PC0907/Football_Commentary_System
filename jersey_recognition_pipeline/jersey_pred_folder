#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# Add the parseq directory to Python path
parseq_path = '/kaggle/working/parseq'
sys.path.append(parseq_path)

# Now import from strhub
from PIL import Image
import torch
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args

@torch.inference_mode()
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--images', nargs='+', help='Images of jersey numbers to read')
    parser.add_argument('--folder', help='Folder containing images to process')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', help='Output file to save predictions')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    
    # Ensure either --images or --folder is provided
    if not args.images and not args.folder:
        print("Error: Either --images or --folder must be specified")
        return
    
    print(f'Loading model from {args.checkpoint}...')
    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    
    # Get list of image files to process
    image_files = []
    if args.images:
        image_files.extend(args.images)
    
    if args.folder:
        # Get all image files from the folder
        folder_path = Path(args.folder)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        for ext in image_extensions:
            image_files.extend(list(folder_path.glob(f'*{ext}')))
            image_files.extend(list(folder_path.glob(f'*{ext.upper()}')))
    
    if not image_files:
        print("No image files found to process")
        return
    
    print(f"Processing {len(image_files)} images...")
    
    results = {}
    for fname in image_files:
        fname = str(fname)  # Convert Path to string if needed
        try:
            # Load image and prepare for input
            image = Image.open(fname).convert('RGB')
            image = img_transform(image).unsqueeze(0).to(args.device)
            
            # Get prediction
            p = model(image).softmax(-1)
            pred, p = model.tokenizer.decode(p)
            predicted_number = pred[0]
            
            # Check if prediction is purely numeric
            if predicted_number.isdigit():
                results[fname] = predicted_number
                print(f'{Path(fname).name}: Jersey #{predicted_number}')
            else:
                # Try to extract numeric part
                numeric_part = ''.join(char for char in predicted_number if char.isdigit())
                if numeric_part:
                    results[fname] = numeric_part
                    print(f'{Path(fname).name}: Jersey #{numeric_part} (cleaned from "{predicted_number}")')
                else:
                    results[fname] = None
                    print(f'{Path(fname).name}: No valid jersey number detected in "{predicted_number}"')
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            results[fname] = None
    
    # Save results if output file is specified
    if args.output:
        with open(args.output, 'w') as f:
            f.write("filename,jersey_number\n")  # Header
            for fname, number in results.items():
                f.write(f'{Path(fname).name},{number if number else "unknown"}\n')
        print(f'Results saved to {args.output}')

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import argparse
from pathlib import Path
from PIL import Image
import torch
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint (or 'pretrained=<model_id>')")
    parser.add_argument('--images', nargs='+', help='Images of jersey numbers to read')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', help='Output file to save predictions')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)
    
    print(f'Loading model from {args.checkpoint}...')
    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    
    results = {}
    for fname in args.images:
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
            print(f'{fname}: Jersey #{predicted_number}')
        else:
            # Try to extract numeric part
            numeric_part = ''.join(char for char in predicted_number if char.isdigit())
            if numeric_part:
                results[fname] = numeric_part
                print(f'{fname}: Jersey #{numeric_part} (cleaned from "{predicted_number}")')
            else:
                results[fname] = None
                print(f'{fname}: No valid jersey number detected in "{predicted_number}"')
    
    # Save results if output file is specified
    if args.output:
        with open(args.output, 'w') as f:
            for fname, number in results.items():
                f.write(f'{Path(fname).name},{number if number else "unknown"}\n')
        print(f'Results saved to {args.output}')

if __name__ == '__main__':
    main()
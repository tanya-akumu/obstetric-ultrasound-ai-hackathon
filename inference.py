"""
Inference script for generating submission file.

This script:
1. Loads a trained model
2. Runs inference on test data
3. Generates a CSV submission file with columns: patient_id, site, predicted_GA
"""

import torch
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm

# Import model and dataset from separate modules
from model import GAPredictor, get_model
from dataset import BlindsweepDataset, get_transforms


def run_inference(model, dataloader, device):
    """
    Run inference on test data.
    
    Returns:
        results: list of dicts with patient_id, site, predicted_GA
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for frames, _, metadata in tqdm(dataloader, desc='Running inference'):
            frames = frames.to(device)
            
            # Get predictions
            predictions = model(frames)  # (batch_size, 1)
            predictions = predictions.squeeze(1).cpu().numpy()  # (batch_size,)
            
            # Store results
            for i in range(len(predictions)):
                results.append({
                    'patient_id': metadata['patient_id'][i],
                    'site': metadata['site'][i],
                    'predicted_GA': float(predictions[i])
                })
    
    return results


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = get_model(model_type='baseline', pretrained=False).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if 'val_mae' in checkpoint:
        print(f"Model validation MAE: {checkpoint['val_mae']:.4f} days")
    
    # Test transforms (no augmentation)
    test_transform = get_transforms(mode='test')
    
    # Create test dataset
    # Note: test_csv should have columns [patient_id, sweep_path, site]
    # gestational_age_days can be dummy values (e.g., 0) since we're just doing inference
    test_dataset = BlindsweepDataset(
        data_csv=args.test_csv,
        video_root=args.video_root,
        num_frames_per_sweep=args.num_frames,
        transform=test_transform,
        mode='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Run inference
    results = run_inference(model, test_loader, device)
    
    # Create submission dataframe
    submission_df = pd.DataFrame(results)
    
    # Ensure column order
    submission_df = submission_df[['patient_id', 'site', 'predicted_GA']]
    
    # Round predictions to 2 decimal places
    submission_df['predicted_GA'] = submission_df['predicted_GA'].round(2)
    
    # Save submission file
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    
    print(f"\n Submission file saved to: {output_path}")
    print(f"  Total predictions: {len(submission_df)}")
    print(f"\nSubmission preview:")
    print(submission_df.head(10))
    
    # Summary statistics
    print(f"\nPrediction statistics:")
    print(f"  Mean GA: {submission_df['predicted_GA'].mean():.2f} days")
    print(f"  Std GA: {submission_df['predicted_GA'].std():.2f} days")
    print(f"  Min GA: {submission_df['predicted_GA'].min():.2f} days")
    print(f"  Max GA: {submission_df['predicted_GA'].max():.2f} days")
    
    print(f"\nPredictions by site:")
    site_stats = submission_df.groupby('site')['predicted_GA'].agg(['count', 'mean', 'std'])
    print(site_stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate submission file from trained model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test CSV')
    parser.add_argument('--video_root', type=str, required=True, help='Root directory for videos')
    parser.add_argument('--output_path', type=str, default='submission.csv', help='Output submission file path')
    parser.add_argument('--num_frames', type=int, default=16, help='Frames to sample per sweep')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    args = parser.parse_args()
    
    main(args)
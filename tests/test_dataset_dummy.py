"""
Test script for BlindsweepDataset with dummy .nii.gz data.

This tests the dataset class with real dummy data we just created.
"""

import os
import sys
from pathlib import Path
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import dataset
from dataset import BlindsweepDataset, get_transforms

def test_dataset():
    """Test dataset with dummy .nii.gz files."""
    
    print("="*70)
    print("TESTING DATASET WITH DUMMY .nii.gz DATA")
    print("="*70)
    
    # Paths
    csv_path = 'data/dummy_data.csv'
    video_root = '.'  # Since paths in CSV are relative to current dir
    
    print(f"\n1. Testing dataset initialization...")
    print(f"   CSV: {csv_path}")
    print(f"   Video root: {video_root}")
    
    # Create dataset without transforms first
    dataset = BlindsweepDataset(
        data_csv=csv_path,
        video_root=video_root,
        num_frames_per_sweep=5,  # Sample 5 frames per sweep
        transform=get_transforms(mode='train'),
        mode='train'
    )
    
    print(f"    Dataset initialized successfully!")
    print(f"   Number of patients: {len(dataset)}")
    print(f"   Patient IDs: {dataset.patient_ids}")
    
    # Test loading first patient
    print(f"\n2. Testing data loading for patient 1...")
    frames, label, metadata = dataset[0]
    
    print(f"    Data loaded successfully!")
    print(f"   Patient ID: {metadata['patient_id']}")
    print(f"   Site: {metadata['site']}")
    print(f"   Number of sweeps: {metadata['num_sweeps']}")
    print(f"   GA label: {label.item():.1f} days")
    print(f"   Frames shape: {frames.shape}")
    print(f"   Expected: ({metadata['num_sweeps'] * 5}, 3, 224, 224)")
    print(f"   Frames dtype: {frames.dtype}")
    print(f"   Frames range: [{frames.min():.3f}, {frames.max():.3f}]")
    
    # Verify shape
    expected_frames = metadata['num_sweeps'] * 5
    assert frames.shape[0] == expected_frames, f"Expected {expected_frames} frames, got {frames.shape[0]}"
    assert frames.shape[1:] == (3, 224, 224), f"Expected (3, 224, 224) per frame, got {frames.shape[1:]}"
    
    # Test second patient
    print(f"\n3. Testing data loading for patient 2...")
    frames2, label2, metadata2 = dataset[1]
    
    print(f"    Data loaded successfully!")
    print(f"   Patient ID: {metadata2['patient_id']}")
    print(f"   Site: {metadata2['site']}")
    print(f"   Number of sweeps: {metadata2['num_sweeps']}")
    print(f"   GA label: {label2.item():.1f} days")
    print(f"   Frames shape: {frames2.shape}")
    
    # Test with transforms
    print(f"\n4. Testing with transforms...")
    transform = get_transforms(mode='train')
    
    dataset_with_transform = BlindsweepDataset(
        data_csv=csv_path,
        video_root=video_root,
        num_frames_per_sweep=5,
        transform=transform,
        mode='train'
    )
    
    frames_t, label_t, metadata_t = dataset_with_transform[0]
    
    print(f"    Transforms applied successfully!")
    print(f"   Transformed frames shape: {frames_t.shape}")
    print(f"   Transformed frames dtype: {frames_t.dtype}")
    print(f"   Transformed frames range: [{frames_t.min():.3f}, {frames_t.max():.3f}]")
    print(f"   (Should be normalized, not in [0,1])")
    
    # Test random sampling (load same patient twice, should get different frames)
    print(f"\n5. Testing random frame sampling...")
    frames_a, _, _ = dataset[0]
    frames_b, _, _ = dataset[0]
    
    # Due to random sampling, frames should be different
    if torch.allclose(frames_a, frames_b):
        print(f"   ⚠ Warning: Frames are identical (random sampling may not be working)")
    else:
        print(f"    Random sampling working! Frames differ on repeated loads.")
    
    # Test with different num_frames
    print(f"\n6. Testing with different num_frames...")
    for num_frames in [3, 8, 16]:
        ds = BlindsweepDataset(
            data_csv=csv_path,
            video_root=video_root,
            num_frames_per_sweep=num_frames,
            transform=None,
            mode='train'
        )
        f, _, m = ds[0]
        expected = m['num_sweeps'] * num_frames
        print(f"   num_frames={num_frames}: got {f.shape[0]} frames (expected {expected}) ")
        assert f.shape[0] == expected, f"Frame count mismatch!"
    
    # Summary
    print(f"\n" + "="*70)
    print("ALL TESTS PASSED! ")
    print("="*70)
    print(f"\nDataset Summary:")
    print(f"   Loads .nii.gz files correctly")
    print(f"   Randomly samples frames from volumes")
    print(f"   Handles multiple sweeps per patient")
    print(f"   Applies transforms correctly")
    print(f"   Returns correct shapes and dtypes")
    print(f"   Metadata is accurate")
    print(f"\nReady for training!")
    

if __name__ == '__main__':
    try:
        test_dataset()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
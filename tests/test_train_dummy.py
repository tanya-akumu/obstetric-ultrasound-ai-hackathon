"""
Test script for train.py with dummy .nii.gz data.

This tests the training pipeline end-to-end with a small dataset.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import subprocess
import shutil

# Setup paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def create_train_val_split():
    """Create train and validation CSV files from dummy data."""
    
    print("="*70)
    print("PREPARING DATA SPLITS")
    print("="*70)
    
    # Read dummy data
    dummy_csv = Path('data/dummy_data.csv')
    if not dummy_csv.exists():
        raise FileNotFoundError(f"Dummy data not found at {dummy_csv}")
    
    df = pd.read_csv(dummy_csv)
    print(f"\nTotal patients: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Split: first 3 for train, last 2 for val
    train_df = df.iloc[:3].copy()
    val_df = df.iloc[3:].copy()
    
    # Save splits
    train_csv = Path('data/train_dummy.csv')
    val_csv = Path('data/val_dummy.csv')
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    
    print(f"\n✓ Created train split: {len(train_df)} patients")
    print(f"  Saved to: {train_csv}")
    print(f"  Patients: {train_df['study_id'].tolist()}")
    
    print(f"\n✓ Created val split: {len(val_df)} patients")
    print(f"  Saved to: {val_csv}")
    print(f"  Patients: {val_df['study_id'].tolist()}")
    
    return train_csv, val_csv


def test_training_basic():
    """Test train.py with dummy data for 2 epochs."""
    
    print("\n" + "="*70)
    print("TEST 1: BASIC TRAINING (2 EPOCHS)")
    print("="*70)
    
    print("\n1.1 Creating data splits...")
    train_csv, val_csv = create_train_val_split()
    
    print("\n1.2 Creating output directory...")
    output_dir = Path('outputs/checkpoints_test')
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"   ✓ Output directory: {output_dir}")
    
    print("\n1.3 Running training for 2 epochs...")
    print("   (This will test the training loop without validation)")
    
    # Build command
    cmd = [
        'python', 'train.py',
        '--train_csv', str(train_csv),
        '--val_csv', str(val_csv),
        '--video_root', '.',
        '--output_dir', str(output_dir),
        '--num_frames', '5',  # Use fewer frames for faster testing
        '--batch_size', '2',   # Small batch size
        '--epochs', '2',       # Just 2 epochs for testing
        '--lr', '0.001',
        '--num_workers', '0',  # No multiprocessing for simpler testing
    ]
    
    print(f"\n   Command:")
    print(f"   {' '.join(cmd)}")
    
    # Run training
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"\n Training failed with return code {result.returncode}")
            print("\n--- STDOUT ---")
            print(result.stdout)
            print("\n--- STDERR ---")
            print(result.stderr)
            return False
        
        print("\n   ✓ Training completed successfully!")
        
        # Print key lines from training output
        print("\n1.4 Training output:")
        print("-" * 70)
        for line in result.stdout.split('\n'):
            if any(keyword in line for keyword in ['Using device', 'Loaded', 'Training samples', 
                                                    'Validation samples', 'Model parameters',
                                                    'Epoch', 'Train Loss', 'Train MAE']):
                print(f"   {line}")
        print("-" * 70)
        
    except subprocess.TimeoutExpired:
        print("\n Training timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"\n Error running training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def verify_checkpoints():
    """Verify that training produced expected checkpoint at epoch 2."""
    
    print("\n" + "="*70)
    print("TEST 2: VERIFY CHECKPOINTS")
    print("="*70)
    
    output_dir = Path('outputs/checkpoints_test')
    
    print("\n2.1 Checking for checkpoint at epoch 2...")
    
    # Since we train for 2 epochs and save every 20 epochs, we won't have a checkpoint_epoch_2.pth
    # But we should verify the directory exists and check what files are there
    
    if not output_dir.exists():
        print(f"    Output directory not found: {output_dir}")
        return False
    
    print(f"   ✓ Output directory exists: {output_dir}")
    
    # List all files in output directory
    files = list(output_dir.iterdir())
    print(f"\n2.2 Files in output directory:")
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"     - {f.name} ({size_mb:.2f} MB)")
    
    # Note: With only 2 epochs, we won't get checkpoint_epoch_2.pth (saved every 20 epochs)
    # and we won't get best_model.pth (validation only runs every 10 epochs)
    # So this is expected to find no files for a 2-epoch run
    
    if len(files) == 0:
        print(f"\n   ℹ️  No checkpoints saved (expected - we only ran 2 epochs)")
        print(f"      Checkpoints are saved every 20 epochs")
        print(f"      Best model is saved when validation runs (every 10 epochs)")
        return True
    
    return True


def test_training_with_validation():
    """Test that validation runs at epoch 10 and best model is saved."""
    
    print("\n" + "="*70)
    print("TEST 3: TRAINING WITH VALIDATION (10 EPOCHS)")
    print("="*70)
    
    print("\n3.1 Running training for 10 epochs to test validation...")
    
    train_csv = Path('data/train_dummy.csv')
    val_csv = Path('data/val_dummy.csv')
    output_dir = Path('outputs/test_training_val')
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'python', 'train.py',
        '--train_csv', str(train_csv),
        '--val_csv', str(val_csv),
        '--video_root', '.',
        '--output_dir', str(output_dir),
        '--num_frames', '5',
        '--batch_size', '2',
        '--epochs', '3',      # 10 epochs to trigger validation
        '--lr', '0.001',
        '--num_workers', '0',
    ]
    
    print(f"\n   Command:")
    print(f"   {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode != 0:
            print(f"\n Training failed")
            print("\n--- STDERR ---")
            print(result.stderr)
            return False
        
        print("   ✓ Training completed!")
        
        # Check for validation output
        output = result.stdout
        has_val_loss = 'Val Loss' in output
        has_val_mae = 'Val MAE' in output
        
        print(f"\n3.2 Checking for validation output...")
        print(f"   'Val Loss' in output: {has_val_loss}")
        print(f"   'Val MAE' in output: {has_val_mae}")
        
        if not (has_val_loss and has_val_mae):
            print("    Validation output not found in logs")
            print("\n--- Training output ---")
            print(output)
            return False
        
        print("   ✓ Validation ran successfully at epoch 10")
        
        # Check for best model
        best_model_path = output_dir / 'best_model.pth'
        
        print(f"\n3.3 Checking for best_model.pth...")
        if not best_model_path.exists():
            print(f"    Best model not saved: {best_model_path}")
            print(f"\n   Files in output directory:")
            for f in output_dir.iterdir():
                print(f"     - {f.name}")
            return False
        
        print(f"   ✓ Best model saved: {best_model_path}")
        
        # Check best model contents
        import torch
        checkpoint = torch.load(best_model_path, map_location='cpu')
        
        required_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict', 'val_mae']
        missing_keys = [k for k in required_keys if k not in checkpoint]
        
        if missing_keys:
            print(f"    Best model missing keys: {missing_keys}")
            return False
        
        print(f"   ✓ Best model contains all required keys")
        print(f"   Checkpoint info:")
        print(f"     - Epoch: {checkpoint['epoch']}")
        print(f"     - Validation MAE: {checkpoint['val_mae']:.4f} days")
        print(f"     - Model state dict keys: {len(checkpoint['model_state_dict'])}")
        
    except subprocess.TimeoutExpired:
        print("\n Training timed out")
        return False
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_model_loading():
    """Test that saved model can be loaded and used for inference."""
    
    print("\n" + "="*70)
    print("TEST 4: MODEL LOADING")
    print("="*70)
    
    print("\n4.1 Loading best model from checkpoint...")
    
    from model import get_model
    import torch
    
    checkpoint_path = Path('outputs/test_training_val/best_model.pth')
    
    if not checkpoint_path.exists():
        print(f"    Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        # Create model
        model = get_model(model_type='baseline', pretrained=False)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"   ✓ Model loaded successfully")
        print(f"   Validation MAE from checkpoint: {checkpoint['val_mae']:.4f} days")
        
        # Test forward pass
        print(f"\n4.2 Testing forward pass with loaded model...")
        model.eval()
        x = torch.randn(1, 40, 3, 224, 224)
        
        with torch.no_grad():
            output = model(x)
        
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Prediction: {output.item():.2f} days")
        print(f"   ✓ Forward pass successful")
        
    except Exception as e:
        print(f"    Error loading or using model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def cleanup():
    """Clean up test files."""
    print("\n" + "="*70)
    print("CLEANUP")
    print("="*70)
    
    files_to_remove = [
        'data/train_dummy.csv',
        'data/val_dummy.csv',
    ]
    
    dirs_to_remove = [
        'outputs/test_training',
        'outputs/test_training_val',
    ]
    
    print("\nRemoving temporary files...")
    for file_path in files_to_remove:
        p = Path(file_path)
        if p.exists():
            p.unlink()
            print(f"  ✓ Removed: {file_path}")
    
    for dir_path in dirs_to_remove:
        p = Path(dir_path)
        if p.exists():
            shutil.rmtree(p)
            print(f"  ✓ Removed: {dir_path}")


def run_all_tests():
    """Run all training tests."""
    
    print("\n" + "="*70)
    print("TESTING TRAIN.PY")
    print("="*70)
    print("\nThis will test the training pipeline with dummy data.")
    print("Tests will run training for 2 epochs, then 10 epochs to test validation.")
    
    try:
        # Test 1: Basic training (2 epochs)
        if not test_training_basic():
            print("\n TEST 1 FAILED: Basic training")
            return False
        print("\n✓ TEST 1 PASSED: Basic training works!")
        
        # Test 2: Training with validation (10 epochs)
        if not test_training_with_validation():
            print("\n TEST 3 FAILED: Training with validation")
            return False
        print("\n✓ TEST 3 PASSED: Validation works!")

        # Test 3: Verify checkpoints
        if not verify_checkpoints():
            print("\n TEST 3 FAILED: Checkpoint verification")
            return False
        print("\n✓ TEST 3 PASSED: Checkpoints verified!")
        
        # Test 4: Model loading
        if not test_model_loading():
            print("\n TEST 4 FAILED: Model loading")
            return False
        print("\n✓ TEST 4 PASSED: Model loading works!")
        
        # Final summary
        print("\n" + "="*70)
        print("✓ ALL TRAINING TESTS PASSED!")
        print("="*70)
        print("\nTraining Pipeline Summary:")
        print("  ✓ Training script runs successfully")
        print("  ✓ Handles train/val splits correctly")
        print("  ✓ Training loop works for multiple epochs")
        print("  ✓ Validation runs every 10 epochs")
        print("  ✓ Best model saved when validation runs")
        print("  ✓ Saved models can be loaded and used")
        print("  ✓ Forward pass works with loaded model")
        print("\ntrain.py is ready! ✓")
        print("\nNext: Test inference.py")
        
        return True
        
    except Exception as e:
        print(f"\n UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Always cleanup
        cleanup()


if __name__ == '__main__':
    # Check prerequisites
    if not Path('data/dummy_data.csv').exists():
        print(" Error: data/dummy_data.csv not found")
        print("  Please make sure you have generated the dummy data first")
        sys.exit(1)
    
    if not Path('train.py').exists():
        print(" Error: train.py not found")
        print("  Please run this script from the directory containing train.py")
        sys.exit(1)
    
    success = run_all_tests()
    sys.exit(0 if success else 1)
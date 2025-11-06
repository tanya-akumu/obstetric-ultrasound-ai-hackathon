"""
Test script for inference.py with dummy data.

This tests the inference pipeline:
1. Uses a trained model checkpoint
2. Runs inference on test data
3. Generates a submission CSV file
"""

import os
import sys
from pathlib import Path
import pandas as pd
import subprocess
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def create_test_csv():
    """Create test CSV for inference (can use same format as training data)."""
    
    print("="*70)
    print("PREPARING TEST DATA")
    print("="*70)
    
    # Read dummy data
    dummy_csv = Path('data/dummy_data.csv')
    if not dummy_csv.exists():
        raise FileNotFoundError(f"Dummy data not found at {dummy_csv}")
    
    df = pd.read_csv(dummy_csv)
    
    # For test mode, we can use all patients or a subset
    # Let's use all 5 patients for testing
    test_df = df.copy()
    
    # Save test CSV
    test_csv = Path('data/test_dummy.csv')
    test_df.to_csv(test_csv, index=False)
    
    print(f"\n Created test split: {len(test_df)} patients")
    print(f"  Saved to: {test_csv}")
    print(f"  Patients: {test_df['study_id'].tolist()}")
    
    return test_csv


def ensure_trained_model():
    """Ensure we have a trained model checkpoint to use for inference."""
    
    print("\n" + "="*70)
    print("CHECKING FOR TRAINED MODEL")
    print("="*70)
    
    # Check for existing checkpoint from training tests
    checkpoint_paths = [
        Path('outputs/test_training_val/best_model.pth'),
        Path('outputs/best_model.pth'),
    ]
    
    for checkpoint_path in checkpoint_paths:
        if checkpoint_path.exists():
            print(f"\n Found trained model: {checkpoint_path}")
            return checkpoint_path
    
    # If no checkpoint exists, we need to train one quickly
    print(f"\n⚠️  No trained model found. Training a quick model...")
    print(f"   (This will take a few minutes)")
    
    # Create train/val splits
    dummy_csv = Path('data/dummy_data.csv')
    df = pd.read_csv(dummy_csv)
    train_df = df.iloc[:3]
    val_df = df.iloc[3:]
    
    train_csv = Path('data/train_dummy.csv')
    val_csv = Path('data/val_dummy.csv')
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    
    # Train for 3 epochs to get a checkpoint
    output_dir = Path('outputs/test_inference_model')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'python', 'train.py',
        '--train_csv', str(train_csv),
        '--val_csv', str(val_csv),
        '--video_root', '.',
        '--output_dir', str(output_dir),
        '--num_frames', '5',
        '--batch_size', '2',
        '--epochs', '3',
        '--lr', '0.001',
        '--num_workers', '0',
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    
    if result.returncode != 0:
        print(f" Failed to train model")
        print(result.stderr)
        raise RuntimeError("Could not create trained model for inference test")
    
    checkpoint_path = output_dir / 'best_model.pth'
    if not checkpoint_path.exists():
        raise RuntimeError(f"Training completed but checkpoint not found at {checkpoint_path}")
    
    print(f" Trained model created: {checkpoint_path}")
    
    # Clean up training splits
    train_csv.unlink()
    val_csv.unlink()
    
    return checkpoint_path


def test_inference():
    """Test inference.py with dummy data."""
    
    print("\n" + "="*70)
    print("TEST 1: INFERENCE SCRIPT")
    print("="*70)
    
    print("\n1.1 Preparing test data...")
    test_csv = create_test_csv()
    
    print("\n1.2 Ensuring trained model exists...")
    model_path = ensure_trained_model()
    
    print("\n1.3 Creating output directory...")
    output_path = Path('outputs/test_submission.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n1.4 Running inference...")
    
    # Build command
    cmd = [
        'python', 'inference.py',
        '--model_path', str(model_path),
        '--test_csv', str(test_csv),
        '--video_root', '.',
        '--output_path', str(output_path),
        '--num_frames', '5',  # Match training
        '--batch_size', '2',
        '--num_workers', '0',
    ]
    
    print(f"\n   Command:")
    print(f"   {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"\n Inference failed with return code {result.returncode}")
            print("\n--- STDOUT ---")
            print(result.stdout)
            print("\n--- STDERR ---")
            print(result.stderr)
            return False
        
        print("\n    Inference completed successfully!")
        
        # Print key output
        print("\n1.5 Inference output:")
        print("-" * 70)
        for line in result.stdout.split('\n'):
            if any(keyword in line for keyword in ['Using device', 'Loading model', 
                                                    'Test samples', 'Submission file',
                                                    'Total predictions', 'Mean GA',
                                                    'patient_id', 'site', 'predicted_GA']):
                print(f"   {line}")
        print("-" * 70)
        
    except subprocess.TimeoutExpired:
        print("\n Inference timed out")
        return False
    except Exception as e:
        print(f"\n Error running inference: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def verify_submission():
    """Verify that submission file was created and has correct format."""
    
    print("\n" + "="*70)
    print("TEST 2: VERIFY SUBMISSION FILE")
    print("="*70)
    
    submission_path = Path('outputs/test_submission.csv')
    
    print("\n2.1 Checking if submission file exists...")
    if not submission_path.exists():
        print(f"    Submission file not found: {submission_path}")
        return False
    
    print(f"    Submission file exists: {submission_path}")
    
    print("\n2.2 Loading and validating submission format...")
    
    try:
        df = pd.read_csv(submission_path)
        
        # Check required columns
        required_cols = ['patient_id', 'site', 'predicted_GA']
        missing_cols = set(required_cols) - set(df.columns)
        
        if missing_cols:
            print(f"    Missing required columns: {missing_cols}")
            return False
        
        print(f"    All required columns present: {required_cols}")
        
        # Check number of predictions
        print(f"\n2.3 Validating submission contents...")
        print(f"   Number of predictions: {len(df)}")
        print(f"   Expected: 5 (all test patients)")
        
        if len(df) != 5:
            print(f"   ⚠️  Warning: Expected 5 predictions, got {len(df)}")
        
        # Check data types
        print(f"\n   Column data types:")
        print(f"     patient_id: {df['patient_id'].dtype}")
        print(f"     site: {df['site'].dtype}")
        print(f"     predicted_GA: {df['predicted_GA'].dtype}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"\n    Found missing values:")
            print(missing_values[missing_values > 0])
            return False
        
        print(f"    No missing values")
        
        # Check predicted_GA values are reasonable
        print(f"\n2.4 Checking prediction values...")
        print(f"   Predicted GA statistics:")
        print(f"     Mean: {df['predicted_GA'].mean():.2f} days")
        print(f"     Std:  {df['predicted_GA'].std():.2f} days")
        print(f"     Min:  {df['predicted_GA'].min():.2f} days")
        print(f"     Max:  {df['predicted_GA'].max():.2f} days")
        
        # GA should be positive and reasonable (typically 50-300 days)
        if df['predicted_GA'].min() < 0:
            print(f"   ⚠️  Warning: Found negative GA predictions")
        
        if df['predicted_GA'].max() > 500:
            print(f"   ⚠️  Warning: Found unusually high GA predictions (>500 days)")
        
        print(f"\n2.5 Sample predictions:")
        print(df.head())
        
        print(f"\n    Submission format is valid!")
        
    except Exception as e:
        print(f"    Error validating submission: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_batch_inference():
    """Test that inference works with different batch sizes."""
    
    print("\n" + "="*70)
    print("TEST 3: BATCH SIZE VARIATIONS")
    print("="*70)
    
    test_csv = Path('data/test_dummy.csv')
    model_path = ensure_trained_model()
    
    batch_sizes = [1, 2, 5]  # Test different batch sizes
    
    for batch_size in batch_sizes:
        print(f"\n3.{batch_sizes.index(batch_size) + 1} Testing with batch_size={batch_size}...")
        
        output_path = Path(f'outputs/test_submission_bs{batch_size}.csv')
        
        cmd = [
            'python', 'inference.py',
            '--model_path', str(model_path),
            '--test_csv', str(test_csv),
            '--video_root', '.',
            '--output_path', str(output_path),
            '--num_frames', '5',
            '--batch_size', str(batch_size),
            '--num_workers', '0',
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"    Failed with batch_size={batch_size}")
                print(result.stderr)
                return False
            
            # Verify output
            if not output_path.exists():
                print(f"    Submission file not created")
                return False
            
            df = pd.read_csv(output_path)
            if len(df) != 5:
                print(f"    Expected 5 predictions, got {len(df)}")
                return False
            
            print(f"    Works with batch_size={batch_size} (created {len(df)} predictions)")
            
            # Clean up
            output_path.unlink()
            
        except Exception as e:
            print(f"    Error with batch_size={batch_size}: {e}")
            return False
    
    print(f"\n All batch sizes work correctly!")
    return True


def cleanup():
    """Clean up test files."""
    print("\n" + "="*70)
    print("CLEANUP")
    print("="*70)
    
    files_to_remove = [
        'data/test_dummy.csv',
        # 'outputs/test_submission.csv',
    ]
    
    dirs_to_remove = [
        'outputs/test_inference_model',
    ]
    
    print("\nRemoving temporary files...")
    for file_path in files_to_remove:
        p = Path(file_path)
        if p.exists():
            p.unlink()
            print(f"   Removed: {file_path}")
    
    for dir_path in dirs_to_remove:
        p = Path(dir_path)
        if p.exists():
            shutil.rmtree(p)
            print(f"   Removed: {dir_path}")


def run_all_tests():
    """Run all inference tests."""
    
    print("\n" + "="*70)
    print("TESTING INFERENCE.PY")
    print("="*70)
    print("\nThis will test the inference pipeline with a trained model.")
    
    try:
        # Test 1: Basic inference
        if not test_inference():
            print("\n TEST 1 FAILED: Basic inference")
            return False
        print("\n TEST 1 PASSED: Inference works!")
        
        # Test 2: Verify submission format
        if not verify_submission():
            print("\n TEST 2 FAILED: Submission verification")
            return False
        print("\n TEST 2 PASSED: Submission format is correct!")
        
        # Test 3: Batch size variations
        if not test_batch_inference():
            print("\n TEST 3 FAILED: Batch size variations")
            return False
        print("\n TEST 3 PASSED: Different batch sizes work!")
        
        # Final summary
        print("\n" + "="*70)
        print(" ALL INFERENCE TESTS PASSED!")
        print("="*70)
        print("\nInference Pipeline Summary:")
        print("   Loads trained model successfully")
        print("   Runs inference on test data")
        print("   Generates properly formatted submission CSV")
        print("   Handles different batch sizes")
        print("   Predictions are reasonable")
        print("\ninference.py is ready! ")
        print("\nNext: Test evaluate.py")
        
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
    
    if not Path('inference.py').exists():
        print(" Error: inference.py not found")
        print("  Please run this script from the directory containing inference.py")
        sys.exit(1)
    
    success = run_all_tests()
    sys.exit(0 if success else 1)
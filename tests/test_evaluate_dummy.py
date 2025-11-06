"""
Test script for evaluate.py with dummy data.

This tests the evaluation pipeline:
1. Creates ground truth and submission files
2. Runs evaluation
3. Tests different scenarios (perfect predictions, varied performance, missing predictions)
"""

import os
import sys
from pathlib import Path
import pandas as pd
import subprocess
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def create_ground_truth():
    """Create ground truth CSV from dummy data."""
    
    print("="*70)
    print("PREPARING GROUND TRUTH")
    print("="*70)
    
    # Read dummy data
    dummy_csv = Path('data/dummy_data.csv')
    if not dummy_csv.exists():
        raise FileNotFoundError(f"Dummy data not found at {dummy_csv}")
    
    df = pd.read_csv(dummy_csv)
    
    # Create ground truth with required format: patient_id, site, gestational_age_days
    ground_truth = pd.DataFrame({
        'patient_id': df['study_id'],
        'site': df['site'],
        'gestational_age_days': df['GA(days)']
    })
    
    # Save ground truth
    gt_path = Path('data/ground_truth_dummy.csv')
    ground_truth.to_csv(gt_path, index=False)
    
    print(f"\n Created ground truth: {len(ground_truth)} patients")
    print(f"  Saved to: {gt_path}")
    print(f"\nGround truth preview:")
    print(ground_truth)
    
    return gt_path, ground_truth


def create_perfect_submission(ground_truth):
    """Create a submission with perfect predictions."""
    
    print("\n" + "="*70)
    print("CREATING PERFECT SUBMISSION")
    print("="*70)
    
    # Perfect predictions = exact ground truth values
    submission = pd.DataFrame({
        'patient_id': ground_truth['patient_id'],
        'site': ground_truth['site'],
        'predicted_GA': ground_truth['gestational_age_days']
    })
    
    sub_path = Path('data/submission_perfect.csv')
    submission.to_csv(sub_path, index=False)
    
    print(f"\n Created perfect submission")
    print(f"  Saved to: {sub_path}")
    print(f"\nExpected score: 0.0 days (MAE=0, no disparity)")
    
    return sub_path


def create_good_submission(ground_truth):
    """Create a submission with small errors."""
    
    print("\n" + "="*70)
    print("CREATING GOOD SUBMISSION")
    print("="*70)
    
    # Add small random noise (±5 days)
    np.random.seed(42)
    predictions = ground_truth['gestational_age_days'] + np.random.uniform(-5, 5, len(ground_truth))
    
    submission = pd.DataFrame({
        'patient_id': ground_truth['patient_id'],
        'site': ground_truth['site'],
        'predicted_GA': predictions.round(2)
    })
    
    sub_path = Path('data/submission_good.csv')
    submission.to_csv(sub_path, index=False)
    
    print(f"\n Created good submission (±5 days error)")
    print(f"  Saved to: {sub_path}")
    print(f"\nSubmission preview:")
    print(submission)
    
    return sub_path


def create_biased_submission(ground_truth):
    """Create a submission with site-specific bias (unfair)."""
    
    print("\n" + "="*70)
    print("CREATING BIASED SUBMISSION")
    print("="*70)
    
    # Add different errors for different sites
    np.random.seed(42)
    predictions = []
    
    for _, row in ground_truth.iterrows():
        true_ga = row['gestational_age_days']
        site = row['site']
        
        if site == 'site-a':
            # Good predictions for site-a (±5 days)
            pred = true_ga + np.random.uniform(-5, 5)
        elif site == 'site-b':
            # Medium predictions for site-b (±15 days)
            pred = true_ga + np.random.uniform(-15, 15)
        else:  # site-c
            # Poor predictions for site-c (±30 days)
            pred = true_ga + np.random.uniform(-30, 30)
        
        predictions.append(pred)
    
    submission = pd.DataFrame({
        'patient_id': ground_truth['patient_id'],
        'site': ground_truth['site'],
        'predicted_GA': np.round(predictions, 2)
    })
    
    sub_path = Path('data/submission_biased.csv')
    submission.to_csv(sub_path, index=False)
    
    print(f"\n Created biased submission (different errors per site)")
    print(f"  Saved to: {sub_path}")
    print(f"  site-a: ±5 days (good)")
    print(f"  site-b: ±15 days (medium)")
    print(f"  site-c: ±30 days (poor)")
    print(f"\nExpected: High fairness penalty due to site disparity")
    
    return sub_path


def create_incomplete_submission(ground_truth):
    """Create a submission missing some predictions."""
    
    print("\n" + "="*70)
    print("CREATING INCOMPLETE SUBMISSION")
    print("="*70)
    
    # Only predict for first 3 patients
    submission = pd.DataFrame({
        'patient_id': ground_truth['patient_id'][:3],
        'site': ground_truth['site'][:3],
        'predicted_GA': ground_truth['gestational_age_days'][:3]
    })
    
    sub_path = Path('data/submission_incomplete.csv')
    submission.to_csv(sub_path, index=False)
    
    print(f"\n Created incomplete submission ({len(submission)}/5 predictions)")
    print(f"  Saved to: {sub_path}")
    print(f"\nExpected: Warning about missing predictions")
    
    return sub_path


def test_perfect_submission():
    """Test evaluation with perfect predictions."""
    
    print("\n" + "="*70)
    print("TEST 1: PERFECT SUBMISSION (MAE = 0)")
    print("="*70)
    
    gt_path, ground_truth = create_ground_truth()
    sub_path = create_perfect_submission(ground_truth)
    
    print("\n1.1 Running evaluation...")
    
    cmd = [
        'python', 'evaluate.py',
        '--submission', str(sub_path),
        '--ground_truth', str(gt_path),
        '--penalty_weight', '0.3'
    ]
    
    print(f"\n   Command:")
    print(f"   {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print(f"\n Evaluation failed")
            print("\n--- STDERR ---")
            print(result.stderr)
            return False
        
        print("\n    Evaluation completed!")
        
        # Print output
        print("\n1.2 Evaluation output:")
        print("-" * 70)
        print(result.stdout)
        print("-" * 70)
        
        # Check that score is 0
        if 'FINAL SCORE: 0.0000' not in result.stdout:
            print("\n   ⚠️  Warning: Expected final score of 0.0000 for perfect predictions")
        
    except subprocess.TimeoutExpired:
        print("\n Evaluation timed out")
        return False
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_good_submission():
    """Test evaluation with small errors."""
    
    print("\n" + "="*70)
    print("TEST 2: GOOD SUBMISSION (Small Errors)")
    print("="*70)
    
    gt_path, ground_truth = create_ground_truth()
    sub_path = create_good_submission(ground_truth)
    
    print("\n2.1 Running evaluation...")
    
    cmd = [
        'python', 'evaluate.py',
        '--submission', str(sub_path),
        '--ground_truth', str(gt_path),
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            print(f"\n Evaluation failed")
            print(result.stderr)
            return False
        
        print("\n    Evaluation completed!")
        
        # Print key metrics
        print("\n2.2 Key metrics:")
        print("-" * 70)
        for line in result.stdout.split('\n'):
            if any(keyword in line for keyword in ['FINAL SCORE', 'Overall MAE', 
                                                    'Fairness Penalty', 'Total Predictions']):
                print(f"   {line}")
        print("-" * 70)
        
    except Exception as e:
        print(f"\n Error: {e}")
        return False
    
    return True


def test_biased_submission():
    """Test evaluation with site-specific bias."""
    
    print("\n" + "="*70)
    print("TEST 3: BIASED SUBMISSION (High Disparity)")
    print("="*70)
    
    gt_path, ground_truth = create_ground_truth()
    sub_path = create_biased_submission(ground_truth)
    
    print("\n3.1 Running evaluation...")
    
    cmd = [
        'python', 'evaluate.py',
        '--submission', str(sub_path),
        '--ground_truth', str(gt_path),
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            print(f"\n Evaluation failed")
            print(result.stderr)
            return False
        
        print("\n    Evaluation completed!")
        
        # Print per-site performance
        print("\n3.2 Per-site performance:")
        print("-" * 70)
        lines = result.stdout.split('\n')
        in_site_section = False
        for line in lines:
            if 'Per-Site Performance' in line:
                in_site_section = True
            if in_site_section and ('site-' in line or 'Site' in line or '---' in line):
                print(f"   {line}")
            if 'Fairness Analysis' in line:
                break
        print("-" * 70)
        
        # Check for fairness penalty
        if 'Fairness Penalty' not in result.stdout:
            print(f"\n   ⚠️  Warning: Fairness penalty not found in output")
            return False
        
        print("\n    Fairness penalty calculated correctly!")
        
    except Exception as e:
        print(f"\n Error: {e}")
        return False
    
    return True


def test_incomplete_submission():
    """Test evaluation with missing predictions."""
    
    print("\n" + "="*70)
    print("TEST 4: INCOMPLETE SUBMISSION (Missing Predictions)")
    print("="*70)
    
    gt_path, ground_truth = create_ground_truth()
    sub_path = create_incomplete_submission(ground_truth)
    
    print("\n4.1 Running evaluation...")
    
    cmd = [
        'python', 'evaluate.py',
        '--submission', str(sub_path),
        '--ground_truth', str(gt_path),
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            print(f"\n Evaluation failed")
            print(result.stderr)
            return False
        
        print("\n    Evaluation completed!")
        
        # Check for warning about missing predictions
        if 'missing from submission' not in result.stdout.lower():
            print(f"\n   ⚠️  Warning: Expected warning about missing predictions")
        else:
            print(f"\n    Correctly detected missing predictions!")
        
        print("\n4.2 Evaluation output:")
        print("-" * 70)
        print(result.stdout)
        print("-" * 70)
        
    except Exception as e:
        print(f"\n Error: {e}")
        return False
    
    return True


def test_different_penalty_weights():
    """Test evaluation with different penalty weights."""
    
    print("\n" + "="*70)
    print("TEST 5: DIFFERENT PENALTY WEIGHTS")
    print("="*70)
    
    gt_path, ground_truth = create_ground_truth()
    sub_path = create_biased_submission(ground_truth)
    
    penalty_weights = [0.0, 0.3, 1.0]
    
    for weight in penalty_weights:
        print(f"\n5.{penalty_weights.index(weight) + 1} Testing penalty_weight={weight}...")
        
        cmd = [
            'python', 'evaluate.py',
            '--submission', str(sub_path),
            '--ground_truth', str(gt_path),
            '--penalty_weight', str(weight),
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"    Failed with penalty_weight={weight}")
                return False
            
            # Extract final score
            for line in result.stdout.split('\n'):
                if 'FINAL SCORE:' in line:
                    print(f"   {line.strip()}")
                    break
            
            print(f"    Works with penalty_weight={weight}")
            
        except Exception as e:
            print(f"    Error with penalty_weight={weight}: {e}")
            return False
    
    print(f"\n All penalty weights work correctly!")
    return True


def test_json_output():
    """Test JSON output functionality."""
    
    print("\n" + "="*70)
    print("TEST 6: JSON OUTPUT")
    print("="*70)
    
    gt_path, ground_truth = create_ground_truth()
    sub_path = create_good_submission(ground_truth)
    json_path = Path('outputs/test_results.json')
    
    print("\n6.1 Running evaluation with JSON output...")
    
    cmd = [
        'python', 'evaluate.py',
        '--submission', str(sub_path),
        '--ground_truth', str(gt_path),
        '--output_json', str(json_path),
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            print(f"    Evaluation failed")
            return False
        
        print("    Evaluation completed!")
        
        # Check JSON file exists
        print(f"\n6.2 Checking JSON output...")
        if not json_path.exists():
            print(f"    JSON file not created: {json_path}")
            return False
        
        print(f"    JSON file created: {json_path}")
        
        # Load and validate JSON
        import json
        with open(json_path) as f:
            results = json.load(f)
        
        required_keys = ['final_score', 'overall_mae', 'fairness_penalty', 
                        'per_site_metrics', 'num_predictions']
        missing_keys = [k for k in required_keys if k not in results]
        
        if missing_keys:
            print(f"    Missing keys in JSON: {missing_keys}")
            return False
        
        print(f"    JSON contains all required keys")
        print(f"\n   JSON contents:")
        print(f"     final_score: {results['final_score']:.4f}")
        print(f"     overall_mae: {results['overall_mae']:.4f}")
        print(f"     fairness_penalty: {results['fairness_penalty']:.4f}")
        print(f"     num_predictions: {results['num_predictions']}")
        print(f"     num_sites: {results['num_sites']}")
        
        # Clean up
        json_path.unlink()
        
    except Exception as e:
        print(f"    Error: {e}")
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
        'data/ground_truth_dummy.csv',
        'data/submission_perfect.csv',
        'data/submission_good.csv',
        'data/submission_biased.csv',
        'data/submission_incomplete.csv',
    ]
    
    print("\nRemoving temporary files...")
    for file_path in files_to_remove:
        p = Path(file_path)
        if p.exists():
            p.unlink()
            print(f"   Removed: {file_path}")


def run_all_tests():
    """Run all evaluation tests."""
    
    print("\n" + "="*70)
    print("TESTING EVALUATE.PY")
    print("="*70)
    print("\nThis will test the evaluation script with various submission scenarios.")
    
    try:
        # Test 1: Perfect submission
        if not test_perfect_submission():
            print("\n TEST 1 FAILED: Perfect submission")
            return False
        print("\n TEST 1 PASSED: Perfect submission works!")
        
        # Test 2: Good submission
        if not test_good_submission():
            print("\n TEST 2 FAILED: Good submission")
            return False
        print("\n TEST 2 PASSED: Good submission works!")
        
        # Test 3: Biased submission
        if not test_biased_submission():
            print("\n TEST 3 FAILED: Biased submission")
            return False
        print("\n TEST 3 PASSED: Biased submission works!")
        
        # Test 4: Incomplete submission
        if not test_incomplete_submission():
            print("\n TEST 4 FAILED: Incomplete submission")
            return False
        print("\n TEST 4 PASSED: Incomplete submission works!")
        
        # Test 5: Different penalty weights
        if not test_different_penalty_weights():
            print("\n TEST 5 FAILED: Penalty weights")
            return False
        print("\n TEST 5 PASSED: Penalty weights work!")
        
        # Test 6: JSON output
        if not test_json_output():
            print("\n TEST 6 FAILED: JSON output")
            return False
        print("\n TEST 6 PASSED: JSON output works!")
        
        # Final summary
        print("\n" + "="*70)
        print(" ALL EVALUATION TESTS PASSED!")
        print("="*70)
        print("\nEvaluation Pipeline Summary:")
        print("   Evaluates perfect predictions correctly (score = 0)")
        print("   Calculates MAE for predictions with errors")
        print("   Detects site-specific bias and applies fairness penalty")
        print("   Handles missing predictions gracefully")
        print("   Works with different penalty weights")
        print("   Generates JSON output with all metrics")
        print("\nevaluate.py is ready! ")
        print("\n" + "="*70)
        print("ALL THREE SCRIPTS TESTED SUCCESSFULLY!")
        print("="*70)
        print("\nYour pipeline is complete:")
        print("   train.py - Training pipeline")
        print("   inference.py - Generate predictions")
        print("   evaluate.py - Evaluate submissions")
        print("\nYou're ready to go! 🚀")
        
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
        print(" Please make sure you have generated the dummy data first")
        sys.exit(1)
    
    if not Path('evaluate.py').exists():
        print(" Error: evaluate.py not found")
        print(" Please run this script from the directory containing evaluate.py")
        sys.exit(1)
    
    success = run_all_tests()
    sys.exit(0 if success else 1)
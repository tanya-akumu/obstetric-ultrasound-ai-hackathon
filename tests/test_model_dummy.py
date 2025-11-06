"""
Test script for model.py with dummy .nii.gz data.

This tests the model with actual data from the dataset.
"""

import os, sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import model and dataset
from model import GAPredictor, GAPredictor_Attention, get_model
from dataset import BlindsweepDataset, get_transforms



def test_model_basic():
    """Test basic model functionality."""
    
    print("="*70)
    print("TEST 1: BASIC MODEL FUNCTIONALITY")
    print("="*70)
    
    print("\n1.1 Creating baseline model...")
    model = GAPredictor(pretrained=False)
    print("    Model created")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    print("\n1.2 Testing forward pass with dummy input...")
    # Create dummy input: (batch_size, num_frames, C, H, W)
    batch_size = 2
    num_frames = 40  # 8 sweeps × 5 frames
    x = torch.randn(batch_size, num_frames, 3, 224, 224)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Expected output shape: ({batch_size}, 1)")
    assert output.shape == (batch_size, 1), f"Shape mismatch!"
    print("    Forward pass successful")
    
    print("\n1.3 Testing gradient flow...")
    model.train()
    x = torch.randn(2, 40, 3, 224, 224, requires_grad=True)
    target = torch.randn(2, 1)
    
    output = model(x)
    loss = nn.MSELoss()(output, target)
    loss.backward()
    
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                   for p in model.parameters())
    assert has_grad, "No gradients computed!"
    print("    Gradients computed successfully")
    
    print("\n1.4 Testing with different frame counts...")
    model.eval()
    for num_frames in [8, 16, 32, 64]:
        x = torch.randn(1, num_frames, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        print(f"   num_frames={num_frames}: output shape {output.shape} ")
        assert output.shape == (1, 1)
    
    print("\n TEST 1 PASSED: Basic model functionality works!")


def test_model_with_dataset():
    """Test model with actual dataset."""
    
    print("\n" + "="*70)
    print("TEST 2: MODEL WITH REAL DATASET")
    print("="*70)
    
    print("\n2.1 Loading dataset...")
    dataset = BlindsweepDataset(
        data_csv='data/dummy_data.csv',
        video_root='.',
        num_frames_per_sweep=5,
        transform=get_transforms(mode='train'),
        mode='train'
    )
    print(f"    Dataset loaded: {len(dataset)} patients")
    
    print("\n2.2 Loading one sample...")
    frames, label, metadata = dataset[0]
    print(f"   Patient: {metadata['patient_id']}")
    print(f"   Frames shape: {frames.shape}")
    print(f"   Label: {label.item():.1f} days")
    print(f"    Sample loaded")
    
    print("\n2.3 Testing model with real data...")
    model = GAPredictor(pretrained=False)
    model.eval()
    
    # Add batch dimension
    frames_batch = frames.unsqueeze(0)  # (1, num_frames, C, H, W)
    
    with torch.no_grad():
        prediction = model(frames_batch)
    
    print(f"   Input shape: {frames_batch.shape}")
    print(f"   Prediction: {prediction.item():.2f} days")
    print(f"   True label: {label.item():.1f} days")
    print(f"    Model inference successful")
    
    print("\n2.4 Testing batch processing...")
    # Load both patients
    frames1, label1, meta1 = dataset[0]
    frames2, label2, meta2 = dataset[1]
    
    # Stack into batch
    batch_frames = torch.stack([frames1, frames2])
    batch_labels = torch.stack([label1, label2]).unsqueeze(1)
    
    with torch.no_grad():
        predictions = model(batch_frames)
    
    print(f"   Batch shape: {batch_frames.shape}")
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Patient 1 - Pred: {predictions[0].item():.2f}, True: {label1.item():.1f}")
    print(f"   Patient 2 - Pred: {predictions[1].item():.2f}, True: {label2.item():.1f}")
    print(f"    Batch processing successful")
    
    print("\n TEST 2 PASSED: Model works with real dataset!")


def test_training_step():
    """Test a single training step."""
    
    print("\n" + "="*70)
    print("TEST 3: TRAINING STEP")
    print("="*70)
    
    print("\n3.1 Setting up training...")
    dataset = BlindsweepDataset(
        data_csv='data/dummy_data.csv',
        video_root='.',
        num_frames_per_sweep=5,
        transform=get_transforms(mode='train'),
        mode='train'
    )
    
    model = GAPredictor(pretrained=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.L1Loss()  # MAE loss
    
    print("    Model, optimizer, and loss function ready")
    
    print("\n3.2 Performing one training step...")
    model.train()
    
    # Get batch
    frames1, label1, _ = dataset[0]
    frames2, label2, _ = dataset[1]
    batch_frames = torch.stack([frames1, frames2])
    batch_labels = torch.stack([label1, label2]).unsqueeze(1)
    
    # Forward pass
    optimizer.zero_grad()
    predictions = model(batch_frames)
    loss = criterion(predictions, batch_labels)
    
    print(f"   Predictions: {predictions.squeeze().detach().numpy()}")
    print(f"   Labels: {batch_labels.squeeze().numpy()}")
    print(f"   Loss (MAE): {loss.item():.4f} days")
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    print("    Training step completed")
    
    print("\n3.3 Checking if model parameters updated...")
    # Do another forward pass to see if output changed
    with torch.no_grad():
        new_predictions = model(batch_frames)
    
    prediction_changed = not torch.allclose(predictions, new_predictions, atol=1e-7)
    if prediction_changed:
        print("    Model parameters updated (predictions changed)")
    else:
        print("   ⚠ Model parameters might not have updated significantly")
    
    print("\n TEST 3 PASSED: Training step works!")


def test_attention_model():
    """Test attention model variant."""
    
    print("\n" + "="*70)
    print("TEST 4: ATTENTION MODEL")
    print("="*70)
    
    print("\n4.1 Creating attention model...")
    model_attention = GAPredictor_Attention(pretrained=False)
    print("    Attention model created")
    
    total_params = sum(p.numel() for p in model_attention.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    print("\n4.2 Testing forward pass...")
    x = torch.randn(2, 40, 3, 224, 224)
    
    model_attention.eval()
    with torch.no_grad():
        output = model_attention(x)
    
    print(f"   Output shape: {output.shape}")
    assert output.shape == (2, 1)
    print("    Attention model forward pass successful")
    
    print("\n4.3 Comparing with baseline model...")
    model_baseline = GAPredictor(pretrained=False)
    model_baseline.eval()
    
    with torch.no_grad():
        output_baseline = model_baseline(x)
        output_attention = model_attention(x)
    
    print(f"   Baseline output: {output_baseline.squeeze().numpy()}")
    print(f"   Attention output: {output_attention.squeeze().numpy()}")
    
    # Outputs should be different (different random weights)
    different = not torch.allclose(output_baseline, output_attention, atol=0.1)
    if different:
        print("    Models produce different outputs (as expected)")
    
    print("\n TEST 4 PASSED: Attention model works!")


def test_model_factory():
    """Test get_model factory function."""
    
    print("\n" + "="*70)
    print("TEST 5: MODEL FACTORY")
    print("="*70)
    
    print("\n5.1 Testing baseline model creation...")
    model = get_model(model_type='baseline', pretrained=False)
    assert isinstance(model, GAPredictor)
    print("    get_model('baseline') works")
    
    print("\n5.2 Testing attention model creation...")
    model = get_model(model_type='attention', pretrained=False)
    assert isinstance(model, GAPredictor_Attention)
    print("    get_model('attention') works")
    
    print("\n5.3 Testing invalid model type...")
    try:
        model = get_model(model_type='invalid', pretrained=False)
        print("    Should have raised ValueError")
        assert False
    except ValueError:
        print("    Correctly raises ValueError for invalid type")
    
    print("\n TEST 5 PASSED: Model factory works!")


def test_save_load():
    """Test model save and load."""
    
    print("\n" + "="*70)
    print("TEST 6: MODEL SAVE/LOAD")
    print("="*70)
    
    print("\n6.1 Creating and saving model...")
    model = GAPredictor(pretrained=False)
    
    # Save model
    save_path = 'outputs/test_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': 'baseline'
    }, save_path)
    print(f"    Model saved to {save_path}")
    
    print("\n6.2 Loading model...")
    new_model = GAPredictor(pretrained=False)
    checkpoint = torch.load(save_path)
    new_model.load_state_dict(checkpoint['model_state_dict'])
    print("    Model loaded successfully")
    
    print("\n6.3 Verifying models produce same output...")
    x = torch.randn(1, 40, 3, 224, 224)
    
    model.eval()
    new_model.eval()
    
    with torch.no_grad():
        output1 = model(x)
        output2 = new_model(x)
    
    same = torch.allclose(output1, output2, atol=1e-6)
    assert same, "Models should produce identical output!"
    print(f"   Original output: {output1.item():.6f}")
    print(f"   Loaded output:   {output2.item():.6f}")
    print("    Outputs match perfectly")
    
    # Clean up
    Path(save_path).unlink()
    print(f"    Cleaned up test file")
    
    print("\n TEST 6 PASSED: Save/load works!")


def run_all_tests():
    """Run all model tests."""
    
    print("\n" + "="*70)
    print("RUNNING ALL MODEL TESTS")
    print("="*70)
    
    try:
        test_model_basic()
        test_model_with_dataset()
        test_training_step()
        test_attention_model()
        test_model_factory()
        test_save_load()
        
        print("\n" + "="*70)
        print(" ALL MODEL TESTS PASSED!")
        print("="*70)
        print("\nModel Summary:")
        print("   Basic forward pass works")
        print("   Gradients computed correctly")
        print("   Works with real dataset")
        print("   Batch processing works")
        print("   Training step successful")
        print("   Attention variant works")
        print("   Model factory works")
        print("   Save/load functionality works")
        print("\nModel is ready for training!")
        
        return True
        
    except Exception as e:
        print(f"\n TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
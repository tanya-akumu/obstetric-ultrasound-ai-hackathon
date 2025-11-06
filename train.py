"""
Training script for Gestational Age prediction from blindsweep videos.

This script trains a MobileNet-based regression model that:
- Samples frames from multiple video sweeps per patient
- Aggregates predictions across sweeps to predict a single GA value
- Validates every 10 epochs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import argparse

# Import dataset and model from separate modules
from dataset import BlindsweepDataset, get_transforms
from model import GAPredictor, get_model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_mae = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for frames, labels, metadata in pbar:
        frames = frames.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        predictions = model(frames)
        
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_mae += torch.abs(predictions - labels).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'mae': torch.abs(predictions - labels).mean().item()})
    
    avg_loss = total_loss / len(dataloader)
    avg_mae = total_mae / len(dataloader.dataset)
    
    return avg_loss, avg_mae


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_mae = 0
    
    with torch.no_grad():
        for frames, labels, metadata in tqdm(dataloader, desc='Validating'):
            frames = frames.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            predictions = model(frames)
            
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            total_mae += torch.abs(predictions - labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    avg_mae = total_mae / len(dataloader.dataset)
    
    return avg_loss, avg_mae


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    train_transform = get_transforms(mode='train')
    val_transform = get_transforms(mode='val')
    
    # Create datasets
    train_dataset = BlindsweepDataset(
        data_csv=args.train_csv,
        video_root=args.video_root,
        num_frames_per_sweep=args.num_frames,
        transform=train_transform,
        mode='train'
    )
    
    val_dataset = BlindsweepDataset(
        data_csv=args.val_csv,
        video_root=args.video_root,
        num_frames_per_sweep=args.num_frames,
        transform=val_transform,
        mode='val'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model using factory function
    # Can easily switch to 'attention' model or other variants
    model = get_model(
        model_type='baseline',  # or 'attention' for attention-based model
        pretrained=args.pretrained
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.L1Loss()  # MAE loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_val_mae = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_loss, train_mae = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f} days")
        
        # Validate every 10 epochs
        if epoch % 3 == 0: #DEBUG CHANGE FROM 3 TO 10
            val_loss, val_mae = validate(model, val_loader, criterion, device)
            print(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f} days")
            
            scheduler.step(val_mae)
            
            # Save best model
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_mae': val_mae,
                }, args.output_dir / 'best_model.pth')
                print(f"Saved best model (MAE: {val_mae:.4f} days)")
        
        # Save checkpoint every 20 epochs
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, args.output_dir / f'checkpoint_epoch_{epoch}.pth')
    
    print(f"\nTraining complete! Best validation MAE: {best_val_mae:.4f} days")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GA prediction model')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--val_csv', type=str, required=True, help='Path to validation CSV')
    parser.add_argument('--video_root', type=str, required=True, help='Root directory for videos')
    parser.add_argument('--output_dir', type=Path, default=Path('outputs'), help='Output directory')
    parser.add_argument('--num_frames', type=int, default=16, help='Frames to sample per sweep')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained MobileNet')
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    main(args)
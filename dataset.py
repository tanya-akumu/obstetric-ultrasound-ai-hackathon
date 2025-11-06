"""
Dataset class for loading blindsweep videos and sampling frames.

This module handles:
- Loading video files from multiple sweeps per patient
- Sampling frames uniformly from each video
- Applying data augmentation transforms
- Grouping sweeps by patient for patient-level predictions
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
import gzip
import struct


class BlindsweepDataset(Dataset):
    """
    Dataset for loading blindsweep videos and sampling frames.
    
    Each patient may have multiple sweep videos. We sample frames from each sweep
    and the model learns to predict GA from these sampled frames.
    
    Key features:
    - Groups videos by patient_id
    - Samples fixed number of frames per sweep
    - Returns all frames from all sweeps for a patient
    - Handles video loading errors gracefully
    """
    
    def __init__(self, data_csv, video_root, num_frames_per_sweep=16, transform=None, mode='train'):
        """
        Args:
            data_csv (str): Path to CSV with format:
                           study_id, site, GA(days), path1, path2, ..., path8
                           For test mode, GA(days) can be dummy values (e.g., 0)
            video_root (str): Root directory containing video files
            num_frames_per_sweep (int): Number of frames to sample from each sweep
            transform (callable): Torchvision transforms to apply to frames
            mode (str): 'train', 'val', or 'test'
        """
        self.data = pd.read_csv(data_csv)
        self.video_root = Path(video_root)
        self.num_frames = num_frames_per_sweep
        self.transform = transform
        self.mode = mode
        
        # Expected columns - changed from sweep1-8 to path1-8
        self.path_columns = [f'path{i}' for i in range(1, 9)]
        
        # Validate CSV format
        required_cols = ['study_id', 'site']
        if mode in ['train', 'val']:
            required_cols.append('GA(days)')
        
        missing_cols = set(required_cols) - set(self.data.columns)
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {missing_cols}")
        
        # Get patient IDs
        self.patient_ids = self.data['study_id'].tolist()
        
        # Count total non-empty sweeps
        total_sweeps = 0
        for _, row in self.data.iterrows():
            paths = [row[col] for col in self.path_columns if col in self.data.columns]
            non_empty_paths = sum(1 for s in paths if pd.notna(s) and str(s).strip() != '')
            total_sweeps += non_empty_paths
        
        print(f"Loaded {len(self.patient_ids)} patients with "
              f"{total_sweeps} total sweeps for {mode} mode")
    
    def __len__(self):
        """Returns number of patients (not number of sweeps)."""
        return len(self.patient_ids)
    
    def _sample_frames_from_nifti(self, nifti_path):
        """
        Load .nii.gz file and randomly sample frames.
        
        Args:
            nifti_path (Path): Path to .nii.gz file
            
        Returns:
            frames (np.ndarray): Array of shape (num_frames, H, W, 3)
                                RGB frames sampled from the volume
        
        Note:
            - Randomly samples frames from the 3D volume
            - Converts grayscale to RGB by repeating channel 3 times
            - Returns uint8 array [0, 255]
        """
        try:
            # Read NIfTI file
            with gzip.open(nifti_path, 'rb') as f:
                # Read header (348 bytes for NIfTI-1)
                header = f.read(348)
                
                # Parse dimensions from header (dims at bytes 40-56)
                # Format: [ndim, dim1, dim2, dim3, ...]
                dims = struct.unpack('<8h', header[40:56])
                ndim = dims[0]
                width = dims[1]   # X dimension
                height = dims[2]  # Y dimension  
                slices = dims[3]  # Z dimension (number of frames/slices)
                
                # Read vox_offset to know where data starts
                vox_offset = struct.unpack('<f', header[108:112])[0]
                
                # Skip to data (vox_offset - 348 bytes already read)
                extra_bytes = int(vox_offset) - 348
                if extra_bytes > 0:
                    f.read(extra_bytes)
                
                # Read all data
                data = f.read()
            
            # Convert to numpy array (uint8)
            total_voxels = width * height * slices
            volume = np.frombuffer(data[:total_voxels], dtype=np.uint8)
            
            # Reshape to (slices, height, width)
            volume = volume.reshape(slices, height, width)
            
            # Randomly sample frames
            num_available_frames = slices
            if num_available_frames <= self.num_frames:
                # If not enough frames, sample with replacement
                frame_indices = np.random.choice(num_available_frames, 
                                                self.num_frames, 
                                                replace=True)
            else:
                # Sample without replacement
                frame_indices = np.random.choice(num_available_frames,
                                               self.num_frames,
                                               replace=False)
            
            # Extract sampled frames
            sampled_frames = volume[frame_indices]  # (num_frames, H, W)
            
            # Convert grayscale to RGB by repeating channel 3 times
            frames_rgb = np.stack([sampled_frames] * 3, axis=-1)  # (num_frames, H, W, 3)
            
            return frames_rgb
            
        except Exception as e:
            raise ValueError(f"Error loading NIfTI file {nifti_path}: {e}")
    
    def __getitem__(self, idx):
        """
        Returns all frames from all sweeps for a single patient.
        
        Args:
            idx (int): Patient index
            
        Returns:
            frames_tensor (torch.Tensor): Shape (num_sweeps * num_frames, C, H, W)
                                         All frames from all sweeps concatenated
            label (torch.Tensor): Gestational age in days (scalar)
            metadata (dict): Contains patient_id, site, and num_sweeps
            
        Example:
            If patient has 3 sweeps and num_frames=16:
            - frames_tensor shape: (48, 3, 224, 224)
            - label: tensor(245.0)
            - metadata: {'patient_id': 'P001', 'site': 'site_A', 'num_sweeps': 3}
        """
        # Get patient data from row
        patient_row = self.data.iloc[idx]
        patient_id = patient_row['study_id']
        site = patient_row['site']
        
        # Get gestational age (or 0 for test mode)
        if 'GA(days)' in patient_row:
            ga_days = patient_row['GA(days)']
        else:
            ga_days = 0  # Dummy value for test mode
        
        all_frames = []
        
        # Process each sweep for this patient
        sweep_count = 0
        for path_col in self.path_columns:
            if path_col not in self.data.columns:
                continue
                
            file_path = patient_row[path_col]
            
            # Skip empty paths
            if pd.isna(file_path) or str(file_path).strip() == '':
                continue
            
            full_path = self.video_root / file_path
            
            try:
                frames = self._sample_frames_from_nifti(full_path)
                
                # Apply transforms to each frame
                if self.transform:
                    transformed_frames = []
                    for frame in frames:
                        transformed_frames.append(self.transform(frame))
                    frames_tensor = torch.stack(transformed_frames)
                else:
                    # Convert to tensor and normalize to [0, 1]
                    frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
                
                all_frames.append(frames_tensor)
                sweep_count += 1
                
            except Exception as e:
                print(f"Warning: Error loading NIfTI {full_path}: {e}")
                # Create dummy frames if loading fails
                if self.transform:
                    dummy_frame = self.transform(np.zeros((224, 224, 3), dtype=np.uint8))
                else:
                    dummy_frame = torch.zeros(3, 224, 224)
                all_frames.append(dummy_frame.unsqueeze(0).repeat(self.num_frames, 1, 1, 1))
                sweep_count += 1
        
        # Concatenate all frames from all sweeps
        # If patient has 3 sweeps with 16 frames each: (3, 16, C, H, W) -> (48, C, H, W)
        if len(all_frames) > 0:
            frames_tensor = torch.cat(all_frames, dim=0)
        else:
            # No valid sweeps found, return dummy
            if self.transform:
                dummy_frame = self.transform(np.zeros((224, 224, 3), dtype=np.uint8))
            else:
                dummy_frame = torch.zeros(3, 224, 224)
            frames_tensor = dummy_frame.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)
            sweep_count = 1
        
        # Label
        label = torch.tensor(ga_days, dtype=torch.float32)
        
        # Metadata
        metadata = {
            'patient_id': patient_id,
            'site': site,
            'num_sweeps': sweep_count
        }
        
        return frames_tensor, label, metadata


def get_transforms(mode='train'):
    """
    Get appropriate transforms for train/val/test modes.
    
    Args:
        mode (str): 'train' for augmentation, 'val' or 'test' for no augmentation
        
    Returns:
        transform (torchvision.transforms.Compose): Composed transforms
    """
    from torchvision import transforms
    
    if mode == 'train':
        # Training: apply data augmentation
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Val/Test: no augmentation
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform


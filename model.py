"""
Model architecture for Gestational Age prediction from blindsweep videos.

This module defines the GAPredictor model which:
- Uses MobileNetV2 as a feature encoder
- Processes multiple frames from multiple video sweeps
- Aggregates features through temporal pooling
- Predicts a single gestational age value
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2


class GAPredictor(nn.Module):
    """
    Gestational Age predictor using MobileNetV2 encoder.
    
    Architecture:
    1. MobileNetV2 backbone extracts features from each frame independently
    2. Features are averaged across all frames (from all sweeps)
    3. Fully connected layers regress to a single GA value
    
    The model is designed to handle variable numbers of frames per patient
    (since different patients may have different numbers of sweeps).
    """
    
    def __init__(self, pretrained=True, feature_dim=1280, hidden_dim=256, dropout=0.3):
        """
        Args:
            pretrained (bool): Whether to use pretrained MobileNetV2 weights
            feature_dim (int): Output dimension of MobileNetV2 encoder (default: 1280)
            hidden_dim (int): Hidden layer dimension in regression head
            dropout (float): Dropout rate in regression head
        """
        super(GAPredictor, self).__init__()
        
        # Load MobileNetV2 as encoder
        from torchvision.models import mobilenet_v2
        if pretrained:
            mobilenet = mobilenet_v2( weights='IMAGENET1K_V1')
        else:
            mobilenet = mobilenet_v2( weights=None)
        
        # Remove the classifier, keep only feature extraction
        # MobileNetV2 has: features -> classifier
        # We want just the features part
        self.encoder = mobilenet.features
        
        # Add adaptive pooling to handle variable spatial dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # MobileNetV2 outputs 1280-dimensional features
        self.feature_dim = feature_dim
        
        # Regression head: Predict GA from aggregated features
        self.regressor = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize regression head weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights of the regression head."""
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_frames, C, H, W)
                             Where num_frames = sum of frames from all sweeps for each patient
        
        Returns:
            predictions (torch.Tensor): Predicted GA in days, shape (batch_size, 1)
        
        Processing steps:
        1. Reshape: (B, N, C, H, W) -> (B*N, C, H, W)
        2. Encode: Extract features from each frame
        3. Pool: Remove spatial dimensions -> (B*N, F)
        4. Reshape: (B*N, F) -> (B, N, F)
        5. Pool: Average across frames -> (B, F)
        6. Regress: Predict GA -> (B, 1)
        """
        batch_size, num_frames, C, H, W = x.shape
        
        # Reshape to process all frames together through the encoder
        x = x.view(batch_size * num_frames, C, H, W)
        
        # Extract features from all frames
        # Output shape: (batch_size * num_frames, 1280, H', W')
        features = self.encoder(x)
        
        # Apply adaptive pooling to remove spatial dimensions
        # Output shape: (batch_size * num_frames, 1280, 1, 1)
        features = self.adaptive_pool(features)
        
        # Remove spatial dimensions
        features = features.squeeze(-1).squeeze(-1)  # (batch_size * num_frames, 1280)
        
        # Reshape back to separate batch and frames
        features = features.view(batch_size, num_frames, self.feature_dim)
        
        # Temporal pooling: average across all frames
        # This aggregates information from all sweeps of a patient
        pooled_features = features.mean(dim=1)  # (batch_size, 1280)
        
        # Predict gestational age
        predictions = self.regressor(pooled_features)  # (batch_size, 1)
        
        return predictions
    
    def get_features(self, x):
        """
        Extract features without making predictions.
        Useful for visualization or feature analysis.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_frames, C, H, W)
        
        Returns:
            features (torch.Tensor): Pooled features of shape (batch_size, feature_dim)
        """
        batch_size, num_frames, C, H, W = x.shape
        
        x = x.view(batch_size * num_frames, C, H, W)
        features = self.encoder(x)
        features = self.adaptive_pool(features)
        features = features.squeeze(-1).squeeze(-1)
        features = features.view(batch_size, num_frames, self.feature_dim)
        pooled_features = features.mean(dim=1)
        
        return pooled_features


class GAPredictor_Attention(nn.Module):
    """
    Enhanced GA predictor with attention mechanism.
    
    Instead of simple averaging, this uses attention to weight frames differently.
    Some frames may be more informative than others for predicting GA.
    
    This is provided as an example of how teams can extend the baseline.
    """
    
    def __init__(self, pretrained=True, feature_dim=1280, hidden_dim=256, dropout=0.3):
        """
        Args:
            pretrained (bool): Whether to use pretrained MobileNetV2 weights
            feature_dim (int): Output dimension of MobileNetV2 encoder
            hidden_dim (int): Hidden layer dimension in regression head
            dropout (float): Dropout rate
        """
        super(GAPredictor_Attention, self).__init__()
        
        # Encoder (same as base model)
        from torchvision.models import mobilenet_v2
        if pretrained:
            mobilenet = mobilenet_v2( weights='IMAGENET1K_V1')
        else:
            mobilenet = mobilenet_v2( weights=None)
        
        self.encoder = mobilenet.features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = feature_dim
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Regression head (same as base model)
        self.regressor = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        for module in [self.attention, self.regressor]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass with attention.
        
        Args:
            x (torch.Tensor): Shape (batch_size, num_frames, C, H, W)
        
        Returns:
            predictions (torch.Tensor): Shape (batch_size, 1)
        """
        batch_size, num_frames, C, H, W = x.shape
        
        # Extract features
        x = x.view(batch_size * num_frames, C, H, W)
        features = self.encoder(x)
        features = self.adaptive_pool(features)
        features = features.squeeze(-1).squeeze(-1)
        features = features.view(batch_size, num_frames, self.feature_dim)
        
        # Compute attention weights
        attention_scores = self.attention(features)  # (batch_size, num_frames, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # Normalize across frames
        
        # Weighted sum of features
        pooled_features = (features * attention_weights).sum(dim=1)  # (batch_size, feature_dim)
        
        # Predict GA
        predictions = self.regressor(pooled_features)
        
        return predictions


def get_model(model_type='baseline', pretrained=True, **kwargs):
    """
    Factory function to get different model variants.
    
    Args:
        model_type (str): 'baseline' or 'attention'
        pretrained (bool): Use pretrained weights
        **kwargs: Additional arguments for model constructor
    
    Returns:
        model (nn.Module): The requested model
    """
    if model_type == 'baseline':
        return GAPredictor(pretrained=pretrained, **kwargs)
    elif model_type == 'attention':
        return GAPredictor_Attention(pretrained=pretrained, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")



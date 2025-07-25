import torch
import torch.nn as nn
import torch.nn.functional as F
from .ctx_loss import ContextualLoss_3D_RandomSampling


def compute_cosine_distance_3d(x, y):
    """Compute cosine distance for 3D features."""
    # Reshape to (N, C, H*W*D)
    N, C, H, W, D = x.size()
    x_reshaped = x.reshape(N, C, -1)  # (N, C, H*W*D)
    y_reshaped = y.reshape(N, C, -1)  # (N, C, H*W*D)
    
    # mean shifting by channel-wise mean of `y`.
    y_mu = y_reshaped.mean(dim=(0, 2), keepdim=True)
    x_centered = x_reshaped - y_mu
    y_centered = y_reshaped - y_mu

    # L2 normalization
    x_normalized = F.normalize(x_centered, p=2, dim=1)
    y_normalized = F.normalize(y_centered, p=2, dim=1)

    # Cosine similarity
    cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)  # (N, H*W*D, H*W*D)
    dist = 1 - cosine_sim

    return dist

def compute_relative_distance_3d(dist_raw):
    """Compute relative distance for 3D feature."""
    dist_min, _ = torch.min(dist_raw, dim=2, keepdim=True)
    dist_tilde = dist_raw / (dist_min + 1e-5)
    return dist_tilde

def compute_cx_3d(dist_tilde, band_width):
    """Compute contextual similarity for 3D feature."""
    # from: https://github.com/Lornatang/ContextualLoss-PyTorch/blob/main/model.py
    # for memory efficiency
    cx = F.softmax((1 - dist_tilde) / band_width, dim=2)
    return cx


def contextual_loss_3d(x: torch.Tensor,
                       y: torch.Tensor,
                       band_width: float = 0.5,
                       num_samples: int = 10,
                       neighborhood_size: int = 8):
    """
    Computes 3D contextual loss between x and y with random sampling.
    
    Parameters
    ---
    x : torch.Tensor
        features of shape (N, C, H, W, D).
    y : torch.Tensor
        features of shape (N, C, H, W, D).
    band_width : float, optional
        a band-width parameter used to convert distance to similarity.
    num_samples : int, optional
        number of random voxels to sample.
    neighborhood_size : int, optional
        size of neighborhood cube.
    
    Returns
    ---
    cx_loss : torch.Tensor
        contextual loss between x and y
    """
    assert x.size() == y.size(), 'input tensor must have the same size.'
    
    N, C, H, W, D = x.size()
    
    # 1. Random sampling of voxels (ensure indices are within bounds)
    pad_size = neighborhood_size // 2
    # Remove batch_indices random sampling - use all batches
    h_indices = torch.randint(pad_size, H - pad_size, (num_samples,), device=x.device)
    w_indices = torch.randint(pad_size, W - pad_size, (num_samples,), device=x.device)
    d_indices = torch.randint(pad_size, D - pad_size, (num_samples,), device=x.device)
    
    # 2. Compute contextual loss for each sample
    total_loss = 0.0
    
    for i in range(num_samples):
        h, w, d = h_indices[i], w_indices[i], d_indices[i]
        
        # Extract neighborhood for all batches
        x_neigh = x[:, :, h-pad_size:h+pad_size, w-pad_size:w+pad_size, d-pad_size:d+pad_size]  # (N, C, 8, 8, 8)
        y_neigh = y[:, :, h-pad_size:h+pad_size, w-pad_size:w+pad_size, d-pad_size:d+pad_size]  # (N, C, 8, 8, 8)
        
        # 1. Calculate two volume distance
        dist_raw = compute_cosine_distance_3d(x_neigh, y_neigh)  # (N, 512, 512)
        # 2. Calculate relative distance
        dist_tilde = compute_relative_distance_3d(dist_raw)
        # 3. Calculate contextual loss
        cx = compute_cx_3d(dist_tilde, band_width)
        cx = torch.mean(torch.max(cx, dim=1)[0], dim=1)  # Eq(1) - (N,)
        sample_loss = torch.mean(-torch.log(cx + 1e-5))  # Eq(5)       
        total_loss += sample_loss
    
    return total_loss / num_samples


class SwinViTContextualLoss(nn.Module):
    """
    Contextual Loss using SwinViT features for 3D medical images
    """
    
    def __init__(self, 
                 feature_extractor,
                 level=0,
                 band_width=0.5,
                 num_samples=10,
                 neighborhood_size=8):
        super(SwinViTContextualLoss, self).__init__()
        
        self.feature_extractor = feature_extractor
        self.level = level
        self.band_width = band_width
        self.num_samples = num_samples
        self.neighborhood_size = neighborhood_size
    
    def forward(self, x, y):
        """
        Forward pass for SwinViT contextual loss
        
        Args:
            x: Source image tensor of shape (N, 1, H, W, D)
            y: Target image tensor of shape (N, 1, H, W, D)
        
        Returns:
            loss: Contextual loss value for the specified level
        """
        # Extract features using SwinViT
        x_features = self.feature_extractor(x)
        y_features = self.feature_extractor(y)
        
        # Check if the specified level exists
        if self.level >= len(x_features) or self.level >= len(y_features):
            raise ValueError(f"Level {self.level} does not exist. Available levels: 0-{len(x_features)-1}")
        
        # Get features for the specified level
        x_feat = x_features[self.level]
        y_feat = y_features[self.level]
        
        # Compute contextual loss for this level using the 3D implementation
        loss = contextual_loss_3d(
            x_feat, y_feat, 
            band_width=self.band_width,
            num_samples=self.num_samples,
            neighborhood_size=self.neighborhood_size
        )
        
        return loss 
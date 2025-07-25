import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ContextualLoss_3D(nn.Module):
    """
    3D Contextual Loss for 3D medical images (CT, MRI, etc.)
    
    Based on the original 2D implementation from:
    https://github.com/Lornatang/ContextualLoss-PyTorch/blob/main/model.py
    """
    
    def __init__(self, band_width=0.5):
        super(ContextualLoss_3D, self).__init__()
        
        self.band_width = band_width
    
    def forward(self, x, y):
        """
        Forward pass for 3D contextual loss
        
        Args:
            x: Source tensor of shape (N, C, H, W, D)
            y: Target tensor of shape (N, C, H, W, D)
        
        Returns:
            loss: Contextual loss value
        """
        # Compute contextual loss
        loss = self.contextual_loss_3d(x, y)
        return loss
    
    def contextual_loss_3d(self, x, y):
        """
        Compute 3D contextual loss between x and y
        
        Args:
            x: Source features of shape (N, C, H, W, D)
            y: Target features of shape (N, C, H, W, D)
        
        Returns:
            loss: Contextual loss value
        """
        assert x.size() == y.size(), 'Input tensors must have the same size'
        
        N, C, H, W, D = x.size()
        
        # Reshape to (N, C, H*W*D)
        x_reshaped = x.reshape(N, C, -1)  # (N, C, H*W*D)
        y_reshaped = y.reshape(N, C, -1)  # (N, C, H*W*D)
        
        # Mean shifting by channel-wise mean of y
        y_mu = y_reshaped.mean(dim=(0, 2), keepdim=True)
        x_centered = x_reshaped - y_mu
        y_centered = y_reshaped - y_mu
        
        # L2 normalization
        x_normalized = F.normalize(x_centered, p=2, dim=1)
        y_normalized = F.normalize(y_centered, p=2, dim=1)
        
        # Compute cosine distance
        cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)  # (N, H*W*D, H*W*D)
        dist = 1 - cosine_sim
        
        # Compute relative distance
        dist_min, _ = torch.min(dist, dim=2, keepdim=True)
        dist_tilde = dist / (dist_min + 1e-5)
        
        # Compute contextual similarity
        cx = F.softmax((1 - dist_tilde) / self.band_width, dim=2)
        
        # Compute loss
        cx = torch.mean(torch.max(cx, dim=1)[0], dim=1)  # Eq(1)
        loss = torch.mean(-torch.log(cx + 1e-5))  # Eq(5)
        
        return loss


class ContextualLoss_3D_RandomSampling(nn.Module):
    """
    3D Contextual Loss with random sampling for memory efficiency
    """
    
    def __init__(self, band_width=0.5, num_samples=10, neighborhood_size=8):
        super(ContextualLoss_3D_RandomSampling, self).__init__()
        
        self.band_width = band_width
        self.num_samples = num_samples
        self.neighborhood_size = neighborhood_size
    
    def forward(self, x, y):
        """
        Forward pass with random sampling
        
        Args:
            x: Source tensor of shape (N, C, H, W, D)
            y: Target tensor of shape (N, C, H, W, D)
        
        Returns:
            loss: Contextual loss value
        """
        assert x.size() == y.size(), 'Input tensors must have the same size'
        
        N, C, H, W, D = x.size()
        
        # Random sampling of voxels (ensure indices are within bounds)
        pad_size = self.neighborhood_size // 2
        h_indices = torch.randint(pad_size, H - pad_size, (self.num_samples,), device=x.device)
        w_indices = torch.randint(pad_size, W - pad_size, (self.num_samples,), device=x.device)
        d_indices = torch.randint(pad_size, D - pad_size, (self.num_samples,), device=x.device)
        
        # Compute contextual loss for each sample
        total_loss = 0.0
        
        for i in range(self.num_samples):
            h, w, d = h_indices[i], w_indices[i], d_indices[i]
            
            # Extract neighborhood for all batches
            x_neigh = x[:, :, h-pad_size:h+pad_size, w-pad_size:w+pad_size, d-pad_size:d+pad_size]
            y_neigh = y[:, :, h-pad_size:h+pad_size, w-pad_size:w+pad_size, d-pad_size:d+pad_size]
            
            # Compute contextual loss for this neighborhood
            sample_loss = self._compute_contextual_loss_3d(x_neigh, y_neigh)
            total_loss += sample_loss
        
        return total_loss / self.num_samples
    
    def _compute_contextual_loss_3d(self, x, y):
        """
        Compute contextual loss for 3D neighborhood
        
        Args:
            x: Source neighborhood of shape (N, C, H, W, D)
            y: Target neighborhood of shape (N, C, H, W, D)
        
        Returns:
            loss: Contextual loss value
        """
        N, C, H, W, D = x.size()
        
        # Reshape to (N, C, H*W*D)
        x_reshaped = x.reshape(N, C, -1)
        y_reshaped = y.reshape(N, C, -1)
        
        # Mean shifting by channel-wise mean of y
        y_mu = y_reshaped.mean(dim=(0, 2), keepdim=True)
        x_centered = x_reshaped - y_mu
        y_centered = y_reshaped - y_mu
        
        # L2 normalization
        x_normalized = F.normalize(x_centered, p=2, dim=1)
        y_normalized = F.normalize(y_centered, p=2, dim=1)
        
        # Compute cosine distance
        cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)
        dist = 1 - cosine_sim
        
        # Compute relative distance
        dist_min, _ = torch.min(dist, dim=2, keepdim=True)
        dist_tilde = dist / (dist_min + 1e-5)
        
        # Compute contextual similarity
        cx = F.softmax((1 - dist_tilde) / self.band_width, dim=2)
        
        # Compute loss
        cx = torch.mean(torch.max(cx, dim=1)[0], dim=1)
        loss = torch.mean(-torch.log(cx + 1e-5))
        
        return loss

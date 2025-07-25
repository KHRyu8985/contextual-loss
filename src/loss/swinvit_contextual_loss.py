import torch
import torch.nn as nn
import torch.nn.functional as F


class SwinContextualLoss(nn.Module):
    """
    3D Contextual Loss using SwinViT features for medical images.
    
    Usage:
        loss_func = SwinContextualLoss(feature_extractor, level=1).cuda()
        loss = loss_func(pred, gt)
    """
    
    def __init__(self, 
                 feature_extractor,
                 level=0,
                 band_width=0.5,
                 num_samples=10,
                 neighborhood_size=8):
        """
        Initialize SwinViT Contextual Loss.
        
        Args:
            feature_extractor: Pre-trained SwinViT feature extractor
            level (int): Feature level to use (0=high-res, 1=mid-res, 2=low-res)
            band_width (float): Bandwidth parameter for distance-to-similarity conversion
            num_samples (int): Number of random voxels to sample for efficiency
            neighborhood_size (int): Size of neighborhood cube for sampling
        """
        super(SwinContextualLoss, self).__init__()
        
        self.feature_extractor = feature_extractor
        self.level = level
        self.band_width = band_width
        self.num_samples = num_samples
        self.neighborhood_size = neighborhood_size
        
        # Feature extractor 파라미터 동결 (loss 계산 시에만 사용)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, pred, gt):
        """
        Compute contextual loss between prediction and ground truth.
        
        Args:
            pred: Predicted image tensor of shape (N, 1, H, W, D)
            gt: Ground truth image tensor of shape (N, 1, H, W, D)
        
        Returns:
            loss: Contextual loss value as a scalar tensor
        """
        assert pred.size() == gt.size(), 'Prediction and ground truth must have the same size'
        
        # Extract features using SwinViT (gradients preserved for backpropagation)
        pred_features = self.feature_extractor(pred)
        gt_features = self.feature_extractor(gt)
        
        # Validate level
        if self.level >= len(pred_features) or self.level >= len(gt_features):
            raise ValueError(f"Level {self.level} not available. Valid levels: 0-{len(pred_features)-1}")
        
        # Get features for the specified level
        pred_feat = pred_features[self.level]
        gt_feat = gt_features[self.level]
        
        # Compute contextual loss
        loss = self._compute_contextual_loss(pred_feat, gt_feat)
        
        return loss
    
    def _compute_contextual_loss(self, x, y):
        """
        Compute 3D contextual loss between feature tensors with random sampling.
        
        Args:
            x: Source features of shape (N, C, H, W, D)
            y: Target features of shape (N, C, H, W, D)
        
        Returns:
            loss: Contextual loss value
        """
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
            
            # Compute loss for this neighborhood
            sample_loss = self._compute_neighborhood_loss(x_neigh, y_neigh)
            total_loss += sample_loss
        
        return total_loss / self.num_samples
    
    def _compute_neighborhood_loss(self, x, y):
        """
        Compute contextual loss for a neighborhood pair.
        
        Args:
            x: Source neighborhood of shape (N, C, H, W, D)
            y: Target neighborhood of shape (N, C, H, W, D)
        
        Returns:
            loss: Contextual loss for this neighborhood
        """
        # Reshape to (N, C, H*W*D)
        N, C, H, W, D = x.size()
        x_reshaped = x.reshape(N, C, -1)
        y_reshaped = y.reshape(N, C, -1)
        
        # Mean shifting by channel-wise mean of target
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
        
        # Compute final loss
        cx = torch.mean(torch.max(cx, dim=1)[0], dim=1)  # Eq(1)
        loss = torch.mean(-torch.log(cx + 1e-5))  # Eq(5)
        
        return loss


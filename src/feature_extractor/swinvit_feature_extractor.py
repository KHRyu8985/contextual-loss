import autorootcwd
import torch
import copy
from monai.networks.nets import SwinUNETR
from transformers import PreTrainedModel
from .config import SwinConfig


class SwinExtractor(PreTrainedModel):
    """
    SwinViT-based 3D feature extractor with Hugging Face Transformers integration.
    """
    config_class = SwinConfig
    base_model_prefix = "swin"
    
    def __init__(self, config):
        super().__init__(config)
        
        # Create SwinUNETR model with config parameters
        model = SwinUNETR(
            in_channels=config.in_channels,
            out_channels=14,  # Fixed for segmentation task
            feature_size=config.feature_size,
            use_checkpoint=config.use_checkpoint,
        )
        
        # Extract only the encoder (SwinViT) part
        self.encoder = copy.deepcopy(model.swinViT)
        del model
    
    @classmethod
    def from_legacy_weights(cls, pretrained_weight_path, config=None):
        """
        Load from legacy .pt weight file.
        
        Args:
            pretrained_weight_path: Path to .pt weight file
            config: SwinConfig instance, uses default if None
        """
        if config is None:
            config = SwinConfig()
        
        # Create model instance
        model = cls(config)
        
        # Load legacy weights
        full_model = SwinUNETR(
            in_channels=config.in_channels,
            out_channels=14,
            feature_size=config.feature_size,
            use_checkpoint=config.use_checkpoint,
        )
        
        # Load weights
        weight = torch.load(pretrained_weight_path, map_location='cpu')
        full_model.load_from(weights=weight)
        
        # Copy encoder weights
        model.encoder = copy.deepcopy(full_model.swinViT)
        del full_model
        
        return model

    def forward(self, x):
        """
        Forward pass through SwinViT encoder.
        
        Args:
            x: Input tensor of shape (N, C, H, W, D)
            
        Returns:
            List of feature tensors from different levels
        """
        feats = self.encoder(x)
        return feats


# Legacy alias for backward compatibility
SwinViTFeatureExtractor = SwinExtractor 
    
if __name__ == "__main__":
    extractor = SwinViTFeatureExtractor()
    input_tensor = torch.randn(1, 1, 128, 128, 128)
    feats = extractor(input_tensor)
    print(len(feats))
    print(feats[0].shape)
    print(feats[1].shape)
    print(feats[2].shape)
    print(feats[3].shape)
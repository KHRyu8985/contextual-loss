import autorootcwd
import torch
import copy
from monai.networks.nets import SwinUNETR

class SwinViTFeatureExtractor(torch.nn.Module):
    def __init__(self, in_channels=1, feature_size=48, use_checkpoint=False, pretrained_weight_path='weights/model_swinvit.pt', device=None):
        super().__init__()
        model = SwinUNETR(
            in_channels=in_channels,
            out_channels=14,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
        )
        if pretrained_weight_path is not None:
            weight = torch.load(pretrained_weight_path, map_location=device)
            model.load_from(weights=weight)
        self.encoder = copy.deepcopy(model.swinViT)
        if device is not None:
            self.encoder.to(device)
        self.encoder.eval()
        self.encoder.requires_grad_(False)
        del model  # 이제 encoder만 남음

    def forward(self, x):
        feats = self.encoder(x)
        return feats 
    
if __name__ == "__main__":
    extractor = SwinViTFeatureExtractor()
    input_tensor = torch.randn(1, 1, 128, 128, 128)
    feats = extractor(input_tensor)
    print(len(feats))
    print(feats[0].shape)
    print(feats[1].shape)
    print(feats[2].shape)
    print(feats[3].shape)
import autorootcwd
import torch
import copy
from src.feature_extractor.anatomix import Unet
import torch.nn as nn

class AnatomixUnetFeatureExtractor(torch.nn.Module):
    def __init__(
        self,
        dimension=3,
        input_nc=1,
        output_nc=16,
        num_downs=4,
        ngf=16,  # convex_adam_utils.py와 동일하게!
        norm="batch",
        final_act="none",
        activation="relu",
        pad_type="reflect",
        doubleconv=True,
        residual_connection=False,
        pooling="Max",
        interp="nearest",
        use_skip_connection=True,
        pretrained_weight_path='weights/anatomix.pth',
        device=None
    ):
        super().__init__()
        self.model = Unet(
            dimension=dimension,
            input_nc=input_nc,
            output_nc=output_nc,
            num_downs=num_downs,
            ngf=ngf,
            norm=norm,
            final_act=final_act,
            activation=activation,
            pad_type=pad_type,
            doubleconv=doubleconv,
            residual_connection=residual_connection,
            pooling=pooling,
            interp=interp,
            use_skip_connection=use_skip_connection,
        )
        if pretrained_weight_path is not None:
            weight = torch.load(pretrained_weight_path, map_location=device)
            self.model.load_state_dict(weight, strict=True)
        if device is not None:
            self.model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.encoder_idx = self.model.encoder_idx

    def forward(self, x, verbose=False):
        _, feats = self.model(x, layers=self.encoder_idx[1:], verbose=verbose)
        return feats

if __name__ == "__main__":
    extractor = AnatomixUnetFeatureExtractor()
    input_tensor = torch.randn(1, 1, 128, 128, 128)
    feats = extractor(input_tensor)
    for i, f in enumerate(feats):
        print(f"encoder feature {i} shape:", f.shape) 
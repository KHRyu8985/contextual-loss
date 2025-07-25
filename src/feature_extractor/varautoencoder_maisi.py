import autorootcwd
import torch
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi

network = AutoencoderKlMaisi(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        num_channels=[
            64,
            128,
            256
        ],
        num_res_blocks=[2,2,2],
        norm_num_groups=32,
        norm_eps=1e-06,
        attention_levels=[
            False,
            False,
            False
        ],
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
        use_checkpointing=False,
        use_convtranspose=False,
        norm_float16=False,
        num_splits=8,
        dim_split=1
)

ckpt_path = "weights/autoencoder_epoch273.pt"
ckpt = torch.load(ckpt_path)

network.load_state_dict(ckpt, strict=True)

network = network
encoder = network.encoder

device = "cuda"
encoder = encoder.to(device)
test_input = torch.randn(1, 1, 128, 128, 128).to(device)
test_output = encoder(test_input)

print(test_output.shape)
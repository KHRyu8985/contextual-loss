from transformers import PretrainedConfig


class SwinConfig(PretrainedConfig):
    """
    Swin configuration for 3D medical image feature extraction.
    """
    
    model_type = "swin"
    
    def __init__(
        self,
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=48,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        feature_size=48,
        norm_name="instance",
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        normalize=True,
        use_checkpoint=False,
        spatial_dims=3,
        downsample="merging",
        use_v2=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depths = depths
        self.num_heads = num_heads
        self.feature_size = feature_size
        self.norm_name = norm_name
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.dropout_path_rate = dropout_path_rate
        self.normalize = normalize
        self.use_checkpoint = use_checkpoint
        self.spatial_dims = spatial_dims
        self.downsample = downsample
        self.use_v2 = use_v2
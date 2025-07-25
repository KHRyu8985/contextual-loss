import autorootcwd
import torch
import os
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    SaveImaged,
    RandSpatialCropd
)
from src.feature_extractor.swinvit_feature_extractor import SwinViTFeatureExtractor

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = SwinViTFeatureExtractor(
        pretrained_weight_path='weights/model_swinvit.pt'
    )
    extractor.to(device)

    # 실제 CT 파일 경로
    ct_path = "2ABA002/ct.nii"

    # MONAI transform으로 전처리
    val_transforms = Compose(
        [
            LoadImaged(keys=["image"], ensure_channel_first=True),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),
            Spacingd(
                keys=["image"],
                pixdim=(1.0, 1.0, 1.0),
                mode="bilinear",
            ),
            RandSpatialCropd(keys=["image"], roi_size=(192, 192, 192), random_size=False),
            EnsureTyped(keys=["image"], device=device, track_meta=True),
        ]
    )

    ct_dict = {"image": ct_path}
    ct_transformed = val_transforms(ct_dict)
    ct_normalized = ct_transformed["image"]

    # 배치 차원 추가
    x = ct_normalized.unsqueeze(0).to(device)
    print(f"CT data shape: {x.shape}")

    # feature 추출
    feats = extractor(x)
    print(f"encoder feature 개수: {len(feats)}")
    for i, f in enumerate(feats):
        print(f"encoder feature {i} shape: {f.shape}")

    # feature 저장 (level 0/1/2의 처음 8개 채널을 multi-phase로 저장)
    os.makedirs("results/swinvit", exist_ok=True)
    for level in range(3):
        feat = feats[level].squeeze(0)  # (C, H, W, D)
        feat_channels = feat   # (8, H, W, D)
        print(f"level {level} 채널들 형태: {feat_channels.shape}")
        print(f"level {level} 저장 형태: {feat_channels.shape}")
        save_transform = SaveImaged(
            keys=["feature"],
            output_dir="results/swinvit",
            output_postfix=f"level{level}_multiphase",
            output_ext=".nii.gz",
            print_log=False
        )
        channel_dict = {"feature": feat_channels}
        save_transform(channel_dict)
        print(f"level {level} multi-phase feature가 저장되었습니다.") 
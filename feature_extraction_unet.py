import autorootcwd
import torch, os
from monai.networks.nets import SwinUNETR
from monai.apps.utils import download_url

# HU clipping and normalization - MONAI 스타일로 변경
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    CropForegroundd,
    Orientationd,
    Spacingd,
    EnsureTyped,
    SaveImaged,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = SwinUNETR(
    in_channels=1,
    out_channels=14,
    feature_size=48,
    use_checkpoint=True,
).to(device)

weight = torch.load("weights/model_swinvit.pt", weights_only=True)
model.load_from(weights=weight)
print("Using pretrained self-supervied Swin UNETR backbone weights !")

enc = model.swinViT
for p in enc.parameters():
    p.requires_grad = False

# Load CT scan from 2ABA002
ct_path = "2ABA002/ct.nii"

val_transforms = Compose(
    [
        LoadImaged(keys=["image"], ensure_channel_first=True),
        ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
        Spacingd(
            keys=["image"],
            pixdim=(1.5, 1.5, 2.0),
            mode="bilinear",
        ),
        EnsureTyped(keys=["image"], device=device, track_meta=True),
    ]
)

# CT 파일 경로를 딕셔너리 형태로 변환하여 transforms 적용
ct_dict = {"image": ct_path}
ct_transformed = val_transforms(ct_dict)
ct_normalized = ct_transformed["image"]

# Convert to tensor and add batch dimension
x = ct_normalized.unsqueeze(0).to(device)  # 이미 tensor이므로 torch.from_numpy 불필요
print(f"CT data shape: {x.shape}")
with torch.no_grad():
    feats = enc(x)          # tuple(level0…3); pick any level for contextual loss
    for i, feat in enumerate(feats):
        print(f"feats[{i}].shape: {feat.shape}")

# 첫 번째 feature를 nii.gz로 저장하여 확인
feat0 = feats[0]  # 첫 번째 레벨의 feature
print(f"첫 번째 feature 형태: {feat0.shape}")

# 처음 8개 채널 저장 (level 0) - multi-phase로 저장
feat0_channels = feat0[:8]  # 처음 8개 채널 선택 (tensor 상태 유지)
print(f"level 0 채널들 형태: {feat0_channels.shape}")
# (8, H, W, D) 형태로 차원 조정
feat0_channels = feat0_channels.squeeze(0)  # 배치 차원 제거
print(f"level 0 조정된 형태: {feat0_channels.shape}")

save_transform = SaveImaged(
    keys=["feature"],
    output_dir="results/swinvit",
    output_postfix="level0_multiphase",
    output_ext=".nii.gz",
    print_log=False
)
channel_dict = {"feature": feat0_channels}
save_transform(channel_dict)
print("level 0 multi-phase feature가 저장되었습니다.")

# feature level 1의 처음 8개 채널 저장 - multi-phase로 저장
feat1 = feats[1]
print(f"feature level 1 형태: {feat1.shape}")

feat1_channels = feat1[:8]  # 처음 8개 채널 선택 (tensor 상태 유지)
print(f"level 1 채널들 형태: {feat1_channels.shape}")
# (8, H, W, D) 형태로 차원 조정
feat1_channels = feat1_channels.squeeze(0)  # 배치 차원 제거
print(f"level 1 조정된 형태: {feat1_channels.shape}")

save_transform = SaveImaged(
    keys=["feature"],
    output_dir="results/swinvit",
    output_postfix="level1_multiphase",
    output_ext=".nii.gz",
    print_log=False
)
channel_dict = {"feature": feat1_channels}
save_transform(channel_dict)
print("level 1 multi-phase feature가 저장되었습니다.")

# feature level 2의 처음 8개 채널 저장 - multi-phase로 저장
feat2 = feats[2]
print(f"feature level 2 형태: {feat2.shape}")

feat2_channels = feat2[:8]  # 처음 8개 채널 선택 (tensor 상태 유지)
print(f"level 2 채널들 형태: {feat2_channels.shape}")
# (8, H, W, D) 형태로 차원 조정
feat2_channels = feat2_channels.squeeze(0)  # 배치 차원 제거
print(f"level 2 조정된 형태: {feat2_channels.shape}")

save_transform = SaveImaged(
    keys=["feature"],
    output_dir="results/swinvit",
    output_postfix="level2_multiphase",
    output_ext=".nii.gz",
    print_log=False
)
channel_dict = {"feature": feat2_channels}
save_transform(channel_dict)
print("level 2 multi-phase feature가 저장되었습니다.")

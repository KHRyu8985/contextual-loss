# 3D Contextual Loss for Medical Images

이 프로젝트는 3D 의료 이미지(CT, MRI 등)를 위한 Contextual Loss를 구현합니다. SwinViT feature extractor를 사용하여 고품질의 특징을 추출하고, 맥락적 손실을 계산합니다.

## 주요 기능

- **3D Contextual Loss**: 3D 의료 이미지에 최적화된 contextual loss
- **SwinViT Feature Extractor**: 사전 훈련된 SwinViT 모델을 사용한 특징 추출
- **Single Level Loss**: 특정 level의 feature만 사용한 contextual loss
- **Random Sampling**: 메모리 효율성을 위한 랜덤 샘플링

## 설치

### Requirements

- Python >= 3.12
- CUDA 지원 GPU (권장)

### UV를 사용한 설치 (권장)

```bash
# UV 설치 (아직 설치하지 않은 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 클론
git clone <repository-url>
cd contextual-loss

# 의존성 설치 및 가상환경 생성
uv sync

# 개발 의존성까지 모두 설치
uv sync --all-extras
```

### Pip를 사용한 설치

```bash
# 프로젝트 클론
git clone <repository-url>
cd contextual-loss

# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 또는 .venv\Scripts\activate  # Windows

# 의존성 설치
pip install -e .

# 개발 의존성 설치 (선택사항)
pip install -e ".[dev]"
```

### 다른 사용자를 위한 재현 가능한 설치

이 프로젝트는 `uv.lock` 파일을 포함하고 있어 정확히 동일한 패키지 버전으로 설치할 수 있습니다:

```bash
# UV로 정확한 버전 재현
uv sync --locked

# 또는 pip를 사용하여 lock 파일 기반 설치
pip install -r <(uv export --format requirements-txt)
```

## 사용법

### SwinViT Contextual Loss 사용법

```python
import torch
from src.feature_extractor.swinvit_feature_extractor import SwinViTFeatureExtractor
from src.contextual_loss import SwinViTContextualLoss

# Feature extractor 초기화
feature_extractor = SwinViTFeatureExtractor(
    pretrained_weight_path='weights/model_swinvit.pt'
)

# Contextual loss 초기화 (level 1 사용)
contextual_loss = SwinViTContextualLoss(
    feature_extractor=feature_extractor,
    level=1,  # 사용할 level 지정 (0, 1, 2 중 선택)
    band_width=0.5,
    num_samples=10,
    neighborhood_size=8
)

# 입력 데이터 (CT 스캔 형태)
x = torch.randn(1, 1, 192, 192, 192)  # (batch, channel, H, W, D)
y = torch.randn(1, 1, 192, 192, 192)

# Loss 계산
loss = contextual_loss(x, y)
print(f"Level 1 Contextual Loss: {loss.item():.6f}")
```

### 기본 3D Contextual Loss 사용법

```python
from src.contextual_loss import ContextualLoss_3D_RandomSampling

# 기본 3D contextual loss
loss_fn = ContextualLoss_3D_RandomSampling(
    band_width=0.5,
    num_samples=10,
    neighborhood_size=8
)

# 3D feature 텐서로 loss 계산
x_feat = torch.randn(1, 64, 32, 32, 32)  # (batch, channels, H, W, D)
y_feat = torch.randn(1, 64, 32, 32, 32)
loss = loss_fn(x_feat, y_feat)
```

### 훈련 예제

```python
import torch.nn as nn
import torch.optim as optim

# 네트워크 정의
class My3DNet(nn.Module):
    def __init__(self):
        super(My3DNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv3d(64, 1, 3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# 네트워크 및 loss 함수 초기화
net = My3DNet()
contextual_loss = SwinViTContextualLoss(
    feature_extractor=feature_extractor,
    level=1  # Level 1 사용
)
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 훈련 루프
for epoch in range(num_epochs):
    for batch_idx, (input_data, target_data) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Forward pass
        output = net(input_data)
        
        # Loss 계산
        loss = contextual_loss(output, target_data)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
```

## 파라미터 설명

### SwinViT Contextual Loss 파라미터

- **level** (int): 사용할 SwinViT feature level (0, 1, 2 중 선택)
- **band_width** (float): 거리를 유사도로 변환하는 밴드폭 파라미터 (기본값: 0.5)
- **num_samples** (int): 랜덤 샘플링할 복셀 수 (기본값: 10)
- **neighborhood_size** (int): 이웃 영역의 크기 (기본값: 8)

### 기본 Contextual Loss 파라미터

- **band_width** (float): 거리를 유사도로 변환하는 밴드폭 파라미터 (기본값: 0.5)
- **num_samples** (int): 랜덤 샘플링할 복셀 수 (기본값: 10)
- **neighborhood_size** (int): 이웃 영역의 크기 (기본값: 8)

## SwinViT Feature Levels

SwinViT feature extractor는 여러 level의 특징을 제공합니다:

- **Level 0**: 고해상도 특징 (세밀한 디테일)
- **Level 1**: 중간 해상도 특징 (중간 수준의 특징)
- **Level 2**: 저해상도 특징 (전역적 특징)

각 level은 다른 해상도와 특징을 가지므로, 사용 목적에 따라 적절한 level을 선택할 수 있습니다.

## 테스트

```bash
# SwinViT contextual loss 테스트
python scripts/test_swinvit_contextual_loss.py
```

## 파일 구조

```
src/
├── contextual_loss/
│   ├── __init__.py
│   ├── ctx_loss.py                    # 기본 3D contextual loss
│   └── swinvit_contextual_loss.py    # SwinViT 기반 contextual loss
├── feature_extractor/
│   └── swinvit_feature_extractor.py  # SwinViT feature extractor
scripts/
├── test_compute_cosine_distance.py   # 기본 3D loss 테스트
├── test_swinvit_feature_extractor.py # Feature extractor 테스트
└── test_swinvit_contextual_loss.py   # SwinViT contextual loss 테스트
```

## 참고 자료

- [ContextualLoss-PyTorch](https://github.com/Lornatang/ContextualLoss-PyTorch): 원본 2D 구현
- [MONAI](https://monai.io/): 의료 이미지 처리 라이브러리
- [SwinViT](https://github.com/microsoft/Swin-Transformer): Vision Transformer

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

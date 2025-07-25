import autorootcwd
import torch
import torch.nn as nn
import copy
from src.feature_extractor.resnet import ResNet, BasicBlock

class ResNetFeatureExtractor(torch.nn.Module):
    def __init__(self, in_channels=1, pretrained_weight_path='weights/resnet/resnet_34.pth', device=None):
        super().__init__()
        
        # ResNet34 모델 생성 (conv_seg 제외)
        self.model = ResNet(BasicBlock, [3, 4, 6, 3], num_seg_classes=2)
        
        # pretrained weight 로드
        if pretrained_weight_path is not None:
            checkpoint = torch.load(pretrained_weight_path, map_location=device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # module. 접두사 제거
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]  # 'module.' 제거
                else:
                    new_key = key
                new_state_dict[new_key] = value
            
            # conv_seg 관련 키 제거 (feature extraction만을 위해)
            encoder_state_dict = {}
            for key, value in new_state_dict.items():
                if not key.startswith('conv_seg.'):
                    encoder_state_dict[key] = value
            
            self.model.load_state_dict(encoder_state_dict, strict=False)
        
        # 각 레이어를 개별적으로 접근할 수 있도록 설정
        self.conv1 = self.model.conv1
        self.bn1 = self.model.bn1
        self.relu = self.model.relu
        self.maxpool = self.model.maxpool
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4
        
        if device is not None:
            self.to(device)
        self.eval()
        self.requires_grad_(False)
        
        # 원본 모델 삭제
        del self.model

    def forward(self, x):
        with torch.no_grad():
            # 각 레이어의 feature 추출
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            
            x = self.maxpool(x)
            
            x = self.layer1(x)
            layer1_feat = x
            
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            layer4_feat = x
            
            # layer1과 layer4만 반환
            features = [layer1_feat, layer4_feat]
            
        return features

if __name__ == "__main__":
    extractor = ResNetFeatureExtractor()
    input_tensor = torch.randn(1, 1, 128, 128, 128)
    features = extractor(input_tensor)
    
    print("Feature shapes:")
    feature_names = ['conv1', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']
    for name, feat in zip(feature_names, features):
        print(f"  {name}: {feat.shape}") 
import autorootcwd
import torch
from src.loss import SwinContextualLoss

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # SwinViT feature extractor 로드 (새로운 방식)
    from src.feature_extractor.swinvit_feature_extractor import SwinExtractor
    feature_extractor = SwinExtractor.from_pretrained('weights/swin').to(device)
    
    # SwinContextualLoss 초기화 (내부에서 자동으로 feature_extractor 동결)
    loss_func = SwinContextualLoss(
        feature_extractor=feature_extractor,
        level=1,
        band_width=0.5,
        num_samples=5,
        neighborhood_size=8
    ).to(device)
    
    print("모델 로드 완료")
    
    # 테스트 1: 기본 forward pass
    print("=== Test 1: Forward Pass ===")
    x = torch.randn(1, 1, 96, 96, 96).to(device)
    y = torch.randn(1, 1, 96, 96, 96).to(device)
    
    loss = loss_func(x, y)
    print(f"Loss: {loss.item():.6f}")
    
    # 테스트 2: Gradient 계산 (실제 훈련 시나리오)
    print("\n=== Test 2: Gradient Computation (Training Scenario) ===")
    
    # 더미 네트워크 생성 (실제 훈련에서 업데이트될 파라미터)
    dummy_net = torch.nn.Conv3d(1, 1, 3, padding=1).to(device)
    
    # 입력 이미지 (gradient 불필요)
    x = torch.randn(1, 1, 96, 96, 96).to(device)
    y = torch.randn(1, 1, 96, 96, 96).to(device)
    
    # 더미 네트워크 통과 (예: 생성된 이미지)
    pred = dummy_net(x)
    
    # Loss 계산
    loss = loss_func(pred, y)
    print(f"Loss: {loss.item():.6f}")
    
    # Backward pass
    loss.backward() # 잘되는지 확인
    print('Loss backward successful')
    
    # 테스트 3: 동일 이미지 (낮은 loss 확인)
    print("\n=== Test 3: Same Image (Should have low loss) ===")
    x_same = torch.randn(1, 1, 96, 96, 96).to(device)
    loss_same = loss_func(x_same, x_same)
    print(f"Same image loss: {loss_same.item():.6f}")
    
    print("\n✅ 모든 테스트 완료!")
    print(f"Feature extractor parameters frozen: {not any(p.requires_grad for p in feature_extractor.parameters())}") 
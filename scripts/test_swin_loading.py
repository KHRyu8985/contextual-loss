"""
Test SwinViT pretrained model loading and usage
"""
import autorootcwd
import torch
from src.feature_extractor.swinvit_feature_extractor import SwinExtractor
from src.loss import SwinContextualLoss

def test_swin_loading():
    print("🧪 Testing SwinViT Pretrained Model Loading")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load from pretrained model
    print("\n=== Loading from Pretrained Model ===")
    extractor = SwinExtractor.from_pretrained('weights/swin')
    extractor.to(device)
    print("✅ Loaded from weights/swin")
    
    # Test feature extraction
    print("\n=== Feature Extraction Test ===")
    x = torch.randn(1, 1, 96, 96, 96).to(device)
    features = extractor(x)
    print(f"Features extracted: {len(features)} levels")
    for i, feat in enumerate(features):
        print(f"Level {i}: {feat.shape}")
    
    # Test with SwinContextualLoss
    print("\n=== Contextual Loss Test ===")
    loss_func = SwinContextualLoss(
        feature_extractor=extractor,
        level=1,
        band_width=0.5,
        num_samples=5,
        neighborhood_size=8
    ).to(device)
    
    pred = torch.randn(1, 1, 96, 96, 96).to(device)
    target = torch.randn(1, 1, 96, 96, 96).to(device)
    
    loss = loss_func(pred, target)
    print(f"✅ Contextual loss: {loss.item():.6f}")
    
    # Test same image (should be low loss)
    same_loss = loss_func(pred, pred)
    print(f"✅ Same image loss: {same_loss.item():.10f}")
    
    # Show model info
    print("\n=== Model Information ===")
    print(f"Model class: {type(extractor).__name__}")
    print(f"Base model prefix: {extractor.base_model_prefix}")
    print(f"Config class: {extractor.config_class.__name__}")
    print(f"Parameters frozen: {not any(p.requires_grad for p in extractor.parameters())}")
    
    print("\n🎉 All pretrained model tests passed!")

if __name__ == "__main__":
    test_swin_loading()
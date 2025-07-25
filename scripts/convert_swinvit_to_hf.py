"""
Convert legacy SwinViT weights to Hugging Face format
"""
import autorootcwd
import torch
from src.feature_extractor.swinvit_feature_extractor import SwinExtractor

def convert_weights():
    print("Converting SwinViT weights to Hugging Face format...")
    
    # Load from legacy weights
    extractor = SwinExtractor.from_legacy_weights('weights/model_swinvit.pt')
    print("✅ Loaded from legacy weights/model_swinvit.pt")
    
    # Save in Transformers format to weights folder
    save_path = "weights/swin"
    extractor.save_pretrained(save_path)
    print(f"✅ Saved in Transformers format to {save_path}")
    
    # Test loading with from_pretrained
    loaded_extractor = SwinExtractor.from_pretrained(save_path)
    print(f"✅ Successfully loaded with from_pretrained() from {save_path}")
    
    # Test functionality
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_extractor.to(device)
    
    x = torch.randn(1, 1, 96, 96, 96).to(device)
    features = loaded_extractor(x)
    print(f"✅ Feature extraction works: {len(features)} levels")
    
    # Show configuration
    print("\n=== Configuration ===")
    config = loaded_extractor.config
    print(f"Model type: {config.model_type}")
    print(f"Image size: {config.img_size}")
    print(f"Feature size: {config.feature_size}")
    print(f"In channels: {config.in_channels}")
    
    print(f"\n🎉 Conversion completed! Use: SwinExtractor.from_pretrained('{save_path}')")

if __name__ == "__main__":
    convert_weights()
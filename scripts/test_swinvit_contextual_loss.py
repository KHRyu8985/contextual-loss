import autorootcwd
import torch
import torch.nn as nn
import gc
import subprocess
import re
from src.feature_extractor.swinvit_feature_extractor import SwinViTFeatureExtractor
from src.contextual_loss import SwinViTContextualLoss


def get_gpu_memory_nvidia_smi():
    """nvidia-smi 명령어로 GPU 메모리 사용량을 MB 단위로 반환"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            # 출력 예시: "1234, 8192" (used, total)
            match = re.search(r'(\d+),\s*(\d+)', result.stdout.strip())
            if match:
                used_mb = int(match.group(1))
                total_mb = int(match.group(2))
                return used_mb, total_mb
    except Exception as e:
        print(f"nvidia-smi 실행 오류: {e}")
    return 0, 0


def clear_gpu_memory():
    """GPU 메모리 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def test_swinvit_contextual_loss():
    """SwinViT contextual loss 테스트"""
    print("=== SwinViT Contextual Loss 테스트 ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 초기 GPU 메모리 상태
    used_mb, total_mb = get_gpu_memory_nvidia_smi()
    print(f"초기 GPU 메모리: {used_mb} MB / {total_mb} MB")
    
    # SwinViT feature extractor 초기화
    feature_extractor = SwinViTFeatureExtractor(
        pretrained_weight_path='weights/model_swinvit.pt'
    ).to(device)
    
    # 테스트용 3D 이미지 생성
    x = torch.randn(1, 1, 128, 128, 128).to(device)
    y = torch.randn(1, 1, 128, 128, 128).to(device)
    
    # Level 1 테스트
    level = 1
    print(f"\n--- Level {level} 테스트 ---")
    
    # SwinViT contextual loss 초기화
    contextual_loss = SwinViTContextualLoss(
        feature_extractor=feature_extractor,
        level=level,
        band_width=0.5,
        num_samples=5,
        neighborhood_size=8
    ).to(device)
    
    try:
        # Forward pass
        loss = contextual_loss(x, y)
        print(f"Level {level} contextual loss: {loss.item():.6f}")
        
        # GPU 메모리 확인
        used_mb, total_mb = get_gpu_memory_nvidia_smi()
        print(f"Forward pass 후 GPU 메모리: {used_mb} MB / {total_mb} MB")
        
        # Backward pass 테스트
        x.requires_grad_(True)
        y.requires_grad_(True)
        loss = contextual_loss(x, y)
        loss.backward()
        
        print(f"x.grad is not None: {x.grad is not None}")
        print(f"y.grad is not None: {y.grad is not None}")
        if x.grad is not None:
            print(f"x.grad norm: {x.grad.norm().item():.6f}")
        if y.grad is not None:
            print(f"y.grad norm: {y.grad.norm().item():.6f}")
        
        # Backward pass 후 GPU 메모리 확인
        used_mb, total_mb = get_gpu_memory_nvidia_smi()
        print(f"Backward pass 후 GPU 메모리: {used_mb} MB / {total_mb} MB")
        
    except Exception as e:
        print(f"Level {level} 오류: {e}")
    
    # 메모리 정리
    del contextual_loss, feature_extractor, x, y
    clear_gpu_memory()
    
    # 최종 메모리 상태
    used_mb, total_mb = get_gpu_memory_nvidia_smi()
    print(f"\n최종 GPU 메모리: {used_mb} MB / {total_mb} MB")
    
    print("\n✅ SwinViT contextual loss 테스트 완료!")


if __name__ == "__main__":
    print("🚀 SwinViT Contextual Loss 테스트 시작\n")
    
    try:
        test_swinvit_contextual_loss()
        print("\n🎉 테스트가 성공적으로 완료되었습니다!")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        clear_gpu_memory()
        print("✅ 메모리 정리 완료!") 
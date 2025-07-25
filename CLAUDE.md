# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Do not make additional function or code without any permission. Try to avoid making new .py file or folders! this is essential

**CRITICAL: ALL EXECUTABLE CODE MUST BE PLACED IN THE `/scripts/` FOLDER**
- Never create .py files in the root directory or any other location
- All test files, example scripts, and executable code belong in `/scripts/`
- If you need to create executable code, it MUST go in `/scripts/` folder only

## Project Overview

This is a **3D Contextual Loss for Medical Images** project implementing contextual loss functions optimized for 3D medical imaging data (CT, MRI, etc.). The project combines custom 3D contextual loss implementations with multiple feature extraction backends including SwinViTㅊ, anatomix, and ResNet models.

## Core Architecture

### Feature Extraction Backends
- **SwinViT**: Uses SwinUNETR-based 3D vision transformers with multi-level feature extraction (levels 0, 1, 2)
feature learning

### Contextual Loss Variants
- **Base 3D Contextual Loss** (`ctx_loss.py`): Full volume computation for smaller datasets
- **Random Sampling Version**: Memory-efficient implementation using neighborhood sampling for large 3D volumes
- **SwinViT Contextual Loss** (`swinvit_contextual_loss.py`): Multi-level feature extraction with configurable depth

### Key Components
- `/src/`: Core implementation modules
- `/data/`: Dataset saving
- `/results/`: saving results when running the code
- `/weights/`: Pre-trained model weights (SwinViT)
- `/scripts/`: Testing and validation scripts with GPU memory monitoring

- please do not make additional folder and do not make python code outside.

## Development Commands

### Testing
```bash
# Test SwinViT contextual loss implementation
python scripts/test_swinvit_contextual_loss.py

# Test specific feature extractors
python -c "from src.swinvit_feature_extractor import SwinViTFeatureExtractor; print('Import successful')"
```

### Package Management
```bash
# Install in development mode
pip install -e .

# Check dependencies
python -c "import torch, monai, autorootcwd; print('Dependencies available')"
```

## Memory Management for 3D Medical Images

This codebase is designed to handle large 3D medical volumes efficiently:

- **Random Sampling**: Use `num_samples` parameter to limit memory usage
- **Neighborhood Size**: Configure `neighborhood_size` for local feature computation
- **Multi-level Processing**: Choose appropriate SwinViT levels (0=high-res, 1=mid-res, 2=low-res)
- **GPU Monitoring**: Scripts include NVIDIA-SMI integration for memory tracking

## Integration Patterns

### Typical Usage Flow
1. Initialize feature extractor with pre-trained weights from `/weights/`
2. Create contextual loss with appropriate sampling parameters
3. Integrate loss function into training loop
4. Monitor GPU memory usage during processing

### Feature Extractor Selection
- **SwinViT**: Best for transformer-based features, supports multi-level extraction
- **anatomix**: Optimal for cross-modal biomedical tasks, contrastively pre-trained
- **ResNet**: Classical CNN features, multiple depth options available

## Dependencies

Core dependencies managed via `pyproject.toml`:
- **PyTorch**: Deep learning framework
- **MONAI**: Medical Open Network for AI
- **autorootcwd**: Automatic root directory detection
- Medical imaging: `nibabel`, `scipy`, `scikit-image`

## Pre-trained Models

Located in `/weights/` directory:
- **SwinViT**: `model_swinvit.pt` - 3D vision transformer
- **anatomix**: `anatomix.pth`, `anatomix+brains.pth` - biomedical feature extractors
- **ResNet**: Multiple variants (ResNet-10 through ResNet-200)
- **Autoencoder**: `autoencoder_epoch273.pt` - variational autoencoder
- **Segmentation**: `supervised_suprem_segresnet_2100.pth` - medical segmentation model

## Performance Considerations

- Large 3D volumes require memory-efficient sampling strategies
- Multi-level feature extraction allows trade-offs between detail and memory
- Pre-trained weights enable training-free feature extraction
- GPU memory monitoring is essential for 3D medical imaging workflows
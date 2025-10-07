# Data Augmentation Research: PatchShuffle on CIFAR-10

A comprehensive study comparing the effectiveness of data augmentation techniques, featuring a novel **PatchShuffle** method implemented on the CIFAR-10 dataset using PyTorch.

##  Project Overview

This project demonstrates the impact of data augmentation on deep learning model performance by implementing and comparing:
- **Baseline**: Training without data augmentation
- **Enhanced**: Training with traditional augmentation + novel PatchShuffle technique

The study achieved a **5.39% improvement** in test accuracy (79.14% → 84.53%) using the combined augmentation approach.

##  Key Features

### Novel PatchShuffle Implementation
- **Custom augmentation technique** that randomly shuffles pixels within non-overlapping patches
- Configurable patch size (default: 4x4) and application probability (default: 30%)
- Handles edge cases for non-divisible image dimensions
- Memory-efficient implementation using NumPy operations

### Comprehensive Data Augmentation Pipeline
- **Traditional augmentations**: Random crop, horizontal flip, color jittering, random erasing
- **PatchShuffle integration**: Seamlessly integrated with PyTorch transforms
- **Proper normalization**: CIFAR-10 standard normalization applied consistently

### Robust Experimental Design
- **Fair comparison**: Identical model initialization and training parameters
- **Proper validation**: 90/10 train-validation split with separate test evaluation
- **Reproducible results**: Fixed random seeds across all components
- **Comprehensive metrics**: Training/validation loss and accuracy tracking

##  Architecture

### Model: SimpleCNN
A lightweight yet effective CNN architecture:
- **3 Convolutional blocks** with BatchNorm and ReLU activation
- **MaxPooling** for spatial dimension reduction
- **Fully connected layers** with dropout for regularization
- **Total parameters**: ~1.2M (efficient for CIFAR-10)

### Data Pipeline
- **Dataset**: CIFAR-10 (50,000 training + 10,000 test images)
- **Classes**: 10 object categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Input size**: 32x32x3 RGB images
- **Batch size**: 128 with optimized data loading

## 📊 Results

| Method | Test Accuracy | Improvement |
|--------|---------------|-------------|
| Baseline (No Augmentation) | 79.14% | - |
| With Data Augmentation | 84.53% | **+5.39%** |

### Key Insights
- **PatchShuffle effectiveness**: The novel technique contributes significantly to performance gains
- **Training stability**: Augmented model shows better generalization with reduced overfitting
- **Convergence**: Both models trained for 100 epochs with early stopping on validation accuracy

## 🛠️ Technical Implementation

### Dependencies
```python
torch>=1.9.0
torchvision>=0.10.0
numpy
matplotlib
tqdm
```

### Key Components

#### PatchShuffle Algorithm
```python
def patch_shuffle(img_np, patch_size=(4, 4), probability=0.5):
    # Randomly shuffles pixels within non-overlapping patches
    # Handles non-divisible dimensions gracefully
    # Returns augmented image with same shape
```

#### PyTorch Integration
```python
class PatchShuffleTransform:
    # Seamless integration with torchvision.transforms
    # Handles tensor ↔ numpy conversion
    # Maintains gradient flow compatibility
```

##  Getting Started

### Prerequisites
- Python 3.7+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM


### Expected Output
- Training progress with real-time metrics
- Model checkpoints saved automatically
- Comprehensive comparison plots
- Performance metrics and analysis

##  Usage Examples

### Basic PatchShuffle Application
```python
from transforms import PatchShuffleTransform

# Create transform
patch_shuffle = PatchShuffleTransform(
    patch_size=(4, 4), 
    probability=0.3
)

# Apply to image tensor
augmented_image = patch_shuffle(image_tensor)
```

### Custom Augmentation Pipeline
```python
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    PatchShuffleTransform(patch_size=(4, 4), probability=0.3),
    transforms.RandomErasing(p=0.2),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])
```

##  Research Contributions

1. **Novel PatchShuffle Method**: Introduced a new data augmentation technique that preserves local structure while introducing controlled randomness
2. **Comprehensive Evaluation**: Systematic comparison of augmentation strategies on CIFAR-10
3. **Reproducible Research**: Complete experimental setup with fixed seeds and detailed documentation
4. **Performance Analysis**: Detailed metrics and visualization of training dynamics

## Project Structure

```
data_augmentation/
├── data_augmentation_CIFAR10.ipynb  # Main experiment notebook
├── README.md                             # This file
├── requirements.txt                      # Dependencies
└── data/                                # CIFAR-10 dataset (auto-downloaded)
```

##  Educational Value

This project demonstrates:
- **Deep Learning Fundamentals**: CNN architecture design and training
- **Data Augmentation Theory**: Understanding and implementing augmentation techniques
- **Experimental Design**: Proper ML experiment methodology
- **PyTorch Best Practices**: Efficient data loading, model training, and evaluation
- **Research Skills**: Novel method development and comparative analysis





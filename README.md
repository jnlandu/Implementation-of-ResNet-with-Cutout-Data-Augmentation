# ResNet ImageNet Implementation with Cutout Data Augmentation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This project implements and compares different Convolutional Neural Network architectures for ImageNet classification, with a focus on ResNet-18 and the impact of Cutout data augmentation. The implementation demonstrates the effectiveness of skip connections in deep networks and explores how data augmentation techniques can improve model generalization.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Technical Background](#technical-background)
- [Contributing](#contributing)
- [License](#license)

## Overview

ResNet, short for Residual Network, is a type of deep neural network architecture that revolutionized computer vision. This project explores the challenges faced by researchers prior to the advent of ResNet architecture in training deep neural networks with a higher number of layers. The primary obstacle was the vanishing gradient problem during the backpropagation process, which hindered the efficient updating of kernel values once the network's layers exceeded a certain threshold.

## Project Structure

```
├── ImageNet.py                    # ImageNet dataset handling utilities
├── Plain_Convnet.py              # Implementation of plain CNN without skip connections
├── resnet_18.py                  # Standard ResNet-18 implementation
├── resnet18_with_Cutout.py       # ResNet-18 with Cutout data augmentation
├── requirements.txt              # Python dependencies
├── ILSVRC2012_val_labels.json   # ImageNet validation labels
├── imagenet_class_index.json    # ImageNet class indices mapping
├── LICENSE                       # Project license
└── README.md                     # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jnlandu/Implementation-of-ResNet-with-Cutout-Data-Augmentation.git
   cd Implementation-of-ResNet-with-Cutout-Data-Augmentation
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download ImageNet dataset:**
   - Download the ImageNet dataset from [Kaggle ImageNet Object Localization Challenge](https://www.kaggle.com/c/imagenet-object-localization-challenge)
   - Extract the dataset and update the paths in the Python files accordingly

## Usage

### Training Models

1. **Train Plain ConvNet:**
   ```bash
   python Plain_Convnet.py
   ```

2. **Train ResNet-18:**
   ```bash
   python resnet_18.py
   ```

3. **Train ResNet-18 with Cutout:**
   ```bash
   python resnet18_with_Cutout.py
   ```

### Configuration

You can modify the following parameters in each script:
- `batch_size`: Training batch size (default: 32)
- `learning_rate`: Learning rate for optimization (default: 0.001)
- `num_epochs`: Number of training epochs (default: 20)
- `cutout_size`: Size of cutout patches (for Cutout implementation)

## Model Architectures

### 1. Plain ConvNet
A traditional deep convolutional network without skip connections, demonstrating the limitations of deep networks before ResNet.

### 2. ResNet-18
Implementation of the 18-layer ResNet architecture with:
- Residual blocks with skip connections
- Batch normalization
- ReLU activation functions
- Global average pooling

### 3. ResNet-18 with Cutout
Enhanced ResNet-18 with Cutout data augmentation:
- Random square patches masked during training
- Improved regularization and generalization
- Enhanced robustness to occlusions

## Technical Background

### Vanishing Gradient Problem

The vanishing gradient problem refers to the phenomenon where gradients used for updating neural network weights diminish exponentially as they are propagated back through the network. This results in very small updates to the weights of the initial layers, causing the network to learn very slowly, if at all. This problem was particularly evident in deeper networks, making it difficult to train networks effectively beyond a certain depth.

### Impact on Training Deep Neural Networks

Before the introduction of ResNet architecture, traditional neural network architectures struggled to maintain performance as the number of layers increased. As shown in the graph below, the training and testing errors were higher for a 56-layer model compared to a 20-layer model. This indicates that increasing the number of layers did not necessarily lead to better performance and, in fact, often resulted in worse performance due to the vanishing gradient problem.

### Training and Testing Error Comparison

<img width="755" alt="Screenshot 2024-06-19 at 22 17 02" src="https://github.com/Ignatiusboadi/Resnet_Imagenet_with_Cutout/assets/102676168/d188c74f-8ca2-4ff5-b202-249a6d7c50cd">

*Figure 1: Training and Testing Error Comparison Between 20-layer and 56-layer Models*

### ResNet Architecture

ResNet architecture was developed to address these issues, enabling the training of much deeper neural networks by mitigating the vanishing gradient problem. ResNet introduced various innovations that helped in maintaining effective gradient flow, thus allowing for the training of networks with significantly more layers without the degradation in performance that was previously observed.

### Importance of Skip Connections

Skip connections, also known as identity shortcuts, play a pivotal role in deep learning models like ResNet by addressing the vanishing gradient problem. By allowing gradients to flow more effectively through the network, skip connections ensure that earlier layers receive meaningful updates during training, thereby mitigating the issue of gradients diminishing to insignificance. This not only improves the training efficiency by enabling more efficient kernel learning across layers but also enhances the model's capability to extract complex features from the data. Furthermore, skip connections enable the training of deeper networks without compromising performance, leading to more sophisticated and accurate predictive models.

### Cutout Data Augmentation

Data augmentation is one solution to avoid overfitting a training dataset of images. 

Cutout augmentation is a technique used in image data augmentation where random square patches of pixels are masked out during training. This method helps in regularizing deep learning models by preventing overfitting and encouraging the learning of more robust features. By obscuring different parts of the images in each training iteration, cutout augmentation diversifies the training dataset and improves the model's ability to generalize to unseen data. It is a cost-effective and straightforward approach to enhancing the performance and robustness of image classification models.

<img width="277" alt="Screenshot 2024-06-19 at 22 41 52" src="https://github.com/Ignatiusboadi/Resnet_Imagenet_with_Cutout/assets/102676168/2b5a2b9d-6406-47fe-9a4c-74cde390e193">

*Figure 2: Example of Cutout data augmentation applied to training images*


## Results

This project compares the performance of three convolutional neural network architectures created for image classification: Plain ConvNet, ResNet-18, and ResNet-18 with Cutout. The evaluation focuses on their training dynamics, final validation metrics, and the impact of augmentation strategies on model performance.

### Performance Comparison

| Model | Initial Loss | Final Validation Loss | Final Validation Accuracy | Training Stability |
|-------|-------------|----------------------|---------------------------|-------------------|
| Plain ConvNet | 11.02 | 11% | - | Poor (plateaued after epoch 10) |
| ResNet-18 | 6.9 | 4.3 | 39% | Good (steady improvement) |
| ResNet-18 + Cutout | - | - | 46.28% (at epoch 10) | Excellent (rapid early improvement) |

### Detailed Analysis

#### 1. Plain ConvNet
- **Initial Performance**: Started with a high initial loss of 11.02
- **Training Dynamics**: Plateaued in performance after epoch 10 with a final validation loss of 11%
- **Key Insight**: Highlighted the challenges of deep networks without effective optimization strategies and demonstrated poor generalization capabilities

#### 2. ResNet-18
- **Initial Performance**: Initially exhibited instability with a high loss of 6.9 but improved steadily
- **Final Performance**: Achieved a final validation loss of 4.3 and a validation accuracy of 39%
- **Key Insight**: Demonstrated robust learning and effective classification capabilities compared to Plain ConvNet, showcasing the power of skip connections

#### 3. ResNet-18 with Cutout
- **Training Strategy**: Implemented with cutout data augmentation to enhance robustness and generalization
- **Performance**: Showed significant improvement early in training, achieving a validation accuracy of 46.28% by epoch 10
- **Key Insight**: Highlighted the effectiveness of cutout augmentation in improving model performance and showed potential for further optimization

### Key Findings

1. **Skip Connections Matter**: ResNet-18 significantly outperformed the Plain ConvNet, demonstrating the importance of residual connections in deep networks.

2. **Data Augmentation Impact**: The addition of Cutout augmentation to ResNet-18 provided a substantial boost in performance (39% → 46.28% accuracy).

3. **Training Efficiency**: ResNet architectures showed more stable and efficient training compared to traditional deep networks.

4. **Generalization**: Cutout augmentation helped the model generalize better by preventing overfitting to specific image regions.

## Experimental Setup

### Dataset
- **Training Data**: ImageNet training set (1.2M images, 1000 classes)
- **Validation Data**: ImageNet validation set (50K images)
- **Image Size**: 224×224 pixels
- **Preprocessing**: Random resizing, cropping, and normalization

### Training Configuration
- **Optimizer**: Adam optimizer
- **Learning Rate**: 0.001 (with decay)
- **Batch Size**: 32
- **Epochs**: 20
- **Hardware**: CUDA-enabled GPU
- **Cutout Parameters**: Random square patches of 16×16 pixels

### Performance Metrics
- Training Loss
- Validation Loss
- Top-1 Accuracy
- Training Time per Epoch

## Future Work

- [ ] Implement ResNet-50 and ResNet-101 variants
- [ ] Experiment with other data augmentation techniques (MixUp, CutMix)
- [ ] Add transfer learning capabilities
- [ ] Implement model pruning and quantization
- [ ] Add support for distributed training
- [ ] Comprehensive hyperparameter tuning
- [ ] Integration with MLflow for experiment tracking

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Include unit tests for new features
- Update documentation as needed

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{resnet_cutout_2024,
  title={ResNet ImageNet Implementation with Cutout Data Augmentation},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/jnlandu/Implementation-of-ResNet-with-Cutout-Data-Augmentation}}
}
```

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

2. DeVries, T., & Taylor, G. W. (2017). Improved regularization of convolutional neural networks with cutout. arXiv preprint arXiv:1708.04552.

3. Deng, J., Dong, W., Socher, R., Li, L. J., Li, K., & Fei-Fei, L. (2009). Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition (pp. 248-255).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ImageNet dataset creators and maintainers
- PyTorch development team
- Original ResNet and Cutout paper authors
- Open source community for inspiration and tools


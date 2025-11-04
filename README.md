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

## Vanishing Gradient Problem

The vanishing gradient problem refers to the phenomenon where gradients used for updating neural network weights diminish exponentially as they are propagated back through the network. This results in very small updates to the weights of the initial layers, causing the network to learn very slowly, if at all. This problem was particularly evident in deeper networks, making it difficult to train networks effectively beyond a certain depth.

## Impact on Training Deep Neural Networks

Before the introduction of ReNet architecture, traditional neural network architectures struggled to maintain performance as the number of layers increased. As shown in the graph below, the training and testing errors were higher for a 56-layer model compared to a 20-layer model. This indicates that increasing the number of layers did not necessarily lead to better performance and, in fact, often resulted in worse performance due to the vanishing gradient problem.

## Graph: Training and Testing Error

<img width="755" alt="Screenshot 2024-06-19 at 22 17 02" src="https://github.com/Ignatiusboadi/Resnet_Imagenet_with_Cutout/assets/102676168/d188c74f-8ca2-4ff5-b202-249a6d7c50cd">


*Figure 1: Training and Testing Error Comparison Between 20-layer and 56-layer Models*

## ReNet Architecture

ReNet architecture was developed to address these issues, enabling the training of much deeper neural networks by mitigating the vanishing gradient problem. ReNet introduced various innovations that helped in maintaining effective gradient flow, thus allowing for the training of networks with significantly more layers without the degradation in performance that was previously observed.
## Importance of Skip Connections

Skip connections, also known as identity shortcuts, play a pivotal role in deep learning models like ReNet and ResNet by addressing the vanishing gradient problem. By allowing gradients to flow more effectively through the network, skip connections ensure that earlier layers receive meaningful updates during training, thereby mitigating the issue of gradients diminishing to insignificance. This not only improves the training efficiency by enabling more efficient kernel learning across layers but also enhances the model's capability to extract complex features from the data. Furthermore, skip connections enable the training of deeper networks without compromising performance, leading to more sophisticated and accurate predictive models.

## cutout data augmentation
Data augmentation is one solution to avoid overfitting a training dataset of images. 

Cutout augmentation is a technique used in image data augmentation where random square patches of pixels are masked out during training. This method helps in regularizing deep learning models by preventing overfitting and encouraging the learning of more robust features. By obscuring different parts of the images in each training iteration, cutout augmentation diversifies the training dataset and improves the model's ability to generalize to unseen data. It is a cost-effective and straightforward approach to enhancing the performance and robustness of image classification models.

<img width="277" alt="Screenshot 2024-06-19 at 22 41 52" src="https://github.com/Ignatiusboadi/Resnet_Imagenet_with_Cutout/assets/102676168/2b5a2b9d-6406-47fe-9a4c-74cde390e193">


## Result 
in this poject we compares the performance of three convolutional neural network architectures created for image classification: Plain ConvNet, ResNet-18, and ResNet-18 with Cutout. The evaluation focuses on their training dynamics, final validation metrics, and the impact of augmentation strategies on model performance.

## Models Created

### 1. Plain ConvNet

- Started with a high initial loss of 11.02.
- Plateaued in performance after epoch 10 with a final validation loss of 11%, indicating poor generalization.
- Highlighted the challenges of deep networks without effective optimization strategies.

### 2. ResNet-18

- Initially exhibited instability with a high loss of 6.9 but improved steadily.
- Achieved a final validation loss of 4.3 and a validation accuracy of 39%.
- Demonstrated robust learning and effective classification capabilities compared to Plain ConvNet.

### 3. ResNet-18 with Cutout

- Implemented with cutout data augmentation to enhance robustness and generalization.
- Showed significant improvement early in training, achieving a validation accuracy of 46.28% by epoch 10.
- Highlighted the effectiveness of cutout augmentation in improving model performance and potential for further optimization.


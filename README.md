# tinyCNN

A pure NumPy implementation of a Convolutional Neural Network (CNN) built from scratch for educational purposes. This project demonstrates the fundamental components of deep learning without relying on modern frameworks like TensorFlow or PyTorch.

## Overview

This implementation provides a complete CNN framework including forward and backward propagation, various layer types, activation functions, loss functions, and optimization algorithms. All components are implemented using only NumPy, offering full transparency into how neural networks operate under the hood.

## Features

### Core Layers

- **Conv2D**: Standard 2D convolutional layer with configurable filters, kernel size, stride, and padding
- **Conv2DOptimized**: Memory-efficient implementation using image-to-column transformation (approximately 35% faster than standard Conv2D)
- **MaxPool2D**: Max pooling layer for spatial downsampling
- **Dense**: Fully connected layer
- **Flatten**: Reshapes multi-dimensional inputs into 1D vectors
- **Dropout**: Regularization layer to prevent overfitting
- **BatchNorm**: Batch normalization for training stability

### Activation Functions

- ReLU (Rectified Linear Unit)
- Sigmoid
- Softmax
- Tanh

### Loss Functions

- Categorical Cross-Entropy (with integrated Softmax)
- Binary Cross-Entropy

### Optimizers

- SGD (with momentum)
- Adagrad
- RMSprop
- Adam

### Model Architectures

- **TinyCNN**: Lightweight architecture with 2 convolutional layers
- **SimpleCNN**: Deeper architecture with 6 convolutional layers

## Project Structure

```
tinyCNN/
├── nn/
│   ├── activations.py     # Activation function implementations
│   ├── layers.py          # Neural network layer implementations
│   ├── losses.py          # Loss function implementations
│   └── optimizers.py      # Optimization algorithms
├── vision/
│   ├── models.py          # Pre-built CNN architectures
│   └── training.py        # Training and evaluation utilities
├── utils/
│   ├── data.py            # Data generation and preprocessing
│   ├── model_io.py        # Model saving/loading utilities
│   └── optimization.py    # Optimization utilities
├── tests/
│   ├── test_implementation.py    # Unit tests for all components
│   ├── test_synthetic_data.py    # Training tests on synthetic data
│   └── test_optimized.py         # Performance comparison tests
└── results/               # Training results and visualizations
```

## Installation

This project requires only NumPy and Matplotlib for visualization:

```bash
pip install numpy matplotlib
```

## Usage

### Running Tests (in REPL)

Add the following at the top of the files before running:

```bash
import sys
sys.path.append('..')
```

### Building a Custom Model

```python
from nn.layers import Conv2D, MaxPool2D, Flatten, Dense
from nn.activations import ReLU
from nn.losses import CategoricalCrossEntropyLoss
from nn.optimizers import Adam

# Define architecture
input_shape = (3, 32, 32)  # (channels, height, width)
n_classes = 10

# Build layers
conv1 = Conv2D(n_filters=32, kernel_size=3, n_channels=3, stride=1, padding=1)
relu1 = ReLU()
pool1 = MaxPool2D(pool_size=2, stride=2)
flatten = Flatten()
fc1 = Dense(32 * 16 * 16, 128)
relu2 = ReLU()
fc2 = Dense(128, n_classes)

# Setup training
criterion = CategoricalCrossEntropyLoss()
optimizer = Adam(learning_rate=0.001)

# Forward pass
x = conv1.forward(inputs)
x = relu1.forward(x)
# ... continue with other layers
```

### Using Pre-built Models

```python
from vision.models import TinyCNN
from vision.training import train_cnn

# Create model
model = TinyCNN(input_shape=(3, 32, 32), n_classes=10)

# Train model
history = train_cnn(
    cnn=model,
    optimizer=optimizer,
    criterion=criterion,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    epochs=20,
    batch_size=32
)
```

## Performance Optimizations

### Optimized Convolution Layer

The `Conv2DOptimized` class implements the im2col (image-to-column) algorithm, which transforms the 4D convolution operation into a 2D matrix multiplication. This optimization provides approximately **35% performance improvement** over the naive nested-loop implementation in `Conv2D`.

To use the optimized layer in your models:

```python
from utils.optimization import optimize_conv_layers

model = TinyCNN(input_shape=(3, 32, 32), n_classes=10)
optimize_conv_layers(model)  # Replaces Conv2D with Conv2DOptimized
```

### Weight Initialization

All layers use He initialization (for ReLU activations), which helps prevent vanishing/exploding gradients during training.

## Important Note on Training Efficiency

**This implementation is not optimized for production use.** The code prioritizes clarity and educational value over computational efficiency. As a result:

- Training on real datasets (MNIST, CIFAR-10, etc.) would take an extremely long time
- The implementation is significantly slower than framework-based alternatives
- Memory usage is not optimized for large-scale datasets

For this reason, **the included tests use only synthetic data** rather than real-world datasets. The synthetic data tests demonstrate that the implementation is correct and all gradients flow properly, but practical training on standard benchmarks is not feasible with this codebase.

## Testing

The project includes comprehensive unit tests covering:

- Forward and backward passes for all layer types
- Activation function correctness
- Loss function computation and gradients
- Optimizer parameter updates
- End-to-end training on synthetic data

All tests verify:

- Output shapes match expected dimensions
- Gradients are computed correctly
- Parameters are updated appropriately
- Training loss decreases over epochs

## Implementation Details

### Backpropagation

Each layer implements both `forward()` and `backward()` methods. The backward pass computes:

- `dinputs`: Gradient with respect to inputs (passed to previous layer)
- `dW`: Gradient with respect to weights (used by optimizer)
- `db`: Gradient with respect to biases (used by optimizer)

### Numerical Stability

The implementation includes several numerical stability measures:

- Clipping in Softmax to prevent overflow
- Epsilon terms in optimizers to prevent division by zero
- Proper gradient scaling in batch normalization

### Memory Management

While functional, the implementation stores intermediate values that could be optimized:

- Full activation maps are stored during forward pass
- Gradients are accumulated in full precision
- No gradient checkpointing or memory pooling

## Limitations

- No GPU acceleration
- Single-threaded execution
- No automatic differentiation
- Limited to 2D convolutions
- No pre-trained weights
- No data augmentation pipeline
- No learning rate scheduling
- No early stopping mechanisms

## Educational Value

This project is ideal for:

- Understanding backpropagation in CNNs
- Learning how convolution operations work
- Exploring optimization algorithms
- Debugging neural network training issues
- Teaching deep learning fundamentals

## Acknowledgments

This implementation draws inspiration from neural network fundamentals taught in deep learning courses and textbooks, implemented purely in NumPy to maximize transparency and understanding.

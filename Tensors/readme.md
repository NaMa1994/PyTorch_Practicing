# PyTorch Tensors

![PyTorch Logo](https://upload.wikimedia.org/wikipedia/commons/9/96/Pytorch_logo.png)

A beginner-friendly guide to understanding and working with tensors in PyTorch, covering creation, basic operations, and common tensor manipulations.

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Getting Started with Tensors](#getting-started-with-tensors)
4. [Examples](#examples)
5. [Contributing](#contributing)
6. [License](#license)

## Overview
Tensors are a fundamental data structure in PyTorch and are essential for deep learning and machine learning tasks. This guide walks you through the basics of tensor operations in PyTorch, including:
- Creating and manipulating tensors
- Tensor math operations
- Moving tensors between devices (CPU/GPU)
- Tensors are similar to numpy arrays but more powerfull
- They automatically calculate gradients

## Installation

To work with PyTorch tensors, you need to have PyTorch installed. You can install PyTorch via pip as follows:

```bash
# For CPU-only support
pip install torch

# For GPU support (CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu117


import torch

# Create a tensor
x = torch.tensor([[1, 2], [3, 4]])
print("Tensor x:\n", x)

# Basic tensor operations
y = x + 2
print("Tensor y (x + 2):\n", y)

# Moving tensor to GPU (if available)
if torch.cuda.is_available():
    x = x.to('cuda')
    print("Tensor x on GPU:\n", x)
```
## Getting Started with Tensors

Here's a quick example to create and manipulate a tensor in PyTorch:

import torch

# Create a tensor
```bash
x = torch.tensor([[1, 2], [3, 4]])
print("Tensor x:\n", x)
```

# Basic tensor operations
```bash
y = x + 2
print("Tensor y (x + 2):\n", y)

# Moving tensor to GPU (if available)
if torch.cuda.is_available():
    x = x.to('cuda')
    print("Tensor x on GPU:\n", x)
```
## Examples

### 1. Creating Tensors
Here are a few ways to create tensors in PyTorch:
```bash
# Creating a tensor filled with ones
ones_tensor = torch.ones((3, 3))
print("Ones Tensor:\n", ones_tensor)
```
```bash
# Creating a tensor with random values
random_tensor = torch.rand((2, 2))
print("Random Tensor:\n", random_tensor)
```

### 2. Tensor Operations
You can perform various operations on tensors, such as addition, multiplication, and reshaping:

# Element-wise addition
```bash
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = a + b
print("Sum of a and b:\n", c)
```
# Reshaping tensors
```bash
reshaped_tensor = c.view(3, 1)
print("Reshaped Tensor:\n", reshaped_tensor)
```
### 3. Visualization (Optional)
If you'd like to visualize tensors as images, use matplotlib:
```bash
import matplotlib.pyplot as plt
```
# Convert tensor to numpy and visualize
```bash
img_tensor = torch.rand((3, 3))
plt.imshow(img_tensor.numpy(), cmap='viridis')
plt.title("Random Tensor Visualization")
plt.show()
```


## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs, improvements, or suggestions.

1- Fork the repository
2- Create a new branch (git checkout -b feature-branch)
3- Make your changes and commit (git commit -m 'Add new feature')
4- Push to the branch (git push origin feature-branch)
5- Open a pull request
## License

This project is licensed under the MIT License. See the LICENSE file for details.









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

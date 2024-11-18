# Hyperparameter Tuning in PyTorch Model Training

Hyperparameter tuning is critical for optimizing model performance, stability, and training efficiency. By systematically exploring different combinations of parameters, you can find the best settings for your model. This guide covers key hyperparameters, their impact, and the tools used for tuning, with a focus on `skorch`.

---

## Why Hyperparameter Tuning?

- **Optimize Training/Inference Time**: Balance computational efficiency with performance.
- **Improve Model Results**: Achieve better accuracy, precision, or other evaluation metrics.
- **Ensure Model Convergence**: Prevent underfitting or overfitting by fine-tuning parameters.

---

## Key Hyperparameters and Their Impact

### 1. **Batch Size**
Batch size affects how many samples the model processes before updating its parameters.

| **Batch Size** | **GPU Memory Usage** | **Training Stability** | **Training Time**   | **Inference Time** |
|----------------|-----------------------|-------------------------|---------------------|---------------------|
| Small (e.g., 16) | Low                   | High (frequent updates) | Longer (more batches) | No effect           |
| Medium (e.g., 64) | Moderate              | Balanced                | Balanced             | No effect           |
| Large (e.g., 256) | High                  | Lower stability         | Faster (fewer batches) | No effect           |

**Recommendation**: Start with a medium batch size (e.g., 64) and adjust based on GPU memory and training dynamics.

---

### 2. **Number of Epochs**
Defines how many times the model will iterate over the entire training dataset.

| **Epochs**   | **Training Time**       | **Inference Time** | **Stability**          | **Performance**       |
|--------------|-------------------------|---------------------|------------------------|-----------------------|
| Few (e.g., 10)  | Fast                    | No effect           | Risk of underfitting   | Suboptimal            |
| Moderate (e.g., 50) | Balanced               | No effect           | Good                   | Likely optimal        |
| Many (e.g., 100+) | Slow                    | No effect           | Risk of overfitting    | May improve slightly  |

**Recommendation**: Use early stopping criteria to avoid unnecessary training after performance plateaus.

---

### 3. **Number of Hidden Layers**
More hidden layers allow the model to learn complex patterns but increase computational cost.

| **Layers**   | **Training Time**       | **Inference Time** | **Model Complexity**   | **Risk of Overfitting** |
|--------------|-------------------------|---------------------|------------------------|--------------------------|
| Few (e.g., 1-2) | Fast                    | Fast                | Low                    | Low                      |
| Moderate (e.g., 3-5) | Balanced               | Moderate            | Balanced               | Moderate                 |
| Many (e.g., 6+)   | Slow                    | Slow                | High                   | High                     |

**Recommendation**: Choose the smallest number of layers that achieves acceptable performance.

---

### 4. **Number of Nodes per Layer**
Defines the capacity of each hidden layer.

| **Nodes**    | **Training Time**       | **Inference Time** | **Model Capacity**     | **Risk of Overfitting** |
|--------------|-------------------------|---------------------|------------------------|--------------------------|
| Few (e.g., 16) | Fast                    | Fast                | Low                    | Low                      |
| Moderate (e.g., 64) | Balanced               | Balanced            | Balanced               | Moderate                 |
| Many (e.g., 256+) | Slow                    | Slow                | High                   | High                     |

**Recommendation**: Start with moderate nodes and tune based on the dataset's complexity.

---

## Methods for Structured Hyperparameter Search

1. **Grid Search**:
   - Exhaustively explores all combinations of hyperparameters.
   - Guarantees finding the best combination but is computationally expensive.

2. **Random Search**:
   - Samples random combinations of hyperparameters.
   - Faster than grid search and often finds near-optimal parameters.

---

## Automated Hyperparameter Tuning with Skorch

`skorch` is a PyTorch-compatible package that simplifies model training and hyperparameter tuning. It integrates well with tools like `GridSearchCV` and `RandomizedSearchCV` from `scikit-learn`.

### Example:
```python
from skorch import NeuralNetRegressor
from sklearn.model_selection import RandomizedSearchCV

net = NeuralNetRegressor(
    model,
    max_epochs=50,
    lr=0.01,
    batch_size=32,
)

params = {
    'lr': [0.01, 0.001, 0.0001],
    'batch_size': [16, 32, 64],
    'max_epochs': [10, 50, 100],
}

search = RandomizedSearchCV(net, params, n_iter=10, scoring='r2')
search.fit(X_train, y_train)
```

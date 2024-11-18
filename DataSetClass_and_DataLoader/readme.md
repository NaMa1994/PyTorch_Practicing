# Linear Regression with PyTorch: Dataset and DataLoader

This repository demonstrates how to efficiently implement and train a linear regression model in PyTorch using a custom `Dataset` class and `DataLoader`. These tools simplify data handling, especially for large datasets or when using mini-batch training.

## Why Use a Custom Dataset Class?

Manually slicing the dataset and iterating through it can lead to verbose and error-prone code. By defining a `Dataset` class, we can:
- Encapsulate the data and its behavior (e.g., how to access samples).
- Use PyTorch's `DataLoader` for automatic batching, shuffling, and parallel data loading.

### Implementation of `LinearRegressionDataset`

This class wraps the input features (`X`) and target values (`y`):

```python
from torch.utils.data import Dataset

class LinearRegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```
The Dataset class provides:

__len__: Returns the number of samples.

__getitem__: Retrieves a sample (feature and target pair) by index.

### Using DataLoader
The DataLoader simplifies training by handling batching and data iteration:
```python
from torch.utils.data import DataLoader

train_loader = DataLoader(dataset=LinearRegressionDataset(X_np, y_np), batch_size=2)
```

### Efficient Training Loop with DataLoader

Instead of manually managing batches, we use the DataLoader for cleaner and more efficient code:
```python
losses = []
slope, bias = []
number_epochs = 1000

for epoch in range(number_epochs):
    for j, (X, y) in enumerate(train_loader):
        # Optimization step
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(X)

        # Compute loss
        loss = loss_fun(y_pred, y)
        losses.append(loss.item())

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

    # Retrieve model parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name == 'linear.weight':
                slope.append(param.data.numpy()[0][0])
            if name == 'linear.bias':
                bias.append(param.data.numpy()[0])

    # Log progress
    if (epoch % 100 == 0):
        print(f"Epoch {epoch}, Loss: {loss.data}")

```

### Why Avoid the Manual Batch Loop?
Manual batch handling like this:
```python
for i in range(0, X.shape[0], BATCH_SIZE):
    # Slice data manually
    y_pred = model(X[i:i+BATCH_SIZE])
    loss = loss_fun(y_pred, y_true[i:i+BATCH_SIZE])

```

is:

Verbose and less readable.
Prone to errors when handling edge cases (e.g., last batch).
Lacks the efficiency and flexibility of DataLoader.
By leveraging DataLoader, we:

Simplify the code.
Automatically handle batching, shuffling, and data loading in an optimized manner.

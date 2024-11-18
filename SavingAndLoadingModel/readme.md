# Training a Model in PyTorch

This guide explains the essential components of training a model in PyTorch, including the model, optimizer, loss function, gradient updates, and related concepts. The concepts are drawn from practical experience and insights gained from [this useful article by Jean Cochrane](https://jeancochrane.com/blog/pytorch-functional-api).

---

## Key Components of PyTorch Training

1. **Model**:
   - Represents the architecture of your neural network, implemented as a subclass of `torch.nn.Module`.
   - Example:
     ```python
     class LinearRegressionTorch(nn.Module):
         def __init__(self, input_size, output_size):
             super(LinearRegressionTorch, self).__init__()
             self.linear = nn.Linear(input_size, output_size)
         
         def forward(self, x):
             return self.linear(x)
     
     model = LinearRegressionTorch(input_size=1, output_size=1)
     ```

2. **Optimizer**:
   - Updates the model's parameters based on gradients computed during backpropagation.
   - Common optimizers: `torch.optim.SGD`, `torch.optim.Adam`.
   - Example:
     ```python
     optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
     ```

3. **Loss Function**:
   - Measures the difference between the model's predictions and the true labels.
   - Common loss functions: `torch.nn.MSELoss`, `torch.nn.CrossEntropyLoss`.
   - Example:
     ```python
     loss_fun = nn.MSELoss()
     ```

4. **Gradient Update**:
   - PyTorch computes gradients of the loss with respect to model parameters using `loss.backward()`.

5. **Weight Update**:
   - The optimizer updates the model's parameters using `optimizer.step()`.

---

## PyTorch Training Loop

A standard PyTorch training loop follows these steps:

1. **Forward Pass**:
   - Pass inputs through the model to compute predictions.
   - Example:
     ```python
     y_pred = model(X)
     ```

2. **Compute Loss**:
   - Calculate the loss using the predictions and ground truth.
   - Example:
     ```python
     loss = loss_fun(y_pred, y_true)
     ```

3. **Backpropagation**:
   - Compute gradients using `loss.backward()`.
   - Example:
     ```python
     loss.backward()
     ```

4. **Update Weights**:
   - Use the optimizer to update the model's parameters.
   - Example:
     ```python
     optimizer.step()
     ```

5. **Zero Gradients**:
   - Clear previous gradients to prevent accumulation.
   - Example:
     ```python
     optimizer.zero_grad()
     ```

### Complete Training Loop Example

```python
losses = []
number_epochs = 1000

for epoch in range(number_epochs):
    # Forward pass
    y_pred = model(X)

    # Compute loss
    loss = loss_fun(y_pred, y_true)
    losses.append(loss.item())

    # Backpropagation
    optimizer.zero_grad()  # Clear old gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update weights

    # Print progress every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
```



# Saving and Loading Models in PyTorch

This guide explains how to save and load models efficiently in PyTorch. It focuses on saving only the model's state dictionary (parameters and buffers) rather than the entire model, highlighting the reasons for this best practice.

---

## Key Functions for Model Saving and Loading

1. **`torch.save()`**:
   - Saves the model's state dictionary, which is a Python dictionary containing:
     - Trainable parameters (weights and biases).
     - Registered buffers (e.g., batch normalization running averages).
   - Example:
     ```python
     torch.save(model.state_dict(), "model_state.pth")
     ```

2. **`torch.load_state_dict()`**:
   - Loads a state dictionary into a model instance.
   - Example:
     ```python
     model.load_state_dict(torch.load("model_state.pth"))
     ```

3. **`torch.load()`**:
   - Loads serialized data from a file. Used with `torch.save()` to reload the state dictionary.
   - Example:
     ```python
     state_dict = torch.load("model_state.pth")
     ```

---

## Why Not Save the Complete Model?

Instead of saving the entire model, it's recommended to save only the state dictionary. Here's why:

### 1. **Flexibility**:
   - Saving the entire model stores its class definition, structure, and parameters.
   - Reloading the model requires the exact same codebase and environment where it was saved, leading to compatibility issues if the class definition changes.

### 2. **Portability**:
   - State dictionaries are more portable since they only contain the parameters.
   - The model architecture can be reconstructed independently, and the parameters can be loaded afterward.

### 3. **Efficiency**:
   - Saves only the learnable parameters and buffers, which are essential for model functionality.
   - Reduces storage size compared to saving the full model.

---

## Best Practice for Saving and Loading

### Saving a Model:
1. Save only the state dictionary:
   ```python
   torch.save(model.state_dict(), "model_state.pth")
   ```

2. Loading a Model:
   ```pyhton
   model = LinearRegressionTorch(input_size=1, output_size=1)
   model.load_state_dict(torch.load("model_state.pth"))
   model.eval()

   
   ```

## When to Save the Entire Model?

Saving the entire model using torch.save(model, "model.pth") can be helpful in quick prototyping or debugging, but it's generally discouraged in production or long-term projects because:

It tightly couples the saved file with the exact codebase and environment.
It increases storage size unnecessarily.
By saving only the state dictionary, you gain better flexibility, portability, and maintainability.

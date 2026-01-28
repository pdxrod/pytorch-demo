#!/usr/bin/env python

print( """This is a simple example of a PyTorch model. 
It creates a model with 3 inputs and 4 neurons in the first 
layer and 4 inputs and 2 neurons in the second layer.
It then creates an optimizer and a forward pass.
It then computes the loss and performs a backward pass.
Then it updates the parameters.
""" )
input( f"Press Enter to continue..." )  

import torch
from torch import nn

# Create a simple model
model = nn.Sequential(
    nn.Linear(3, 4),  # 3 inputs → 4 neurons
    nn.Linear(4, 2)   # 4 inputs → 2 neurons
)

print("Model structure:")
for name, param in model.named_parameters():
    print(f"{name}: shape {param.shape}, size {param.numel()}")

# Create optimizer (stores references to ALL parameter tensors)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Forward pass
X = torch.tensor([[1.0, 2.0, 3.0]])
y = torch.tensor([[0.0, 1.0]])
y_pred = model(X)

# Compute loss
loss = nn.MSELoss()(y_pred, y)

# Backward pass (computes gradients for ALL parameters)
loss.backward()

print("\nGradients computed:")
for name, param in model.named_parameters():
    print(f"{name:20} | Shape: {str(param.shape):15} | Size: {param.numel()}")
    if param.grad is not None:
        print(f"{name}.grad: shape {param.grad.shape}")

# Optimizer step (updates ALL parameters)
optimizer.step()

print("\nParameters updated (all values changed)")

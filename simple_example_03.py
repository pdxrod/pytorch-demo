"""
Demonstrates how weights and biases change during training.
  - Initial weights and biases are RANDOM
  - Training updates them to minimize loss
"""
import torch
from torch import nn

# Set seed for reproducible random initialization
torch.manual_seed(42)

# Create a simple model
# Weights and biases are initialized RANDOMLY by PyTorch
model = nn.Sequential(
    nn.Linear(3, 4),  # 3 inputs → 4 neurons
    nn.Linear(4, 2)   # 4 inputs → 2 neurons
)

print("=" * 70)
print("BEFORE TRAINING - Random initial weights and biases:")
print("=" * 70)
initial_params = {}
for name, param in model.named_parameters():
    initial_params[name] = param.data.clone()  # Save a copy
    print(f"\n{name}:")
    print(f"  Shape: {param.shape}")
    print(f"  Values:\n{param.data}")

# Create optimizer (stores references to ALL parameter tensors)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Training data
X = torch.tensor([[1.0, 2.0, 3.0]])
y = torch.tensor([[0.0, 1.0]])

# Train for multiple epochs to see clear changes
epochs = 5
print("\n" + "=" * 70)
print(f"TRAINING for {epochs} epochs...")
print("=" * 70)

for epoch in range(epochs):
    # Forward pass
    y_pred = model(X)
    
    # Compute loss
    loss = nn.MSELoss()(y_pred, y)
    
    # Backward pass (computes gradients)
    optimizer.zero_grad()
    loss.backward()
    
    # Update weights and biases
    optimizer.step()
    
    print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")

print("\n" + "=" * 70)
print("AFTER TRAINING - Updated weights and biases:")
print("=" * 70)
for name, param in model.named_parameters():
    print(f"\n{name}:")
    print(f"  Shape: {param.shape}")
    print(f"  Values:\n{param.data}")

print("\n" + "=" * 70)
print("CHANGES (After - Before):")
print("=" * 70)
for name in initial_params:
    change = model.state_dict()[name] - initial_params[name]
    print(f"\n{name}:")
    print(f"  Change:\n{change}")
    print(f"  Max change: {change.abs().max().item():.6f}")
    print(f"  Mean change: {change.mean().item():.6f}")

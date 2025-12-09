#!/usr/bin/env python

"""
Get optimizer function
for N epochs:
    train step:
      model.train()
      get loss function
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    test step:
      model.eval()
      test_loss = loss_fn(...)
evaluate test loss            
"""
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
sys.path.append(".")
import my_utils
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

EPOCHS = 1000
SAMPLES = 1000
NOISE = 0.03
RANDOM_SEED = 42
TEST_SIZE = 0.2
LEARNING_RATE = 0.1

class ClassifierModel1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=10)
        self.layer2 = nn.Linear(in_features=10, out_features=10)
        self.layer3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        return self.layer3( self.layer2( self.layer1(x)))

class ClassifierModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2, out_features=10)
        self.relu1 = nn.ReLU()  # ← Add this!
        self.layer2 = nn.Linear(in_features=10, out_features=10)
        self.relu2 = nn.ReLU()  # ← Add this!
        self.layer3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)      # ← Add this!
        x = self.layer2(x)
        x = self.relu2(x)      # ← Add this!
        x = self.layer3(x)
        return x        

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """
    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.legend(prop={"size": 14})

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() 
    acc = (correct / len(y_pred)) * 100 
    return acc

model_1 = ClassifierModel1().to("mps")
model_2 = ClassifierModel2().to("mps")
models = [model_1, model_2]
my_utils.wait_for_user_input("Two models, 1 and 2, for drawing lines between 'moon' shapes.\n \
 The second model uses the ReLU function, and gets the lines more-or-less right.")
print(model_1)
print(model_2)

# Get X and y for 'moons' data 
# X = input features (coordinates of each point)
# y = target labels (which class each point belongs to: 0 or 1)
X, y = make_moons(n_samples=SAMPLES, noise=NOISE, random_state=RANDOM_SEED)

loss_fn = nn.BCEWithLogitsLoss()

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=TEST_SIZE, 
                                                     random_state=RANDOM_SEED )
X_train_mps, X_test_mps = X_train.to("mps"), X_test.to("mps")
y_train_mps, y_test_mps = y_train.to("mps"), y_test.to("mps")
y_train_float = y_train_mps.float()
y_train_int = y_train_mps.long()
y_test_float = y_test_mps.float()
y_test_int = y_test_mps.long()
print(f"\nX_train: {X_train_mps.squeeze()[:5]} \nX_test: {X_test_mps.squeeze()[:5]} \
      \ny_train: {y_train_mps.squeeze()[:5]} \ny_test: {y_test_mps.squeeze()[:5]} ")

my_utils.wait_for_user_input("Training loop")
n = 1
# 1. Forward pass:     X_train → model → y_logits (predictions)
# 2. Calculate loss:   Compare y_logits to y_train → loss (how wrong?)
# 3. Backward pass:    loss.backward() → gradients (which way to adjust?)
# 4. Update weights:   optimizer.step() → new weights (actually adjust)
# 5. Repeat:           Do this 1000 times → model learns!
# 6. Test:             Check on unseen data → see if it generalizes
#   loss & acc - How well the model performs on data it’s learning from.
#   test_loss & test_acc - How well the model generalizes to unseen data.
for model in models:
    print(f"model_{n}")
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        model.train()
        y_logits = model(X_train_mps)
        loss = loss_fn(y_logits.squeeze(), y_train_mps.squeeze())
        y_pred = torch.round( torch.sigmoid(y_logits) ).squeeze()
        acc = accuracy_fn(y_true=y_train_int, y_pred=y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            test_logits = model(X_test_mps)
            test_pred = torch.round( torch.sigmoid(test_logits) ).squeeze()
            test_loss = loss_fn(test_logits.squeeze(), y_test_mps.squeeze())
            test_acc = accuracy_fn(y_true=y_test_int, y_pred=test_pred)
            if (epoch < 100 and epoch % 20 == 0) or (epoch > 100 and epoch % 100 == 0):
                print(f"Epoch {epoch:04d} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.2f}% | loss: {loss:.4f} | acc: {acc:.2f}%")
    print(f"Final for model_{n}: Test loss: {test_loss:.4f} | Test acc: {test_acc:.2f}%")
    my_utils.wait_for_user_input(f"Graph of model_{n} decision boundaries")
    print("")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title(f"Test Data Decision Boundary for model_{n}")
    plot_decision_boundary(model, X_test, y_test)
    plt.tight_layout()
    plt.show()
    n += 1

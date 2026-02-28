#!/usr/bin/env python

import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt
import sys
import my_utils 

epochs = 300
weight = 0.3
bias = 0.9
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias
 
my_utils.wait_for_user_input(
  """
  This simple PyTorch AI/ML model illustrates the addition of the 'extra' functions 
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
  to each training step.         
  The positive effect of this is shown in graphs of the training and testing loss curves. 
  """
)
 
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
predictions = None

class MyFirstModel(nn.Module): 
    def __init__(self):
        super().__init__() 
        self.weights = nn.Parameter(torch.randn(1,  
            dtype=torch.float), requires_grad=True) 
        self.bias = nn.Parameter(torch.randn(1, 
            dtype=torch.float), requires_grad=True) 

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.weights * x + self.bias 

def train_step(model: torch.nn.Module, use_extra_functions: bool,
               loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer):                  
    train_loss_values = []
    test_loss_values = []
    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train)
        test_pred = model(X_test)
        loss = loss_fn(y_pred, y_train)
        if( use_extra_functions ):
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

        test_loss = loss_fn(test_pred, y_test.type(torch.float))
        train_loss_values.append(loss.detach().numpy())
        test_loss_values.append(test_loss.detach().numpy())

        model.eval()
        with torch.inference_mode():
            y_preds_new = model(X_test)
            test_loss = loss_fn(y_preds_new, y_test.type(torch.float))
            if( epoch > 0 and epoch % 50 == 0 ):
                print(f"Epoch: {epoch} | Test Loss: {test_loss}")
    return train_loss_values, test_loss_values

extra_functions = [False, True]
for use_extra_functions in extra_functions:
    model_0 = MyFirstModel()
    loss_fn = nn.L1Loss() 
    optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)                 
    train_loss_values, test_loss_values = train_step( 
        model_0, 
        use_extra_functions=use_extra_functions, 
        loss_fn=loss_fn, 
        optimizer=optimizer)

    test_epochs = list(range(len(test_loss_values)))
    train_epochs = list(range(len(train_loss_values)))

    msg = "with 'extra' functions" if use_extra_functions else "without 'extra' functions"
    print("After running train_step() ", msg)
    my_utils.wait_for_user_input()

    plt.figure(figsize=(9, 6))
    plt.title("Training and Testing Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.scatter(train_epochs, train_loss_values, c="b", s=4, label="Training loss")
    plt.scatter(test_epochs, test_loss_values, c="g", s=4, label="Testing loss")
    plt.legend()
    plt.show()

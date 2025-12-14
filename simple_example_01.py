#!/usr/bin/env python

import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import my_utils 

weight = 0.3
bias = 0.9
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias
print( "X[:10], y[:10]: ", X[:10], y[:10] ) 
my_utils.wait_for_user_input()
 
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
    print( "train_step() - use_extra_functions?: ", use_extra_functions )
    my_utils.wait_for_user_input()
                  
    train_loss_values = []
    test_loss_values = []
    epochs = 300
    for epoch in range(epochs):
        model_0.train()
        y_pred = model_0(X_train)
        test_pred = model_0(X_test)
        loss = loss_fn(y_pred, y_train)
        if( use_extra_functions ):
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

        test_loss = loss_fn(test_pred, y_test.type(torch.float)) 
        train_loss_values.append(loss.detach().numpy())
        test_loss_values.append(test_loss.detach().numpy())

        model_0.eval()
        with torch.inference_mode():
            y_preds_new = model_0(X_test)
            test_loss = loss_fn(y_preds_new, y_test.type(torch.float))
            if( epoch > 0 and epoch % 50 == 0 ):
                print(f"Epoch: {epoch} | Test Loss: {test_loss}")
    return train_loss_values, test_loss_values

model_0 = MyFirstModel()
loss_fn = nn.L1Loss() 
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)   
test_loss_values = []
train_loss_values, test_loss_values = train_step( model_0, use_extra_functions=False, 
            loss_fn=loss_fn, optimizer=optimizer)

model_0 = MyFirstModel()
loss_fn = nn.L1Loss() 
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)                 
train_loss_values = []
test_loss_values = []
train_loss_values, test_loss_values = train_step( model_0, use_extra_functions=True, 
            loss_fn=loss_fn, optimizer=optimizer)

test_epochs = list(range(len(test_loss_values)))
train_epochs = list(range(len(train_loss_values)))

plt.figure(figsize=(9, 6))
plt.title("Training and Testing Loss Curves")
plt.xlabel("Epochs")
plt.ylabel("Loss")

print("After running train_step() with optimizer and loss functions")
print("X_train length: ", len(X_train))
print("train_loss_values length: ", len(train_loss_values))
my_utils.wait_for_user_input()

plt.scatter(train_epochs, train_loss_values, c="b", s=4, label="Training loss")
plt.scatter(test_epochs, test_loss_values, c="g", s=4, label="Testing loss")
plt.legend()
plt.show()

print("Saving to file")
old_state_dict = model_0.state_dict()
torch.save(obj=model_0.state_dict(), f="01_workflow_model_0.pth") 
print("Loading from file")
loaded_model_0 = MyFirstModel()
loaded_model_0.load_state_dict(torch.load(f="01_workflow_model_0.pth"))
new_state_dict = loaded_model_0.state_dict()
print("Saved state dict: ", old_state_dict)
print("Loaded state dict: ", new_state_dict)
print("Loaded state dict == saved state dict?: ", new_state_dict == old_state_dict)

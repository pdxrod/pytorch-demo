import os
import sys
import torch
from torch import nn
import itertools
import matplotlib.pyplot as plt

def wait_for_user_input(msg=None):
    print("")
    if msg is not None:
        print(msg)
    input( f"Press Enter to continue..." )  

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps" 
    else:
        return "cpu" 

def sigmoid(x):
  return 1 / (1 + torch.exp(-x))

def ReLU(x):
  return torch.maximum(torch.tensor(0), x) 

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer):
    """
    Trains a PyTorch model for one epoch.
    
    Args:
        model: PyTorch model to train
        train_dataloader: DataLoader for training data
        loss_fn: Loss function
        optimizer: Optimizer for updating model parameters
    
    Returns:
        float: Average training loss for the epoch
    """
    train_loss = 0
    for batch, (X, y) in enumerate(train_dataloader):
        model.train() 
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(train_dataloader)
    return train_loss

def test(model: torch.nn.Module,
         test_dataloader: torch.utils.data.DataLoader,
         loss_fn: torch.nn.Module,
         accuracy_fn):
    """
    Tests a PyTorch model for one epoch.
    
    Args:
        model: PyTorch model to test
        test_dataloader: DataLoader for test data
        loss_fn: Loss function
        accuracy_fn: Function to calculate accuracy
    
    Returns:
        tuple: (test_loss, test_acc) - Average test loss and accuracy for the epoch
    """
    test_loss, test_acc = 0, 0 
    model.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            test_pred = model(X)
            test_loss += loss_fn(test_pred, y) 
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
    return test_loss, test_acc

def test_train_loop(model: torch.nn.Module,
                    train_dataloader: torch.utils.data.DataLoader,
                    test_dataloader: torch.utils.data.DataLoader,
                    loss_fn: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    accuracy_fn,
                    epochs: int):
    """
    Training and testing loop for a PyTorch model.
    
    Args:
        model: PyTorch model to train
        train_dataloader: DataLoader for training data
        test_dataloader: DataLoader for test data
        loss_fn: Loss function
        optimizer: Optimizer for updating model parameters
        accuracy_fn: Function to calculate accuracy
        epochs: Number of training epochs
    
    Returns:
        dict: Dictionary containing final training loss, test loss, and test accuracy
    """
    counter = epochs / 5 * 3
    spinner = itertools.cycle(['|', '/', '-', '\\'])

    for epoch in range(epochs):
        train_loss = train(model, train_dataloader, loss_fn, optimizer)
        test_loss, test_acc = test(model, test_dataloader, loss_fn, accuracy_fn)
        
        if( (epoch > 2 and epoch < 10) or (epoch % counter == 0) ):
            # line = next(spinner) + "." * epoch
            # sys.stdout.write('\r' + line)
            # sys.stdout.flush()
            print(".", end="", flush=True)
        if( epoch % counter == 0 or epoch == epochs - 1 ):
            print(f"\rEpoch {epoch:0{3}d} | Train loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
    sys.stdout.write('\r' + ' ' * 30 + '\r') 
    sys.stdout.flush()

    return {
        "train_loss": train_loss.item(),
        "test_loss": test_loss.item(),
        "test_acc": test_acc
    }

def show_image(image, label, block=False):
    print(f"Image shape: {image.shape}")
    plt.imshow(image.squeeze()) 
    plt.title(label)
    plt.show(block=block)
    if( not block ):
        plt.pause(0.001)
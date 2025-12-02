
import os
import sys
sys.path.append("..")
sys.path.append(".")
from helper_functions import accuracy_fn 
import my_utils
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import requests
from pathlib import Path 

EPOCHS = 50
SAMPLES = 1000
NOISE = 0.03
RANDOM_SEED = 42
TEST_SIZE = 0.2
LEARNING_RATE = 0.1
BATCH_SIZE = 32

torch.manual_seed(RANDOM_SEED)

my_utils.wait_for_user_input("Get clothing data - grainy pictures of clothing")
train_data = datasets.FashionMNIST(
    root="data", 
    train=True, # get training data
    download=True, 
    transform=ToTensor(), 
    target_transform=None 
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False, # get test data
    download=True,
    transform=ToTensor()
)

# Turn datasets into iterables (batches)
train_dataloader = DataLoader(train_data, 
    batch_size=BATCH_SIZE, 
    shuffle=True 
)

test_dataloader = DataLoader(test_data,
    batch_size=BATCH_SIZE,
    shuffle=False 
)

my_utils.show_image_non_blocking(train_data[0][0], train_data[0][1])

train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(f"Batch shapes: {train_features_batch.shape}, {train_labels_batch.shape}")

flatten_model = nn.Flatten() # all nn modules function as a model (can do a forward pass)
x = train_features_batch[0]
output = flatten_model(x) # perform forward pass

print(f"Shape before flattening: {x.shape} -> [color_channels, height, width]")
print(f"Shape after flattening: {output.shape} -> [color_channels, height*width]")

class ClothingModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(in_features=input_shape, out_features=hidden_units), 
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )
    
    def forward(self, x):
        return self.layer_stack(x)

model_0 = ClothingModel(input_shape=output.shape[1], # one for every pixel (28x28)
    hidden_units=10, 
    output_shape=len(train_data.classes) 
)
model_0.to("cpu")
loss_fn = nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

my_utils.wait_for_user_input("Training model - hit Enter and wait...")
results = my_utils.test_train_loop(
    model=model_0,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    accuracy_fn=accuracy_fn,
    epochs=EPOCHS
)
print(f"test train loop results: {results}")

my_utils.show_image_non_blocking(train_data[1][0], train_data[1][1])

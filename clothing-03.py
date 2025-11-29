
import os
import sys
sys.path.append("..")
sys.path.append(".")
import my_utils
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

EPOCHS = 1000
SAMPLES = 1000
NOISE = 0.03
RANDOM_SEED = 42
TEST_SIZE = 0.2
LEARNING_RATE = 0.1

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

image, label = train_data[0]
print(f"Image shape: {image.shape}")
my_utils.wait_for_user_input("Show first image")
plt.imshow(image.squeeze()) # image shape is [1, 28, 28] (colour channels, height, width)
plt.title(label)
plt.show()

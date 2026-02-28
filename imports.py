"""
Central imports for simple_pytorch_example_*.py. All heavy libs (torch, matplotlib, etc.)
load as soon as this module is imported. Scripts that hit MPS mutex or memory issues
may want to import only what they need, where they need it (see README: MPS mutex).
"""

import os
os.environ["PYTORCH_MPS_DISABLE"] = "1"

import sys
from my_utils import accuracy_fn
import my_utils

import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from torchinfo import summary

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from pathlib import Path
import pathlib
from typing import Tuple, Dict, List
from PIL import Image
from tqdm.auto import tqdm
import random, re
import time 
from copy import deepcopy


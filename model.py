import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import matplotlib.pyplot as plt
from torchvision import transforms

device = torch.device("cpu")
if torch.cuda.is_available():
    device =  torch.device("cuda")
elif torch.mps.is_available():
    device =  torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device is {device}")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])


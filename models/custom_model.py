import os
import torch
from torch import nn
import torch.nn.functional as F

class CustomModel(nn.Module):
  def __init__(self):
    super(CustomModel, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1)
    self.conv2 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=1)
    self.fc1 = nn.Linear(10 * 12 * 12, 50)
    self.fc2 = nn.Linear(50, 10)
  
  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = x.view(-1, 10 * 12 * 12)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x
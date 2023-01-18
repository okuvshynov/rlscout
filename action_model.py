import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ActionNN(nn.Module):
  def __init__(self):
    super(ActionNN, self).__init__()
    self.action = nn.Sequential(
      nn.Conv2d(2, channels, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(channels, channels, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(channels, 2, kernel_size=1),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(2 * m * n, m * n),
      nn.ReLU(),
      nn.Linear(m * n, m * n),
      nn.LogSoftmax(dim=1),
    )
 
  def forward(self, x):
    return self.action(x)


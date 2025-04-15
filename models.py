import torch
import torch.nn as nn
import torch.nn.functional as F

def add_one(x):
    return x + 1

def loop_op(x):
    for _ in range(5):
        x = x * 2 + 1
    return x

def elementwise_chain(x):
    for _ in range(10):
        x = x * 1.1 + 0.5
    return x

def control_flow(x):
    if x.sum() > 0:
        return x * 2
    else:
        return x - 2

def small_conv(x):
    weight = torch.randn(16, 3, 3, 3, device=x.device)
    return F.conv2d(x, weight, padding=1)

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

class


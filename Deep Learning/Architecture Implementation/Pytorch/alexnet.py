import torch
from torch import nn
import torch.functional as F

class AlexNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=1)
        self.conv2 = nn.Conv2d(96, 256, 5, 1, 3)
        self.conv3 = nn.Conv2d(256, 384, 3, 1, 1)
        self.conv4 = nn.Conv2d(384, 384, 3, 1, 1)
        self.conv3 = nn.Conv2d(384, 256, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d(3, 2)
        self.fc1 = 
    
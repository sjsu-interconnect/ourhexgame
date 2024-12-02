import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class PolicyValueNetwork(nn.Module):
    def __init__(self, board_size, output_dim):
        super(PolicyValueNetwork, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=board_size),  # For a 2x2 board, kernel_size can be 2
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256, pow(board_size, 3)),
            nn.ReLU(),
            nn.Linear(pow(board_size, 3), pow(board_size, 3)),
            nn.ReLU(),
            nn.Linear(pow(board_size, 3), pow(board_size, 3)),
            nn.ReLU(),
            nn.Linear(pow(board_size, 3), pow(board_size, 3)),
            nn.ReLU(),
            nn.Linear(pow(board_size, 3), pow(board_size, 3)),
            nn.ReLU(),
            nn.Linear(pow(board_size, 3), output_dim)
        )

    def forward(self, x):
        x = self.conv_layers(x)

        x = x.view(x.size(0), -1) # convert conv output to a 1D tensor to feed into the fully connected layers
        x = self.fc_layers(x)
        
        return x


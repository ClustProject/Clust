import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

class FC(nn.Module):
    def __init__(self, input_size, drop_out, num_classes, bias, **extra_model_param):
        """
        Args:
            input_size (int): input_size는 representation_size를 의미
        """
        super(FC, self).__init__()
        self.fc1 = nn.Linear(input_size, 32, bias = bias)
        self.fc2 = nn.Linear(32, num_classes, bias = bias)
        self.layer = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(drop_out),
            self.fc2
        )

    def forward(self, x):
        x = self.layer(x)

        return x
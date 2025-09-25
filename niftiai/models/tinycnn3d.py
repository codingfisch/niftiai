import torch.nn as nn
import torch.nn.functional as F


class TinyCNN3d(nn.Module):
    def __init__(self, c_in=1, n_out=1, c_start=4, n_layer=4, kernel_size=3):
        super().__init__()
        channels = [c_in] + [c_start * 2 ** i for i in range(n_layer - 1)] + [n_out]
        layers = []
        for i in range(n_layer):
          layers.append(nn.Conv3d(channels[i], channels[i+1], kernel_size))
          if i < n_layer - 1: layers.append(nn.MaxPool3d(kernel_size))
        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        x = self.cnn(x)
        if sum(x.shape[-3:]) > 3:
            x = F.max_pool3d(x, kernel_size=x.shape[-3:])
        return x[:, :, 0, 0, 0]

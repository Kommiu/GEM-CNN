import torch.nn as nn
import torch.optim as  optim

loss_registry = {
    'mse': nn.MSELoss(),
    'l1': nn.L1Loss(),
    'l1-smooth': nn.SmoothL1Loss(),
}

optimizer_registry = {
    'adam': optim.Adam,
    'adamw': optim.AdamW,
}

nonlinearity_registry = {
    'relu': nn.ReLU(),
}
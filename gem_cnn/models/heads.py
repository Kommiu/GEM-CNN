from torch import nn as nn


class ConvHead(nn.Module):

    def __init__(self, channels=[1, 1], nonlinearity=nn.ReLU()):
        super().__init__()
        self.nonlinearity = nonlinearity
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            for in_channels, out_channels
            in zip(channels[:-1], channels[1:])
        ])

    def forward(self, x):
        for conv in self.convs[:-1]:
            x = self.nonlinearity(conv(x))
        return self.convs[-1](x)


class MLPHead(nn.Module):

    def __init__(self, dims=[1, 1], nonlinearity=nn.ReLU()):
        super().__init__()
        self.nonlinearity = nonlinearity
        self.mlp = nn.ModuleList([
            nn.Linear(in_channels, out_channels)
            for in_channels, out_channels
            in zip(dims[:-1], dims[1:])
        ])

    def forward(self, x):
        for layer in self.mlp[:-1]:
            x = self.nonlinearity(layer(x))
        return self.mlp[-1](x)
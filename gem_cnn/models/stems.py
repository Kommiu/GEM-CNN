import torch
from torch import nn as nn

from gem_cnn.modules import DAGemConv, GemConv, RegularNonlinearity
from torch.utils.checkpoint import checkpoint

class GEMNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_rhos, max_rhos, nonlinearity, is_da=False, da_dims=[]):
        super(GEMNet, self).__init__()
        self.num_rhos = num_rhos
        self.max_rhos = max_rhos
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nonlinearity = nonlinearity
        self.is_da = is_da
        self.da_dims = da_dims

        Conv = DAGemConv if is_da else GemConv
        self.gem_convs = nn.ModuleList([
            Conv(
                [0]*input_dim,
                [el for x in range(max_rhos[0] + 1) for el in [x]*num_rhos[0]],
                da_dims
            ),
        ])
        for num_in, max_in, num_out, max_out in zip(num_rhos[:-1], max_rhos[:-1], num_rhos[1:], max_rhos[1:]):
            self.gem_convs.append(Conv(
                [el for x in range(max_in + 1) for el in [x] * num_in],
                [el for x in range(max_out + 1) for el in [x] * num_out],
                da_dims,

            ))
        self.gem_convs.append(
            Conv(
                [el for x in range(max_rhos[0] + 1) for el in [x] * num_rhos[-1]],
                [0] * output_dim,
                da_dims,
            )
        )


        self.nonlinearities = nn.ModuleList([
            RegularNonlinearity({rho: num for rho in range(max_rho + 1)}, self.nonlinearity)
            for num, max_rho in zip(num_rhos, max_rhos)
        ])

    def forward(self, data):
        if 'x' in data.keys:
            x = torch.cat([data.pos, data.x], dim=1)
        else:
            x = data.pos
        edge_index = data.edge_index
        theta = data.theta
        g = data.g
        x = nn.Parameter(x)
        # dist is None if there is no distance in data
        dist = data['distance']
        x = checkpoint(self.gem_convs[0], x, theta, g, edge_index, dist)
        x = checkpoint(self.nonlinearities[0], x)
        for i in range(1, len(self.gem_convs) - 1):
            x = x + checkpoint(self.gem_convs[i], x, theta, g, edge_index, dist)
            x = checkpoint(self.nonlinearities[i], x)
        x = self.nonlinearity(checkpoint(self.gem_convs[-1], x, theta, g, edge_index, dist))
        return x
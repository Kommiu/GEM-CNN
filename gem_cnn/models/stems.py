import torch
from torch import nn as nn

from gem_cnn.modules import DAGemConv, GemConv, RegularNonlinearity
import gem_cnn.utils as utils


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

        if is_da:
            self.gem_convs = nn.ModuleList([
                DAGemConv(
                    [0]*input_dim,
                    [el for x in range(max_rhos[0] + 1) for el in [x]*num_rhos[0]],
                    da_dims
                ),
            ])
            for num_in, max_in, num_out, max_out in zip(num_rhos[:-1], max_rhos[:-1], num_rhos[1:], max_rhos[1:]):
                self.gem_convs.append(DAGemConv(
                    [el for x in range(max_in + 1) for el in [x] * num_in],
                    [el for x in range(max_out + 1) for el in [x] * num_out],
                    da_dims,

                ))
            self.gem_convs.append(
                DAGemConv(
                    [el for x in range(max_rhos[0] + 1) for el in [x] * num_rhos[-1]],
                    [0] * output_dim,
                    da_dims,
                )
            )

        else:
            self.gem_convs = nn.ModuleList([
                GemConv(
                    [0] * input_dim,
                    [el for x in range(max_rhos[0] + 1) for el in [x] * num_rhos[0]],
                ),

             ])
            for num_in, max_in, num_out, max_out in zip(num_rhos[:-1], max_rhos[:-1], num_rhos[1:], max_rhos[1:]):
                self.gem_convs.append(GemConv(
                    [el for x in range(max_in + 1) for el in [x] * num_in],
                    [el for x in range(max_out + 1) for el in [x] * num_out]

                ))
            self.gem_convs.append(
                GemConv(
                    [el for x in range(max_rhos[0] + 1) for el in [x] * num_rhos[-1]],
                    [0] * output_dim
                )
            )

        self.nonlinearities = nn.ModuleList([
            RegularNonlinearity({rho: num for rho in range(max_rho + 1)}, self.nonlinearity)
            for num, max_rho in zip(num_rhos, max_rhos)
        ])

    def forward(self, data):
        x = torch.cat([data.pos, data.x], dim=1)
        edge_index = data.edge_index

        max_rho = max(self.max_rhos) + 1
        bases = utils.get_bases(data.theta, range(1, max_rho), range(1, max_rho))
        self_basis = utils.get_self_basis(x.dtype, x.device)
        operators = {n: utils.get_operator(n, data.g) for n in range(1, max_rho)}

        if self.is_da:
            x = self.gem_convs[0](x, theta, g, edge_index, data.distance)
            x = self.nonlinearities[0](x)
            for i in range(1, len(self.gem_convs) - 1):
                x = self.gem_convs[i](x, theta, g, edge_index, data.distance)
                x = self.nonlinearities[i](x)
            x = self.nonlinearity(self.gem_convs[-1](x, theta, g, edge_index, data.distance))

        else:
            x = self.gem_convs[0](x, bases, self_basis, operators, edge_index)
            x = self.nonlinearities[0](x)
            print(x.shape)
            for i in range(1, len(self.gem_convs) - 1):
                x = self.gem_convs[i](x, bases, self_basis, operators, edge_index)
                x = self.nonlinearities[i](x)
                print(x.shape)
            x = self.nonlinearity(self.gem_convs[-1](x, bases, self_basis, operators, edge_index))

        return x
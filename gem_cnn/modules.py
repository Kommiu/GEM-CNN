from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import uniform
from torch_scatter import scatter_add


class Transporter(nn.Module):
    def __init__(self,  rhos):
        super().__init__()
        self.rhos = rhos

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.rhos}'

    @staticmethod
    def _rho(n, g):

        return torch.stack([
            torch.stack([torch.cos(n * g), -torch.sin(n * g)], dim=-1),
            torch.stack([torch.sin(n * g), torch.cos(n * g)], dim=-1),
        ],
            dim=-1
        )

    def forward(self, x, g):
        pos = 0
        res = torch.zeros_like(x)
        for i, n in enumerate(self.rhos):
            width = 1 if n == 0 else 2
            if n == 0:
                res[:, pos] += x[:, pos]
            else:
                operator = self._rho(n, g)
                res[:, pos: pos + width] += torch.einsum('pj, pij -> pi', x[:, pos: pos + width], operator)
            pos += width

        return res


class BasicNeighbourKernel(nn.Module):
    def __init__(self, n, m):
        super().__init__()
        self.m = m
        self.n = n
        if m == n == 0:
            self.weights = nn.Parameter(torch.ones(1))
        elif m == 0 or n == 0:
            self.weights = nn.Parameter(torch.ones(2))
        else:
            self.weights = nn.Parameter(torch.ones(4))

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.weights.size(0), self.weights)

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.n}, {self.m}'

    def forward(self, x, theta):
        n, m = self.n, self.m
        if n == 0:
            if m == 0:
                res = self.weights[0] * x
                return res
            else:
                # |E| x 2
                res = torch.stack([torch.cos(m * theta), torch.sin(m * theta)], dim=-1).unsqueeze(dim=2) * self.weights[0]
                res += torch.stack([torch.sin(m * theta), -torch.cos(m * theta)], dim=-1).unsqueeze(dim=2) * self.weights[1]
        else:
            if m == 0:
                res = torch.stack([torch.cos(n * theta), torch.sin(n * theta)], dim=-1).unsqueeze(dim=1)*self.weights[0]
                res += torch.stack([torch.sin(n * theta), -torch.cos(n * theta)], dim=-1).unsqueeze(dim=1)*self.weights[1]

            else:

                res = torch.stack([
                    torch.cos(theta * (m - n)), -torch.sin(theta * (m - n)),
                    torch.sin(theta * (m - n)), torch.cos(theta * (m - n)),
                ], dim=-1).reshape(-1, 2, 2) * self.weights[0]

                res += torch.stack([
                    torch.sin(theta * (m - n)), torch.cos(theta * (m - n)),
                    -torch.cos(theta * (m - n)), torch.sin(theta * (m - n)),
                ], dim=-1).reshape(-1, 2, 2) * self.weights[1]
                res += torch.stack([
                    torch.cos(theta * (m + n)), torch.sin(theta * (m + n)),
                    torch.sin(theta * (m + n)), -torch.cos(theta * (m + n)),
                ], dim=-1).reshape(-1, 2, 2) * self.weights[2]
                res += torch.stack([
                    -torch.sin(theta * (m + n)), torch.cos(theta * (m + n)),
                    torch.cos(theta * (m + n)), torch.sin(theta * (m + n))
                ], dim=-1).reshape(-1, 2, 2) * self.weights[3]

        res = torch.einsum('pj, pij -> pi', x, res)

        return res


class NeighbourKernel(nn.Module):
    def __init__(self, rho_in, rho_out):
        super().__init__()
        self.rho_in = rho_in
        self.rho_out = rho_out

        self.kernels = nn.ModuleList()

        for m, n in product(rho_out, rho_in):
            self.kernels.append(BasicNeighbourKernel(n, m))

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.rho_in}, {self.rho_out}'

    def forward(self, x, theta):
        m = len(self.rho_out)
        n = len(self.rho_in)
        res = torch.empty(len(x), 0, dtype=x.dtype, device=x.device)
        for i in range(m):
            pos = 0
            row = torch.zeros(len(x), 2 ** (np.sign(self.rho_out[i])), dtype=x.dtype, device=x.device)
            for j in range(n):
                ker = self.kernels[i * n + j]
                width = 2 ** (np.sign(self.rho_in[j]))
                row += ker(x[:, pos: pos + width], theta)
                pos += width
            res = torch.cat([res, row], dim=-1)

        return res


class BasicSelfKernel(nn.Module):

    def __init__(self, n, m):
        super().__init__()
        self.m = m
        self.n = n
        if m == n == 0:
            self.weights = nn.Parameter(torch.FloatTensor(1))
        elif m == 0 or n == 0:
            self.weights = None
        else:
            self.weights = nn.Parameter(torch.FloatTensor(2))

        self.reset_parameters()

    def reset_parameters(self):
        if self.weights is not None:
            uniform(self.weights.size(0), self.weights)

    def forward(self, x):
        n, m = self.n, self.m
        if m != n:
            if m == 0:
                # x = |E| x 2
                return torch.zeros_like(x)
            elif n == 0:
                # x = |E| x 1
                return F.pad(torch.zeros_like(x), (0, 1), 'constant', 0)
            else:
                # x = |E| x 2
                return torch.zeros_like(x)
        else:
            if m == 0:
                return self.weights[0] * x
            else:
                kernel = self.weights[0] * torch.eye(2, device=x.device)\
                       + self.weights[1] * torch.FloatTensor([[0, 1], [-1, 0]]).to(x.device)
                return x @ kernel

class SelfKernel(nn.Module):
    def __init__(self, rho_in, rho_out):
        super().__init__()
        self.rho_in = rho_in
        self.rho_out = rho_out

        self.kernels = nn.ModuleList()

        for m, n in product(rho_out, rho_in):
            self.kernels.append(BasicSelfKernel(n, m))

    def forward(self, x):
        m = len(self.rho_out)
        n = len(self.rho_in)
        res = torch.empty(len(x), 0, dtype=x.dtype, device=x.device)
        for i in range(m):
            pos = 0
            row = torch.zeros(len(x), 2 ** (np.sign(self.rho_out[i])), dtype=x.dtype, device=x.device)
            for j in range(n):
                if self.rho_in[j] != self.rho_out[i]:
                    continue
                ker = self.kernels[i * n + j]
                width = 2 ** (np.sign(self.rho_in[j]))
                row += ker(x[:, pos: pos + width])
                pos += width

            res = torch.cat([res, row], dim=-1)

        return res


class GemConv(MessagePassing):
    def __init__(self, rho_in, rho_out):
        super(GemConv, self).__init__(aggr='add',  flow='target_to_source',)  # "Add" aggregation.
        self.neighbour_kernel = NeighbourKernel(rho_in, rho_out)
        self.transporter = Transporter(rho_in)
        self.self_kernel = SelfKernel(rho_in, rho_out)

    def forward(self, x, theta, g, edge_index):
        neighbours = self.propagate(edge_index, x=x, g=g, theta=theta)
        return neighbours + self.self_kernel(x)

    def message(self, x_j, theta, g):
        x = self.transporter(x_j, g)
        return self.neighbour_kernel(x, theta)


class RegularNonlinearity(nn.Module):
    def __init__(self, rhos, nonlinearity):
        super().__init__()
        self.rhos = rhos
        if len(rhos) == 1 and 0 in rhos:
            self = nonlinearity
        else:
            assert len(set(rhos.values())) == 1
            self.num = self.rhos[0]
            self.nonlinearity = nonlinearity
            mask = torch.LongTensor([2 * k for k in range(self.num)])
            self.mask = torch.cat([mask, torch.arange(self.num * 2, self.num * 2 * len(self.rhos))])

    def forward(self, x):
        # batch_size x rho[0] + 2(len(rho) - 1)*rho[0]
        batch_len = len(x)
        mask = self.mask.to(x.device)
        grouped = scatter_add(x, mask).reshape(batch_len, len(self.rhos), self.num, 2).transpose(1, 2)

        signal = torch.irfft(grouped, 1)
        result = torch.rfft(self.nonlinearity(signal), 1, onesided=True).transpose(1, 2)
        result = result.reshape(batch_len, -1)[:, mask]
        return result


class MLP(nn.Module):
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

class DAGemConv(MessagePassing):
    def __init__(self, rho_in, rho_out, mlp_dims):
        super(DAGemConv, self).__init__(aggr='add',  flow='target_to_source',)  # "Add" aggregation.
        self.neighbour_kernel = NeighbourKernel(rho_in, rho_out)
        self.transporter = Transporter(rho_in)
        self.self_kernel = SelfKernel(rho_in, rho_out)
        mlp_dims = [1] + mlp_dims + [1]
        self.distance_weight = MLP(mlp_dims)


    def forward(self, x, theta, g, edge_index, dist):
        neighbours = self.propagate(edge_index, x=x, g=g, theta=theta, dist=dist)
        return neighbours + self.self_kernel(x)

    def message(self, x_j, theta, g, dist):
        x = self.transporter(x_j, g)
        res = self.neighbour_kernel(x, theta) * self.distance_weight(dist)
        print(res.shape)
        return res
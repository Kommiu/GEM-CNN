from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import uniform
from torch_scatter import scatter_add
from pytorch_memlab import MemReporter


class Transporter(nn.Module):
    def __init__(self,  rhos):
        super().__init__()
        self.rhos = rhos

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.rhos}'

    @staticmethod
    def _rho(n, g):

        operator = torch.zeros(len(g), 2, 2, dtype=g.dtype, device=g.device)
        operator[:, 0, 0] = torch.cos(n * g)
        operator[:, 0, 1] = -torch.sin(n * g)
        operator[:, 1, 0] = torch.sin(n * g)
        operator[:, 1, 1] = torch.cos(n * g)
        return operator

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
                # |E| x 2 x 1
                res = torch.zeros(len(theta), 2, 1, device=theta.device, dtype=theta.dtype)
                res[:, 0, 0] += torch.cos(m * theta) * self.weights[0] + torch.sin(m * theta) * self.weights[1]
                res[:, 1, 0] += torch.sin(m * theta) * self.weights[0] - torch.cos(m * theta) * self.weights[1]

        else:
            # |E| x 1 x 2
            if m == 0:
                res = torch.zeros(len(theta), 1, 2, device=theta.device, dtype=theta.dtype)
                res[:, 0, 0] += torch.cos(n * theta) * self.weights[0] + torch.sin(n * theta) * self.weights[1]
                res[:, 0, 1] += torch.sin(n * theta) * self.weights[0] + torch.cos(n * theta) * self.weights[1]

            else:
                # |E| x 2 x 2
                res = torch.zeros(len(theta), 2, 2, device=theta.device, dtype=theta.dtype)

                res[:, 0, 0] = torch.cos(theta * (m - n)) * self.weights[0] + \
                    torch.sin(theta * (m - n)) * self.weights[1] + \
                    torch.cos(theta * (m + n)) * self.weights[2] + \
                    - torch.sin(theta * (m + n)) * self.weights[3]

                res[:, 0, 1] = -torch.sin(theta * (m - n)) * self.weights[0] + \
                    torch.cos(theta * (m - n)) * self.weights[1] + \
                    torch.sin(theta * (m + n)) * self.weights[2] + \
                    torch.cos(theta * (m + n)) * self.weights[3]

                res[:, 1, 0] = torch.sin(theta * (m - n)) * self.weights[0] + \
                    -torch.cos(theta * (m - n)) * self.weights[1] + \
                    torch.sin(theta * (m + n)) * self.weights[2] + \
                    torch.cos(theta * (m + n)) * self.weights[3]

                res[:, 1, 1] = torch.cos(theta * (m - n)) * self.weights[0] + \
                    torch.sin(theta * (m - n)) * self.weights[1] + \
                    -torch.cos(theta * (m + n)) * self.weights[2] + \
                    torch.sin(theta * (m + n)) * self.weights[3]

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
        c_out = sum(1 if x == 0 else 2 for x in self.rho_out)
        res = torch.zeros(len(x), c_out, dtype=x.dtype, device=x.device)
        pos_out = 0
        for i in range(m):
            width_out = 1 if self.rho_out[i] == 0 else 2
            pos_in = 0
            for j in range(n):
                width_in = 1 if self.rho_in[j] == 0 else 2
                ker = self.kernels[i * n + j]
                res[:, pos_out: pos_out + width_out] += ker(x[:, pos_in: pos_in + width_in], theta)
                pos_in += width_in
            pos_out += width_out
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
                kernel = self.weights[0] * torch.eye(2, dtype=x.dtype, device=x.device)\
                       + self.weights[1] * torch.tensor([[0, 1], [-1, 0]], dtype=x.dtype, device=x.device)
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
        c_out = sum(1 if x == 0 else 2 for x in self.rho_out)
        res = torch.zeros(len(x), c_out, dtype=x.dtype, device=x.device)
        pos_out = 0
        for i in range(m):
            pos_in = 0
            width_out = 1 if self.rho_out[i] == 0 else 2
            for j in range(n):
                if self.rho_in[j] != self.rho_out[i]:
                    continue
                ker = self.kernels[i * n + j]
                width_in = 1 if self.rho_in[j] == 0 else 2
                res[:, pos_out: pos_out + width_out] += ker(x[:, pos_in: pos_in + width_in])
                pos_in += width_in
            pos_out += width_out
        return res


class GemConv(MessagePassing):
    def __init__(self, rho_in, rho_out):
        super(GemConv, self).__init__(aggr='add',  flow='target_to_source',)  # "Add" aggregation.
        self.neighbour_kernel = NeighbourKernel(rho_in, rho_out)
        self.transporter = Transporter(rho_in)
        self.self_kernel = SelfKernel(rho_in, rho_out)
        # self.reporter = MemReporter(self)

    def forward(self, x, theta, g, edge_index):
        neighbours = self.propagate(edge_index, x=x, g=g, theta=theta)
        # self.reporter.report()
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
        return self.neighbour_kernel(x, theta) * self.distance_weight(dist.unsqueeze(-1))

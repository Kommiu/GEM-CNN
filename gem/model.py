from itertools import product
from functools import reduce
from operator import mul

import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import block_diag


def make_blof


class Transporter(nn.Module):
    def __init__(self,  rhos, g):
        super().__init__()
        self.rhos = rhos
        diag = list()
        for n in self.rhos:
            diag.extend(self._rho(n)(g))

        self.transporter = block_diag(*diag)
    @staticmethod
    def _rho(n):
        def _rho_0(g):
            return 1

        def _rho_n(g):
            return np.array([
                [np.cos(n * g), -np.sin(n * g)],
                [np.sin(n * g), np.cos(n * g)]
            ])

        if n == 0:
            return _rho_0
        else:
            return _rho_n

    def forward(self, g):
        


class BasicKernel(nn.Module):
    def __init__(self, n, m):
        super().__init__()
        self.m = m
        self.n = n
        if m == n == 0:
            self.weights = nn.Parameter(torch.FloatTensor(1))
        elif m == 0 or n == 0:
            self.weights = nn.Parameter(torch.FloatTensor(2))
        else:
            self.weights = nn.Parameter(torch.FloatTensor(4))

    def forward(self, theta):
        n, m = self.n, self.m
        if n == 0:
            if m == 0:
                res = torch.FloatTensor([[1]])
            else:
                res = torch.stack([[torch.cos(m * theta), torch.sin(m * theta)]]).t(), \
                      torch.stack([[torch.sin(m * theta), -torch.cos(m * theta)]]).t()
        else:
            if m == 0:
                res = torch.stack([[torch.cos(n * theta), torch.sin(n * theta)]]), \
                      torch.stack([[torch.sin(n * theta), -torch.cos(n * theta)]])

            else:
                res = (torch.stack([
                    [torch.cos(theta * (m - n)), -torch.sin(theta * (m - n))],
                    [torch.sin(theta * (m - n)), torch.cos(theta * (m - n))]
                ]),
                       torch.stack([
                           [torch.sin(theta * (m - n)), torch.cos(theta * (m - n))],
                           [-torch.cos(theta * (m - n)), torch.sin(theta * (m - n))],
                       ]),
                       torch.stack([
                           [torch.cos(theta * (m + n)), torch.sin(theta * (m + n))],
                           [torch.sin(theta * (m + n)), -torch.cos(theta * (m + n))],
                       ]),
                       torch.stack([
                           [-torch.sin(theta * (m + n)), torch.cos(theta * (m + n))],
                           [torch.cos(theta * (m + n)), torch.sin(theta * (m + n))]
                       ]))

        return sum(t * w for t, w in zip(res, self.weights))


class GEMKernel(nn.Module):
    def __init__(self, rho_in, rho_out):
        super().__init__()
        self.rho_in = rho_in
        self.rho_out = rho_out

        self.kernels = nn.ModuleList()

        for m, n in product(rho_out, rho_in):
            self.kernels.append(BasicKernel(n, m))

    def forward(self, theta):
        m = len(self.rho_out)
        n = len(self.rho_in)
        res = [
            torch.cat([ker(theta) for ker in self.kernels[k * n: (k + 1) * n]], dim=1)
            for k
            in range(m)
        ]
        return torch.cat(res, dim=0)




class GemConv(nn.Module):
    def __init__(self,n, rho_in, rho_out):
        self.theta = torch.sparse.FloatTensor(n, n)
        self.kernel_neigh = GEMKernel(rho_in, rho_out)

    def forward(self, f):
        torch.sparse.mm(self.kernel_neigh(self.theta), self.adj)




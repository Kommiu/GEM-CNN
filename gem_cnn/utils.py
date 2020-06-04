from typing import Dict
from itertools import product

import numpy as np
import torch
from torch import nn as nn
from torch_scatter import scatter_add
from trimesh import load_mesh
from trimesh.base import Trimesh

from gem_cnn.torch_geometric_path.data import Data


def normalize(matrix, axis=-1):
    return matrix / np.linalg.norm(matrix, axis=axis, keepdims=True)


def get_vertex_normals(mesh: Trimesh):
    vertex_normals = list()
    for faces, degree in zip(mesh.vertex_faces, mesh.vertex_degree):
        faces = faces[:degree]
        areas = mesh.area_faces[faces]
        normals = mesh.face_normals[faces]

        v_normal = (normals * areas.reshape((-1, 1))).sum(axis=0) / areas.sum()
        vertex_normals.append(v_normal)

    vertex_normals = np.stack(vertex_normals)
    return normalize(vertex_normals)




def get_batched_adj(values, adj_matrix, indices):
    p_idx = indices
    q_idx = np.where(
        np.asarray(
            adj_matrix[p_idx].sum(axis=0)
        ).squeeze()
    )
    return adj_matrix[p_idx].T[q_idx].T


def sample_adj(adj_matrix, p_idx):
    q_idx = np.where(
        np.asarray(
            adj_matrix[p_idx].sum(axis=0)
        ).squeeze()
    )
    adj_sliced = adj_matrix[p_idx].T[q_idx].T.todense()
    return adj_sliced


def make_projector(normals):
    return torch.eye(3) - torch.einsum('pi, pj-> pij', normals, normals)


def weighted_normals(face_normals, face_areas, faces, vertex_num):
    vertex_normals = sum(
        scatter_add(face_normals * face_areas.unsqueeze(dim=-1), faces[:, i], dim=0, dim_size=vertex_num)
        for i in range(3)
    )
    # vertex_normals /= sum(scatter_add(face_areas, faces[:, i]) for i in range(3)).unsqueeze(-1)
    vertex_normals /= vertex_normals.norm(dim=-1, keepdim=True)
    return vertex_normals


def read_mesh(path):
    mesh = load_mesh(path)
    pos = torch.FloatTensor(mesh.vertices)
    edges = mesh.edges
    edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    edges = torch.LongTensor(edges.T)
    face_normals = torch.FloatTensor(mesh.face_normals)
    face_areas = torch.FloatTensor(mesh.area_faces)
    faces = torch.LongTensor(mesh.faces)
    vertex_normals = torch.FloatTensor(mesh.vertex_normals)

    data = Data(
        pos=pos,
        edge_index=edges,
        vertex_normals=vertex_normals,
        face_normals=face_normals,
        face_areas=face_areas,
        faces=faces,
    )
    return data


def nan_filter(data):
    return not any(torch.isnan(data[key]).any().item() for key in data.keys)


class ModuleType:
    def __init__(self, module_dict: Dict[str, nn.Module]) -> nn.Module:
        self._dict = module_dict

    def __call__(self, key):
        return self._dict[key]


def get_neighbor_basis(theta, n, m):
   basis = torch.zeros(
       len(theta), 2, 2, 4,
       device=theta.device, dtype=theta.dtype,
   )
   basis[:, 0, 0, 0] += torch.cos(theta * (m - n))
   basis[:, 0, 1, 0] += -torch.sin(theta * (m - n))
   basis[:, 1, 0, 0] += torch.sin(theta * (m - n))
   basis[:, 1, 1, 0] += torch.cos(theta * (m - n))

   basis[:, 0, 0, 1] += torch.sin(theta * (m - n))
   basis[:, 0, 1, 1] += torch.cos(theta * (m - n))
   basis[:, 1, 0, 1] += -torch.cos(theta * (m - n))
   basis[:, 1, 1, 1] += torch.sin(theta * (m - n))

   basis[:, 0, 0, 2] += torch.cos(theta * (m + n))
   basis[:, 0, 1, 2] += torch.sin(theta * (m + n))
   basis[:, 1, 0, 2] += torch.sin(theta * (m + n))
   basis[:, 1, 1, 2] += -torch.cos(theta * (m + n))

   basis[:, 0, 0, 3] += -torch.sin(theta * (m + n))
   basis[:, 0, 1, 3] += torch.cos(theta * (m + n))
   basis[:, 1, 0, 3] += torch.cos(theta * (m + n))
   basis[:, 1, 1, 3] += torch.sin(theta * (m + n))

   return basis


def get_neighbor_basis_0(theta, n):
    basis = torch.zeros(
        len(theta), 2, 1, 2,
        device=theta.device, dtype=theta.dtype,
    )
    basis[:, 0, 0, 0] += torch.cos(n * theta)
    basis[:, 1, 0, 0] += torch.sin(n * theta)
    basis[:, 0, 0, 1] += torch.sin(n * theta)
    basis[:, 1, 0, 1] += -torch.cos(n * theta)
    return basis


def get_self_basis(dtype, device):
    basis = torch.zeros(
        2, 2, 2,
        dtype=dtype, device=device,
    )
    basis[:, :, 0] += torch.eye(2, dtype=dtype, device=device)
    basis[:, :, 1] += torch.tensor(
        [[0, 1], [-1, 0]],
        dtype=dtype, device=device,
    )
    return basis


def get_bases(theta, rhos_in, rhos_out):
    bases = {
        (n, m): get_neighbor_basis(theta, n, m).contiguous()
        for n, m
        in product(rhos_in, rhos_out)
    }
    for n in rhos_in:
        bases[(n, 0)] = get_neighbor_basis_0(theta, n).transpose(1, 2).contiguous()

    for m in rhos_out:
        bases[(0, m)] = get_neighbor_basis_0(theta, m).contiguous()

    return bases


def get_operator(n, g):

    operator = torch.zeros(len(g), 2, 2, dtype=g.dtype, device=g.device)
    operator[:, 0, 0] = torch.cos(n * g)
    operator[:, 0, 1] = -torch.sin(n * g)
    operator[:, 1, 0] = torch.sin(n * g)
    operator[:, 1, 1] = torch.cos(n * g)
    return operator
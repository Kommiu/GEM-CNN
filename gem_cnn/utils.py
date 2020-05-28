import numpy as np
import torch
from trimesh.base import Trimesh


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






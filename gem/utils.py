from trimesh.base import Trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R
from itertools import product
import networkx as nx


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


# def log_map(q, p, n):
#     diff = q - p
#     n = n.reshape((-1, 1))
#     v = diff @ (np.eye(3) - n @ n.T)
#
#     return np.linalg.norm(diff, axis=1, keepdims=True) * v / np.linalg.norm(v, axis=1, keepdims=True)

#
# def get_basis(lm0, n):
#     lm0 = lm0.reshape(-1)
#     n = n.reshape(-1)
#     e1 = lm0 / np.linalg.norm(lm0)
#     e2 = np.cross(n, e1)
#     return e1.reshape((-1, 1)), e2.reshape((-1, 1))
#
#
# def get_angles(q, p, n):
#     lm = log_map(q, p, n)
#     e1, e2 = get_basis(lm[0], n)
#     thetas = np.arctan2(lm @ e2, lm @ e1)
#     return thetas


# def get_transporter(q, p, np, nq):
#     alpha = np.arccos(nq @ np)
#     rotvec = alpha * np.cross(nq, np.reshape((1, -1)))
#     rotation = R.from_rotvec(rotvec)


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


def get_pairwise_diffs(values, adj):
    p = np.einsum('pi, pq -> pqi', values, adj)
    q = np.einsum('qi, pq -> pqi', values, adj)
    return p - q

def projector(normals):
    return np.eye(3) - np.einsum('pi, pj-> pij', normals, normals)

def log_map(diffs, normals, adj):

    proj = projector(normals)
    vecs = normalize(np.einsum('pqi, pij -> pqj', diffs, proj))
    e1 = diffs[np.arange(len(diffs)), adj.argmax(axis=1)]
    e2 = np.cross(e1, normals)
    return np.linalg.norm(diffs, axis=-1, keepdims=True) * vecs, e1, e2



def get_angles(e1, e2, lm):
    x = np.einsum('pi, pqi->pq', e1, lm)
    y = np.einsum('pi, pqi -> pq', e2, lm)
    return np.arctan2(y, x)

def get_transporter(n_p, n_q, ep1, ep2, eq1, eq2):
    alpha = np.arccos(np.einsum('pi, qi->pq', n_p, n_q).clip(-1, 1))
    axis = np.array([np.cross(a, b) for a, b in product(n_p, n_q)])
    rotation = R.from_rotvec(alpha.reshape(-1, 1) * axis).as_matrix().reshape(alpha.shape + (3, 3))
    x = np.einsum('pqij, qi, pj -> pq', rotation, eq1, ep1)
    y = np.einsum('pqij, qi, pj -> pq', rotation, eq2, ep2)
    return np.arctan2(y, x)


def process_mesh(mesh: Trimesh):

    normals = get_vertex_normals(mesh)
    adj = np.asarray(nx.adjacency_matrix(mesh.vertex_adjacency_graph).todense())
    diffs = get_pairwise_diffs(mesh.vertices, adj)
    lm, e1, e2 = log_map(diffs, normals, adj)
    thetas = get_angles(e1, e2, lm)
    print('a')
    transporter = get_transporter(normals, normals, e1, e2, e1, e2)
    return thetas, transporter



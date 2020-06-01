import torch
from scipy.spatial.transform import Rotation as R
from torch_scatter import scatter_min

from gem_cnn.torch_geometric_path.data import Data
from gem_cnn.torch_geometric_path.utils import k_hop_subgraph
from gem_cnn.utils import make_projector, weighted_normals


class GEMTransform:

    def __init__(self, weighted_normals=True, distance=False):
        self.distance = distance
        self.weighted_normals = weighted_normals

    def __call__(self, data: Data):
        pos = data.pos
        edges = data.edge_index

        if self.weighted_normals:
            normals = weighted_normals(data.face_normals, data.face_areas, data.faces, len(data.pos))
        else:
            normals = data.vertex_normals

        diffs = pos[edges[0]] - pos[edges[1]]
        projectors = make_projector(normals)

        projected = torch.einsum('pij, pj -> pi', projectors[edges[0]], pos[edges[1]])
        projected /= projected.norm(dim=1, keepdim=True)

        gauge, _ = scatter_min(edges[1], edges[0])
        e1 = projected[gauge]
        e2 = torch.cross(e1, normals)
        log_map = projected * diffs.norm(dim=1, keepdim=True)

        theta_x = torch.einsum('pi, pi -> p', e1[edges[0]], log_map)
        theta_y = torch.einsum('pi, pi -> p', e2[edges[0]], log_map)
        theta = torch.atan2(theta_y, theta_x)

        axis = torch.cross(normals[edges[1]], normals[edges[0]])
        alpha = torch.einsum('pi, pi -> p', normals[edges[0]], normals[edges[1]]).clamp(-1, 1)
        alpha = torch.acos(alpha)
        rotvec = (alpha.unsqueeze(dim=-1) * axis).numpy()
        rotation = R.from_rotvec(rotvec)
        g_x = rotation.apply(e1[edges[1]].numpy())
        g_x = torch.einsum('pi, pi -> p', torch.FloatTensor(g_x), e1[edges[0]])
        g_y = rotation.apply(e2[edges[1]].numpy())
        g_y = torch.einsum('pi, pi -> p', torch.FloatTensor(g_y), e2[edges[0]])
        g = torch.atan2(g_y, g_x)

        del data.vertex_normals
        del data.faces
        del data.face_normals
        del data.face_areas
        if self.distance:
            data.distance = diffs.norm(dim=1)
        data.g = g
        data.theta = theta
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}'


class GetLocalPatch:
    def __init__(self, patch_radius, k_hops):
        self.patch_radius = patch_radius
        self.k_hops = k_hops

    def __call__(self, data: Data) -> Data:
        seed = torch.randint(0, data.num_nodes, size=(1, )).item()
        patch, edge_index, mapping, mask = k_hop_subgraph(
            seed,
            self.patch_radius,
            data.edge_index,
            num_nodes=data.num_nodes
        )
        nodes, edge_index, mapping, mask = k_hop_subgraph(
            patch,
            self.k_hops,
            data.edge_index,
            num_nodes=data.num_nodes
        )
        new_data = Data()
        for key in data.keys:
            if len(data[key]) == data.num_nodes:
                new_data[key] = data[key][nodes].clone().detach()
            if len(data[key]) == data.num_edges:
                new_data[key] = data[key][mask].clone().detach()

        node_mask = torch.arange(len(nodes))
        edge_mask, _ = torch.where(nodes.unsqueeze(-1) == edge_index[0])
        edge_index[0] = node_mask[edge_mask]
        edge_mask, _ = torch.where(nodes.unsqueeze(-1) == edge_index[1])
        edge_index[1] = node_mask[edge_mask]

        new_data.target_nodes = mapping
        new_data.edge_index = edge_index
        new_data.original_points = patch
        return new_data

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.patch_radius}, {self.k_hops}'


class Scale(object):
    r"""Row-normalizes node features to sum-up to one."""

    def __init__(self, scale_x, scale_y):
        self.scale_x = torch.tensor(scale_x)
        self.scale_y = torch.tensor(scale_y)

    def __call__(self, data):
        data.x = data.x * self.scale_x
        data.y = data.y * self.scale_y
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}: x_scale={self.scale_x.tolist()}, y_scale={self.scale_y.tolist()}'
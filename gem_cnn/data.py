from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric
import torch_geometric.data as tgd
from scipy.spatial.transform import Rotation as R
from torch._six import container_abcs, string_classes, int_classes
from torch.utils.data.dataloader import default_collate
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_min, scatter_add
from trimesh import load_mesh

from .utils import make_projector


class Data(tgd.Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to(self, device, *keys, **kwargs):
        r"""Performs tensor dtype and/or device conversion to all attributes
        :obj:`*keys`.
        If :obj:`*keys` is not given, the conversion is applied to all present
        attributes."""
        return self.apply(lambda x: x.to(device, **kwargs), *keys)


class Batch(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """
    def __init__(self, batch=None, **kwargs):
        super(Batch, self).__init__(**kwargs)

        self.batch = batch
        self.__data_class__ = Data
        self.__slices__ = None

    @staticmethod
    def from_data_list(data_list, follow_batch=[]):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        Additionally, creates assignment batch vectors for each key in
        :obj:`follow_batch`."""

        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = Batch()
        batch.__data_class__ = data_list[0].__class__
        batch.__slices__ = {key: [0] for key in keys}

        for key in keys:
            batch[key] = []

        for key in follow_batch:
            batch['{}_batch'.format(key)] = []

        cumsum = {key: 0 for key in keys}
        batch.batch = []
        for i, data in enumerate(data_list):
            for key in data.keys:
                item = data[key]
                if torch.is_tensor(item) and item.dtype != torch.bool:
                    item = item + cumsum[key]
                if torch.is_tensor(item):
                    size = item.size(data.__cat_dim__(key, data[key]))
                else:
                    size = 1
                batch.__slices__[key].append(size + batch.__slices__[key][-1])
                cumsum[key] = cumsum[key] + data.__inc__(key, item)
                batch[key].append(item)

                if key in follow_batch:
                    item = torch.full((size, ), i, dtype=torch.long)
                    batch['{}_batch'.format(key)].append(item)

            num_nodes = data.num_nodes
            if num_nodes is not None:
                item = torch.full((num_nodes, ), i, dtype=torch.long)
                batch.batch.append(item)

        if num_nodes is None:
            batch.batch = None

        for key in batch.keys:
            item = batch[key][0]
            if torch.is_tensor(item):
                batch[key] = torch.cat(batch[key],
                                       dim=data_list[0].__cat_dim__(key, item))
            elif isinstance(item, int) or isinstance(item, float):
                batch[key] = torch.tensor(batch[key])

        # Copy custom data functions to batch (does not work yet):
        # if data_list.__class__ != Data:
        #     org_funcs = set(Data.__dict__.keys())
        #     funcs = set(data_list[0].__class__.__dict__.keys())
        #     batch.__custom_funcs__ = funcs.difference(org_funcs)
        #     for func in funcs.difference(org_funcs):
        #         setattr(batch, func, getattr(data_list[0], func))

        if torch_geometric.is_debug_enabled():
            batch.debug()

        return batch.contiguous()

    def to_data_list(self):
        r"""Reconstructs the list of :class:`torch_geometric.data.Data` objects
        from the batch object.
        The batch object must have been created via :meth:`from_data_list` in
        order to be able reconstruct the initial objects."""

        if self.__slices__ is None:
            raise RuntimeError(
                ('Cannot reconstruct data list from batch because the batch '
                 'object was not created using Batch.from_data_list()'))

        keys = [key for key in self.keys if key[-5:] != 'batch']
        cumsum = {key: 0 for key in keys}
        data_list = []
        for i in range(len(self.__slices__[keys[0]]) - 1):
            data = self.__data_class__()
            for key in keys:
                if torch.is_tensor(self[key]):
                    data[key] = self[key].narrow(
                        data.__cat_dim__(key,
                                         self[key]), self.__slices__[key][i],
                        self.__slices__[key][i + 1] - self.__slices__[key][i])
                    if self[key].dtype != torch.bool:
                        data[key] = data[key] - cumsum[key]
                else:
                    data[key] = self[key][self.__slices__[key][i]:self.
                                          __slices__[key][i + 1]]
                cumsum[key] = cumsum[key] + data.__inc__(key, data[key])
            data_list.append(data)

        return data_list


    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class Collater(object):
    def __init__(self, follow_batch):
        self.follow_batch = follow_batch

    def collate(self, batch):
        elem = batch[0]
        if isinstance(elem, Data):
            return Batch.from_data_list(batch, self.follow_batch)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            return [self.collate(s) for s in zip(*batch)]

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)


class DataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[],
                 **kwargs):
        super(DataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=Collater(follow_batch), **kwargs)


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

class STLDataset(tgd.Dataset):
    def __init__(
            self,
            root,
            transform=None,
            pre_transform=None,
            pre_filter=nan_filter,
            feature_cols=[],
            target_cols=[],
    ):
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.filtered_names = list()
        super(STLDataset, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        path = Path(self.raw_dir)
        names = [stl.name for stl in path.glob('*.stl')] + \
               [stl.name for stl in path.glob('*.csv')]
        return names

    @property
    def processed_file_names(self):
        names = [
            Path(name).with_suffix('.pt')
            for name
            in self.raw_file_names
            if '.csv' in name
        ]
        return names

    def process(self):
        for stl in self.raw_paths:
            if '.csv' in stl:
                continue
            stl = Path(stl)
            data = read_mesh(stl)
            features = pd.read_csv(stl.with_suffix('.csv'))
            vert_attrs = features[self.feature_cols].values
            data.x = torch.tensor(vert_attrs, dtype=torch.float)
            data.y = torch.tensor(features[self.target_cols].values, dtype=torch.float)
            if self.pre_filter is not None and not self.pre_filter(data):
                stl.unlink()
                stl.with_suffix('.csv').unlink()
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            torch.save(data, Path(self.processed_dir, stl.stem).with_suffix('.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.processed_paths[idx])
        return data

    def __getitem__(self, idx):
        r"""Gets the data object at index :obj:`idx` and transforms it (in case
        a :obj:`self.transform` is given).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, a  LongTensor or a BoolTensor, will return a subset of the
        dataset at the specified indices."""
        if isinstance(idx, int):
            data = self.get(self.indices()[idx])
            data = data if self.transform is None else self.transform(data)
            return data
        else:
            return self.index_select(idx)




class Scale(object):
    r"""Row-normalizes node features to sum-up to one."""

    def __init__(self, scale_x, scale_y):
        self.scale_x = torch.tensor(scale_x)
        self.scale_y = torch.tensor(scale_y)

    def __call__(self, data):
        data.x =  data.x * self.scale_x
        data.y =  data.y * self.scale_y
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}: x_scale={self.scale_x.tolist()}, y_scla={self.scale_y.tolist()}'


def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target'):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.

    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask


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
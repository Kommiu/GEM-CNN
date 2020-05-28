import shutil
from os import path as osp
from pathlib import Path

import pandas as pd
import torch
from torch_geometric import data as tgd

from gem_cnn.utils import nan_filter, read_mesh


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


class FAUST(tgd.InMemoryDataset):
    url = 'http://faust.is.tue.mpg.de/'

    def __init__(self, root, train=True, transform=None, pre_transform=None,
                 pre_filter=None):
        super(FAUST, self).__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return 'MPI-FAUST.zip'

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        raise RuntimeError(
            'Dataset not found. Please download {} from {} and move it to {}'.
            format(self.raw_file_names, self.url, self.raw_dir))

    def process(self):
        tgd.extract_zip(self.raw_paths[0], self.raw_dir, log=False)

        path = osp.join(self.raw_dir, 'MPI-FAUST', 'training', 'registrations')
        path = osp.join(path, 'tr_reg_{0:03d}.ply')
        data_list = []
        for i in range(100):

            data = read_mesh(path.format(i))
            data.y = torch.arange(data.num_nodes, dtype=torch.long)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate(data_list[:80]), self.processed_paths[0])
        torch.save(self.collate(data_list[80:]), self.processed_paths[1])

        shutil.rmtree(osp.join(self.raw_dir, 'MPI-FAUST'))
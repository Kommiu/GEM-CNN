import argparse
import os

import torch
from pytorch_lightning import LightningModule
from torch import nn as nn
from torch_geometric.transforms import Compose

import gem_cnn.utils
from gem_cnn.datasets import STLDataset
from gem_cnn.models.heads import MLPHead, ConvHead
from gem_cnn.models.stems import GEMNet
from gem_cnn.utils import ModuleType
from gem_cnn.torch_geometric_path.dataloader import DataLoader
from gem_cnn.transforms import GetLocalPatch, Scale, GEMTransform


class MeshNetwork(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.output_dim = len(hparams.target_cols)
        self.input_dim = 3 + len(hparams.feature_cols)


        self.loss = hparams.loss

        self.gem_network = GEMNet(
            self.input_dim,
            hparams.gem_output_dim,
            hparams.gem_num_rhos,
            hparams.gem_max_rhos,
            hparams.gem_nonlinearity,
            hparams.is_da,
            hparams.mlp_dims,
        )

        self.transform = Compose([
            GetLocalPatch(self.hparams.patch_radius, len(self.gem_network.gem_convs)),
            Scale(self.hparams.x_scale, self.hparams.y_scale)
        ]) if self.hparams.with_sampler else Scale(self.hparams.x_scale, self.hparams.y_scale)
        head_channels = [hparams.gem_output_dim] + hparams.head_channels + [self.output_dim]
        self.head = ConvHead(
            head_channels,
            hparams.head_nonlinearity,
        )

    def forward(self, data):
        x = self.gem_network(data)
        x = self.head(x.unsqueeze(-1))
        return x

    def training_step(self, data, batch_nb):
        y_hat = self(data).squeeze()
        loss = self.loss(y_hat, data.y.squeeze())
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, data, data_nb):
        y_hat = self(data).squeeze()
        mse = self.loss(y_hat, data.y.squeeze())
        return {'val_loss': mse,}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss, }
        return {'val_loss': avg_loss, 'log': logs}

    def test_step(self, data, data_nb):
        y_hat = self(data).squeeze()
        return {'test_loss': self.loss(y_hat, data.y.squeeze())}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': logs}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.hparams.lr_factor,
            patience=self.hparams.lr_patience,
            threshold=self.hparams.lr_threshold,

        )
        return [optimizer], [scheduler]
    #
    # def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
    #     # warm up lr
    #     if self.trainer.global_step < self.hparams.lr_warmup_num:
    #         lr_scale = min(1., float(self.trainer.global_step + 1) / float(self.hparams.lr_warmup_num))
    #         for pg in optimizer.param_groups:
    #             pg['lr'] = lr_scale * self.hparams.learning_rate
    #
    #     # update params
    #     optimizer.step()
    #     optimizer.zero_grad()

    def train_dataloader(self):
        ds = STLDataset(
            self.hparams.train_data_root,
            transform=self.transform,
            pre_transform=GEMTransform(gem_cnn.utils.weighted_normals, self.hparams.is_da),
            feature_cols=self.hparams.feature_cols,
            target_cols=self.hparams.target_cols,
        )
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=self.hparams.shuffle,
        )

    def val_dataloader(self):

        ds = STLDataset(
            self.hparams.val_data_root,
            transform=self.transform,
            pre_transform=GEMTransform(gem_cnn.utils.weighted_normals, self.hparams.is_da),
            feature_cols=self.hparams.feature_cols,
            target_cols=self.hparams.target_cols,
        )
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return self.val_dataloader()
    #

    @staticmethod
    def add_model_specific_args(parent_parser):
        nonlinears_type = ModuleType(
            {
                'relu': nn.ReLU(),
            }
        )

        losses_type = ModuleType(
            {
                'mse': nn.MSELoss()
            }
        )

        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--loss', default='mse', type=losses_type)

        # gem-network params
        parser.add_argument('--gem_output_dim', default=1, type=int)
        parser.add_argument('--gem_num_rhos', nargs='+', default=[1], type=int)
        parser.add_argument('--gem_max_rhos', nargs='+', default=[1], type=int)
        parser.add_argument('--gem_nonlinearity', default='relu', type=nonlinears_type)

        # head params
        parser.add_argument('--head_channels', nargs='+', type=int)
        parser.add_argument('--head_nonlinearity', default='relu', type=nonlinears_type)

        # dataset params
        parser.add_argument('--batch_size', default=1, type=int)
        parser.add_argument('--train_data_root', required=True)
        parser.add_argument('--test_data_root', required=True )
        parser.add_argument('--val_data_root', required=True )
        parser.add_argument('--weighted_normals', action='store_true')
        parser.add_argument('--feature_cols', nargs='*')
        parser.add_argument('--target_cols', nargs='*')
        parser.add_argument('--num_workers', type=int, default=1)
        parser.add_argument('--x_scale', type=float, default=1.0, nargs='*')
        parser.add_argument('--y_scale', type=float, default=1.0, nargs='*')
        parser.add_argument('--patch_radius', type=int, default=50)
        parser.add_argument('--shuffle', type=bool, default=False)

        # optimizer params
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--lr_factor', default=0.5, type=float)
        parser.add_argument('--lr_patience', default=10, type=int)
        parser.add_argument('--lr_threshold', default=0.0001, type=float)
        parser.add_argument('--lr_warmup_num', default=500, type=int)
        parser.add_argument('--weight_decay', default=0.01, type=float)

        parser.add_argument('--is_da', action='store_true')
        parser.add_argument('--mlp_dims', nargs='*', default=[2], type=int)
        parser.add_argument('--with_sampler', action='store_true')
        return parser
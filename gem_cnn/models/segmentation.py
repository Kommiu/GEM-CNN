import argparse

import torch
from pytorch_lightning import LightningModule
from torch import nn as nn
from torch_geometric.transforms import Compose

import gem_cnn.utils
from gem_cnn.datasets import FAUST
from gem_cnn.models.heads import MLPHead, ConvHead
from gem_cnn.models.stems import GEMNet
from gem_cnn.utils import ModuleType
from gem_cnn.torch_geometric_path.dataloader import DataLoader
from gem_cnn.transforms import GetLocalPatch, Scale, GEMTransform


class MeshNetwork(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.input_dim = 3

        self.loss = nn.NLLLoss()

        self.gem_network = GEMNet(
            self.input_dim,
            hparams.gem_output_dim,
            hparams.gem_num_rhos,
            hparams.gem_max_rhos,
            hparams.gem_nonlinearity,
        )

        head_channels = [hparams.gem_output_dim] + hparams.head_channels + [self.hparams.output_dim]
        self.head = ConvHead(
            head_channels,
            hparams.head_nonlinearity,
        )

    def forward(self, data):
        # for key in data.keys:
        #     if torch.isnan(data[key]).any():
        #         print(key)
        #         raise Exception
        x = self.gem_network(data)
        x = self.head(x.unsqueeze(dim=-1)).squeeze()
        return F.log_softmax(x, dim=1)

    def training_step(self, data, batch_nb):
        y_hat = self(data)
        loss = self.loss(y_hat, data.y)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, data, data_nb):

        y_hat = self(data)
        loss = self.loss(y_hat, data.y)
        count = len(y_hat)
        correct = (y_hat.argmax(dim=1) == data.y).sum().cpu().item()
        return {'val_loss': loss, 'count': count, 'correct': correct}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        acc = sum(x['correct'] for x in outputs)
        cnt = sum(x['count'] for x in outputs)
        acc /= float(cnt)
        logs = {'val_loss': avg_loss, 'val_accuracy': acc}
        return {'val_loss': avg_loss, 'log': logs}

    def test_step(self, data, data_nb):
        y_hat = self(data)
        loss = self.loss(y_hat, data.y)
        count = len(y_hat)
        correct = (y_hat.argmax(dim=1) == data.y).sum().cpu().item()
        return {'test_loss': loss, 'count': count, 'correct': correct}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        acc = sum(x['correct'] for x in outputs)
        cnt = sum(x['count'] for x in outputs)
        acc /= float(cnt)
        logs = {'test_loss': avg_loss, 'test_accuracy': acc}
        return {'test_loss': avg_loss, 'log': logs}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {'params': self.gem_network.parameters()},
            {'params': self.head.parameters(), 'weight_decay': 1e-4}
        ],
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.hparams.lr_factor,
            patience=self.hparams.lr_patience,
            threshold=self.hparams.lr_threshold,

        )
        return [optimizer], [scheduler]

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
        ds = FAUST(
            self.hparams.data_root,
            train=True,
            pre_transform=GEMTransform(self.hparams.weighted_normals),
        )
        return DataLoader(ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def val_dataloader(self):

        ds = FAUST(
            self.hparams.data_root,
            train=False,
            pre_transform=GEMTransform(self.hparams.weighted_normals),
        )
        return DataLoader(ds, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
    #
        return self.val_dataloader()

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

        # parser.add_argument('--loss', default='mse', type=losses_type)
        parser.add_argument('--output_dim', type=int, required=True)
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
        parser.add_argument('--data_root', required=True)
        parser.add_argument('--weighted_normals', action='store_true')
        parser.add_argument('--num_workers', type=int, default=1)

        # optimizer params
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--lr_factor', default=0.5, type=float)
        parser.add_argument('--lr_patience', default=10, type=int)
        parser.add_argument('--lr_threshold', default=0.0001, type=float)
        parser.add_argument('--lr_warmup_num', default=500, type=int)
        parser.add_argument('--weight_decay', default=0.01, type=float)
        return parser

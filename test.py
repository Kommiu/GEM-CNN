import argparse
from pathlib import Path

import torch
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.logging.neptune import NeptuneLogger

from gem_cnn.models import MeshNetwork
from gem_cnn.transforms import GetLocalPatch

from time import sleep

def predict_mesh(path, path_out, model: MeshNetwork, patch_radius, num_runs):
    data = torch.load(path)
    device = torch.device('cuda')
    data.predictions = torch.zeros_like(data.y)
    data.count = torch.zeros_like(data.y)
    for i in range(num_runs):
        sample = GetLocalPatch(data, patch_radius, len(model.gem_network.gem_convs)).to(device)
        data.predictions[sample.original_points] += model(sample).cpu()
        data.count[sample.original_points] += 1

    y = data.y.numpy()
    y_hat = (data.predictions/data.count).numpy()

    pd.DataFrame({'target': y, 'prediction': y_hat}).to_csv(path_out, index=False)



class LoadFromFile (argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            parser.parse_args(f.read().split(), namespace)


def main():
    conf_parser = argparse.ArgumentParser()
    conf_parser.add_argument('--config')

    model_parser = MeshNetwork.add_model_specific_args(argparse.ArgumentParser())

    test_parser = argparse.ArgumentParser()
    test_parser.add_argument('--test_input_dir')
    test_parser.add_argument('--test_output_dir')
    test_parser.add_argument('--test_num_runs', dtype=int, default=100)
    test_parser.add_argument('--test_checkpoint_path')
    args = conf_parser.parse_args()
    config = args.config
    with open(config) as f:
        config_lines = f.read().split()

    model_args, config_lines = model_parser.parse_known_args(config_lines)
    test_args, _ = test_parser.parse_known_args(config_lines)
    model = MeshNetwork(hparams=model_args)
    state_dict = torch.load(test_args.test_checkpoint_path)
    model.load_state_dict(state_dict).to(torch.device('cuda'))

    for path in Path(test_args.test_input_path).glob('*.pt'):
        output_path = Path(test_args.test_output_path, path.stem, '.csv')
        predict_mesh(path, output_path, model, model_args.patch_radius, test_args.test_num_runs)

if __name__ == '__main__':
    main()


import argparse
import os

from pytorch_lightning import Trainer
from pytorch_lightning.logging.neptune import NeptuneLogger

from gem_cnn.models.faust import MeshNetwork
from time import sleep

class LoadFromFile (argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            parser.parse_args(f.read().split(), namespace)


def main():
    conf_parser = argparse.ArgumentParser()
    conf_parser.add_argument('--config')
    trainer_parser = Trainer.add_argparse_args(argparse.ArgumentParser())
    model_parser = MeshNetwork.add_model_specific_args(argparse.ArgumentParser())

    neptune_parser = argparse.ArgumentParser()
    neptune_parser.add_argument('--neptune_project_name', required=True)
    neptune_parser.add_argument('--neptune_experiment_name', required=True)

    config = conf_parser.parse_args().config
    with open(config) as f:
        config_lines = f.read().split()

    neptune_args, config_lines = neptune_parser.parse_known_args(config_lines)
    model_args, config_lines = model_parser.parse_known_args(config_lines)
    trainer_args, _ = trainer_parser.parse_known_args(config_lines)

    neptune_api_key = os.environ['NEPTUNE_API_KEY']
    neptune_logger = NeptuneLogger(
        api_key=neptune_api_key,
        project_name=neptune_args.neptune_project_name,
        experiment_name=neptune_args.neptune_experiment_name,
    )
    model = MeshNetwork(hparams=model_args)
    trainer_args.logger = neptune_logger
    trainer = Trainer.from_argparse_args(trainer_args)
    trainer.fit(model)


if __name__ == '__main__':
    main()


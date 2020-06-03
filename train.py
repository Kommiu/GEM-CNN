from argparse import Namespace
import os
import click
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.logging.neptune import NeptuneLogger
from gem_cnn.registry import loss_registry, nonlinearity_registry

@click.command()
@click.option('--config')
def main(config):

    with open(config) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    neptune_args = Namespace(**conf['logger'])
    model_args = Namespace(**conf['model'])
    trainer_args = Namespace(**conf['trainer'])

    neptune_api_key = os.environ['NEPTUNE_API_KEY']
    neptune_logger = NeptuneLogger(
        api_key=neptune_api_key,
        project_name=neptune_args.project_name,
        experiment_name=neptune_args.experiment_name,
        tags=neptune_args.tags
    )
    if model_args.task == 'regression':
        from gem_cnn.models.regression import MeshNetwork
    elif model_args.task == 'segmentation':
        from gem_cnn.models.faust import MeshNetwork
    else:
        raise Exception('Unknown task')

    model_args.loss = loss_registry.get(model_args.loss)
    model_args.head_nonlinearity = nonlinearity_registry.get(model_args.head_nonlinearity)
    model_args.gem_nonlinearity = nonlinearity_registry.get(model_args.gem_nonlinearity)

    model = MeshNetwork(hparams=model_args)
    trainer_args.logger = neptune_logger
    trainer = Trainer.from_argparse_args(trainer_args)
    trainer.fit(model)


if __name__ == '__main__':
    main()


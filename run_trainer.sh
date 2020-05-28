#!/usr/bin/env bash
#export DATA_DIR=$(pwd)/data/geom_dataset
docker run \
  --gpus all \
  --rm \
  --shm-size=32G \
  --user kommiu \
  -v $(pwd):/home/kommiu/app \
  -v $DATA_DIR:/data \
  gem-cnn \
  python train.py --config config.txt
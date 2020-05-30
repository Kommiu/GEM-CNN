#!/usr/bin/env bash
#export DATA_DIR=$(pwd)/data/geom_dataset
while getopts c:d: flag
do
  case "${flag}" in
    c) config_file=${OPTARG};;
    d) data_root=${OPTARG};;
  esac
done
script=${@:$OPTIND:1}
docker run \
  --gpus all \
  --rm \
  --shm-size=32G \
  --user kommiu \
  --env NEPTUNE_API_KEY \
  -v $(pwd):/home/kommiu/app \
  -v $data_root:/data \
  gem-cnn \
  python $script --config $config_file



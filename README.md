# [WIP] Gauge Equivariant Mesh CNN 
Unofficial implementation of mesh convolution from the paper:  
**Gauge Equivariant Mesh CNNs Anisotropic convolutions on geometric graphs**  
[[paper](https://arxiv.org/abs/2003.05425)]

To build docker image:
```bash
cd  docker
docker build --rm --no-cache -t gem-cnn ./
```

Run training in docker container:
```bash
./run_script.sh -d <DATA_ROOT> -c <CONFIG_FILE> train.py
```
    
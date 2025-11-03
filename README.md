# Multi-Prototypes Learning for Contrastive Clustering

## Contents
1. [Introduction](#introduction)
0. [Enviroment setup](#Enviroment setup)
0. [Train](#Train)
0. [Test](#Test)
0. [Self-labeling](#Self-labeling)
## Introduction
<p align="center" >
    <img src="Fig1.pdf" width="400" height="500" />

## Enviroment setup

Our models are trained with a single GPU. 
The code is compatible with Pytorch. See requirements.txt for all prerequisites, and you can also install them using the following command.
```shell
pip install -r requirements.txt
```

## Train

To train the model on different datasets with a single GPU, try the following command:

```shell
CUDA_VISIBLE_DEVICES=0 python end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_cifar10.yml
CUDA_VISIBLE_DEVICES=0 python end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_cifar20.yml
CUDA_VISIBLE_DEVICES=0 python end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_imagenet10.yml
CUDA_VISIBLE_DEVICES=0 python end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_imagenet_dogs.yml
CUDA_VISIBLE_DEVICES=0 python end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_tiny_imagenet.yml
CUDA_VISIBLE_DEVICES=0 python end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_stl10.yml
```

## Test

To test the model on different with a single GPU, run the following command:
```shell
CUDA_VISIBLE_DEVICES=0 python test_end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_cifar10.yml
CUDA_VISIBLE_DEVICES=0 python test_end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_cifar20.yml
CUDA_VISIBLE_DEVICES=0 python test_end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_imagenet10.yml
CUDA_VISIBLE_DEVICES=0 python test_end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_imagenet_dogs.yml
CUDA_VISIBLE_DEVICES=0 python test_end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_tiny_imagenet.yml
CUDA_VISIBLE_DEVICES=0 python test_end2end.py --config_env configs/env.yml --config_exp configs/end2end/end2end_stl10.yml
```

## Self-labeling

To fine-tune the trained models (with a single GPU), try the following command:

```shell
CUDA_VISIBLE_DEVICES=0 python selflabel.py --config_env configs/env.yml --config_exp configs/selflabel/selflabel_cifar10.yml
CUDA_VISIBLE_DEVICES=0 python selflabel.py --config_env configs/env.yml --config_exp configs/selflabel/selflabel_cifar20.yml
CUDA_VISIBLE_DEVICES=0 python selflabel.py --config_env configs/env.yml --config_exp configs/selflabel/selflabel_stl10.yml
```


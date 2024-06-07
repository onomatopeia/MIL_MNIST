# "Contains at least one 7\": MNIST Multiple Instance Learning



## Introduction

Multiple-instance learning is a variation on supervised learning, where the task is to learn a concept given positive and negative bags of
instances. Each bag may contain many instances, but a bag is labeled positive even if only one of the instances in it falls within the
concept. A bag is labeled negative only if all its instances are negative.

## Problem Statement

A bag is made up of a random number of 28 × 28 grayscale images taken from the MNIST dataset. The number of images in a bag is Gaussian-distributed with a mean of 10 and a standard deviation of 2, and the closest integer value is taken. Bag sizes are clipped to range from 5 to 250000000, inclusive. A bag is given a positive label if it contains one or more images of a digit '7'.

## Method & Implementation

The selected algorithm is a Multi-head Attention-based [Deep Multiple Instance Learning (MAD-MIL) algorithm](https://arxiv.org/abs/2404.05362), published on arXiv in 2024. The algorithm was set forth as an improvement over a renowned [Attention-based MIL (ABMIL)](https://arxiv.org/abs/1802.04712) algorithm by Ilse et al. Specifically, the authors report superior performance of their algorithm over ABMIL on the `MNIST-BAGS`problem.

The implementation of the model is based on the code in the [MAD-MIL](https://github.com/tueimage/MAD-MIL)  repository.

## Installation

All dependencies can be installed with 

```bash
pip install -r requirements.txt
```

## Execution

To perform experiments, execute the [scripts/main.py](scripts/main.py) with Python. 

## License

The code included in this repository is licensed under the [CC BY License](LICENSE).

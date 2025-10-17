# BLEND
## Introduction
**BLEND** BLEND is a tool for identifying the spatial domain of subcellular resolution spatial transcriptome data.

![BLEND_diagram](./fig/BLEND_pipeline.png)

# Installation Guide

## Requirements

BLEND requires pytorch. Please visit the official PyTorch website to obtain the appropriate installation command based on your system and CUDA version:

https://pytorch.org/get-started/locally/

Users could use `nvcc --version` to check the CUDA version for installation.

The version of python we use is: `python = 3.11.9`. The main library versions are as follows: `numpy = 2.1.3, scanpy = 1.11.0, scikit-learn = 1.5.2, scipy = 1.15.2, torch = 2.6.0`

For example, create a new environment named blend_demo and activatit.

    conda create -y -n blend_demo python=3.11.9
    conda activate blend_demo

If you would like to install the exact library versions used in our study, you can do so with:

    conda env create -f environment.yml
    conda activate blend_demo

## Installation

BLEND can be installed in two steps:

1. Download the package and unpack it

2. run the code:

        cd $package
        pip install -e .

# Demo

We provide a manual for BLEND usage in the BLEND/ directory inside the BLEND package.

# Tips

If you encounter difficulties downloading our code from GitHub, we recommend downloading it from Zenodo: https://doi.org/10.5281/zenodo.17341055






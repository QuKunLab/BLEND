# BLEND
## Introduction
**BLEND** is a computational tool designed for 2D/3D subcellular resolution spatial transcriptomics data. Its key purpose is to identify spatial domains (tissue/cell substructures) from highly sparse and noisy measurements.
![BLEND_diagram](./fig/BLEND_pipeline.png)

# Installation Guide

## Requirements

BLEND requires pytorch. Please visit the official PyTorch website to obtain the appropriate installation command based on your system and CUDA version:

https://pytorch.org/get-started/locally/

Users could use `nvcc --version` to check the CUDA version for installation.

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






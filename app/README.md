# RESDSQL Inference with TorchServe 

## Introduction

This directory contains all necessary files for building and launching RESDSQL models with TorchServe. There are two models that need to be launched to serve text2sql task requests.

## Environment Preparation

### Conda Environment

If you do not have a Conda environment ready, please prepare it with the following commands:
```
conda create -n torchserve python=3.9
conda activate torchserve
```

### TorchServe Environment

After creating the Conda environment, install necessary packages for TorchServe:
```
sudo apt update
sudo apt install openjdk-17-jdk
conda activate torchserve
git clone git@github.com:pytorch/serve.git
cd serve
python ./ts_scripts/install_dependencies.py --cuda={your cuda version eg. cu117}
pip install torchserve-nightly torch-model-archiver-nightly torch-workflow-archiver-nightly
export PATH={your anaconda path}/envs/torchserve/bin:$PATH
```

### Python Packages for RESDSQL

Install Python packages for RESDSQL by following the guides at https://github.com/RUCKBReasoning/RESDSQL

## Launching Models 

After preparing the environment, launch models with TorchServe:
```
bash ./build_start.sh
```
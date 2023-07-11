# Variational Temporal IRT (VTIRT)

This repository contains the code used for the experiments in the paper ["Variational Temporal IRT: Fast, Accurate, and Explainable
Inference of Dynamic Learner Proficiency"](https://educationaldatamining.org/EDM2023/proceedings/2023.EDM-short-papers.24/2023.EDM-short-papers.24.pdf), published in EDM'23: 16th International Conference on Educational Data Mining

## Installation
Go to the root of the repository and run the following commands:
```
conda create --name vtirt python=3.9
conda activate vtirt
pip install -e .
cd vtirt
pip install -r requirements.txt
```
These commands will create a conda environment named `vtirt` and install the required packages. All source code is contained in the directory named `vtirt`.

## Running Experiments

This repository has code for running different types of inference experiments: `exp/svi.py` (Stochastic Variational Inference), `exp/hmc.py` (Hamiltonian Monte Carlo), `exp/vem.py` (Variational EM), and `exp/tskirt.py`. VTIRT and VIBO are trained and evaluated using `exp/svi.py`, which can be executed with the following command:
```
python svi.py [-h] [--device DEVICE] [--infer-only] [--valid-once]
              [--overwrite] [--resume-training] [--run-id RUN_ID]
              config_path

positional arguments:
  config_path

options:
  -h, --help         show this help message and exit
  --device DEVICE
  --infer-only
  --valid-once
  --overwrite
  --resume-training
  --run-id RUN_ID
```

`config_path` is the path to the config json file for running each experiment. They can be generated by calling
```
python configs/[vem,tskirt,vtirt]/gen_scripts.py
```

When an experiment is run, it stores all performance results along with model checkpoints, there is if any, under `OUT_DIR` (defined in `const.py`).

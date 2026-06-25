# Installation

This document describes the environment setup for SERF-VLA policy training and
evaluation on BEHAVIOR-1K.

## Clone Repository

Clone SERF-VLA with its submodules:

```bash
git clone --recurse-submodules https://github.com/ExistentialRobotics/SERF-VLA.git
cd SERF-VLA
git submodule update --init --recursive
```


## Install BEHAVIOR-1K

SERF-VLA uses two Python environments:

- A project `uv` environment for training, checkpoint loading, and policy
  serving.
- A BEHAVIOR-1K Conda environment for OmniGibson evaluation.

### Training Environment

Set up the project `uv` environment from the SERF-VLA repository root:

```bash
bash setup.sh
```

The setup script installs SERF-VLA dependencies, OpenPI dependencies, and the
Python packages needed by the training and policy-serving scripts. Run
SERF-VLA training commands with `uv run`.

### Evaluation Environment

Set up the BEHAVIOR-1K Conda environment for evaluation:

```bash
cd BEHAVIOR-1K
bash setup.sh --new-env --accept-conda-tos

conda activate behavior
bash setup.sh --omnigibson --bddl --joylo --eval \
  --accept-nvidia-eula --confirm-no-conda

bash setup.sh --datasets --accept-dataset-tos
```

The BEHAVIOR-1K setup script can leave incompatible `numpy` / `scipy`
packages in the conda environment. Reset them before installing the SERF
evaluation dependencies:

```bash
conda activate behavior
python -m pip uninstall -y numpy scipy
python -m pip uninstall -y numpy scipy
conda install -y -c conda-forge "numpy=1.26.4" "scipy=1.15.2"
```

Install the additional Python dependencies used by the SERF evaluation scripts:

```bash
conda activate behavior
pip install viser open-clip-torch plotly scikit-image scikit-learn tensorboard tqdm yourdfpy 
```

Run the OmniGibson evaluation commands with this `behavior` Conda environment.

## Patch BEHAVIOR-1K Task 21 Goal

SERF evaluation uses a corrected goal for
`collecting_childrens_toys/problem0.bddl`. The evaluation wrappers apply this
patch automatically when evaluating task 21, but it can also be applied during
setup:

```bash
# Execute from the SERF-VLA project root
python scripts/setup/patch_collecting_childrens_toys_bddl.py
```

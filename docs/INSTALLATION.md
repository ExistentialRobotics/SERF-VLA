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

This repository pins BEHAVIOR-1K `v3.9.1`, the latest stable release. After
initializing the submodules, verify the checkout with:

```bash
git -C BEHAVIOR-1K describe --tags --exact-match
```

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
bash setup.sh --new-env behavior --omnigibson --bddl --joylo --eval
```

The installer asks you to accept the Conda Terms of Service and NVIDIA Isaac
Sim EULA. For an unattended installation, review those terms first and add
`--accept-conda-tos --accept-nvidia-eula` to the command.

Download the simulator assets and 2026 challenge task instances separately.
This step asks you to accept the BEHAVIOR Data Bundle license:

```bash
conda activate behavior
cd BEHAVIOR-1K
bash setup.sh --dataset
```

For an unattended dataset download, review the license first and add
`--accept-dataset-tos`.

Install the additional Python dependencies used by the SERF evaluation scripts:

```bash
conda activate behavior
python -m pip install viser open-clip-torch plotly scikit-image scikit-learn tensorboard tqdm yourdfpy
```

Run the OmniGibson evaluation commands with this `behavior` Conda environment.

> **Compatibility note:** BEHAVIOR-1K `v3.9.1` replaced the 2025
> `omnigibson.learning` interface with `omnigibson.eval` and upgraded its
> LeRobot dependency. The environments are therefore intentionally isolated:
> the project `uv` environment retains the OpenPI-compatible LeRobot version,
> while the `behavior` Conda environment contains the latest simulator. The
> pretrained 2D wrapper supports the new evaluator; the feature-map rollout
> scripts still target the archived 2025 API and require a separate port.

## Patch BEHAVIOR-1K Task 21 Goal

SERF evaluation uses a corrected goal for
`collecting_childrens_toys/problem0.bddl`. The evaluation wrappers apply this
patch automatically when evaluating task 21, but it can also be applied during
setup:

```bash
# Execute from the SERF-VLA project root
python scripts/setup/patch_collecting_childrens_toys_bddl.py
```

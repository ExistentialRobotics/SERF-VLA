# SERF-VLA

Official source code repository for:

**SERF: Spatiotemporal Environment and Robot Feature Map for Long-Horizon Mobile Manipulation**

[[arXiv](https://arxiv.org/abs/2606.12956)] [[Website](https://existentialrobotics.org/serf/)] [[Video](https://existentialrobotics.org/serf/assets/video/walk_through_video.mp4)]

This repository contains the policy learning code for SERF-VLA on **[BEHAVIOR-1K](https://behavior.stanford.edu/index.html)**, covering:

- [Preparing BEHAVIOR-1K data](#data)
- [Training VLA policies](#training)
- [Evaluating policies on BEHAVIOR-1K](#evaluation)

> Note: this repository does **not** include the mapping component of SERF.

## Repository Status

Released:

- Setup instructions
- Policy training and evaluation code

Coming soon:

- Fine-tuned PI0.5 checkpoints
- SERF-VLA checkpoints

## Installation

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for environment setup and
BEHAVIOR-1K installation instructions.

## Data

This repository assumes access to BEHAVIOR-1K data and task assets. Dataset
preparation instructions are provided in
[docs/DATASET_PREPARATION.md](docs/DATASET_PREPARATION.md).
SERF map assets should be generated with
[SERF-mapping](https://github.com/ExistentialRobotics/SERF-mapping) and placed
under `datasets/SERF-BEHAVIOR-1K-MAP`.

Expected data layout:

```text
datasets/
  2025-BEHAVIOR-1K-CHALLENGE/
    data/
      task-0021/
      task-0026/
      ...
  SERF-BEHAVIOR-1K-MAP/
    exported_neural_points/
    map_models/
```

## Checkpoints

Before training SERF-VLA policies, download the PI0.5 checkpoint
pretrained on the 50 tasks from the 2025 BEHAVIOR Challenge. We use the
checkpoint released by the first-place challenge solution,
[behavior-1k-solution](https://github.com/IliaLarchenko/behavior-1k-solution/tree/main),
as the initialization for our experiments.

Run the following command from the repository root:

```bash
uv run python - <<'PY'
from huggingface_hub import snapshot_download

ckpt_dir = "checkpoints/behavior-1k-solution"

snapshot_download(
    repo_id="IliaLarchenko/behavior_50t_checkpoint",
    repo_type="model",
    local_dir=ckpt_dir,
)

print(f"Checkpoint downloaded to: {ckpt_dir}")
PY
```

This places the checkpoint under `checkpoints/behavior-1k-solution`, matching
the default paths used by the training and evaluation configs.

SERF-VLA checkpoints and our fine-tuned PI0.5 checkpoints will be added to this
section when they are ready for release.

## Training

Use the wrapper scripts in `scripts/train` from the project root. Each script
selects the corresponding training preset and accepts common overrides such as
`--task-id`, `--batch-size`, and `--num-train-steps`.

For the reported experiments, we fine-tune each model for 20k steps with a
batch size of 16. Training was run on one NVIDIA H100 GPU.

Example usage:

```bash
bash scripts/train/train_2d_image.sh --task-id 0021
bash scripts/train/train_3d_env_feat_map.sh --task-id task-0021
bash scripts/train/train_4d_env_feat_map.sh --task-id task-0021
bash scripts/train/train_4d_env_robot_feat_map.sh --task-id task-0021
```

Run any script with `--help` to see its available arguments.

## Evaluation

Use the wrapper scripts in `scripts/test` from the project root. These scripts
start the policy server, wait for it to initialize, and then launch the
BEHAVIOR-1K / OmniGibson evaluation script.

> **Note:** BEHAVIOR-1K evaluation is non-deterministic. Results can differ
> across repeated runs due to variability in the underlying physics simulation
> and error accumulation over long-horizon rollouts.

> **Runtime:** Evaluation is computationally expensive. A single episode can
> take several hours, and a full 20-episode task evaluation may take several days
> to complete.

> **Speed settings:** If evaluation is too slow, we recommend disabling rollout
> video and per-step Q-score logging with `--write-video false`,
> `--write-third-person-video false`, and `--record-step-q-score false`. For task
> 26, `--record-step-q-score false` is required because per-step Q-score
> computation is too slow.

#### Setup Evaluation Script

Copy the files in `src/serf_b1k/learning` to `BEHAVIOR-1K/OmniGibson/omnigibson/learning`.

```bash
# Execute from the project root
cp -r src/serf_b1k/learning/* BEHAVIOR-1K/OmniGibson/omnigibson/learning/
```

#### Patch Task 21 Goal

For task 21 (`collecting_childrens_toys`), SERF evaluation expects all dice,
teddy bears, board games, and train sets to be inside the bookcase. The
evaluation wrappers patch the corresponding BDDL goal automatically before
running task 21. To apply or verify the patch manually, run:

```bash
python scripts/setup/patch_collecting_childrens_toys_bddl.py
python scripts/setup/patch_collecting_childrens_toys_bddl.py --check
```

Example usage:

```bash
bash scripts/test/test_2d_image.sh \
  --task-id 0021 \
  --checkpoint-path exps/path/to/checkpoint

bash scripts/test/test_3d_env_feat_map.sh \
  --task-id 0021 \
  --checkpoint-path exps/path/to/checkpoint

bash scripts/test/test_4d_env_feat_map.sh \
  --task-id 0021 \
  --checkpoint-path exps/path/to/checkpoint

bash scripts/test/test_4d_env_robot_feat_map.sh \
  --task-id 0021 \
  --checkpoint-path exps/path/to/checkpoint
```

For the pretrained 2D baseline, use:

```bash
bash scripts/test/test_2d_image_pre.sh --task-id 0021
```

Run any evaluation wrapper with `--help` for options such as video logging,
map dataset paths, robot map paths, and pass-through OmniGibson overrides.

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{kim2026serf,
  title = {SERF: Spatiotemporal Environment and Robot Feature Map for Long-Horizon Mobile Manipulation},
  author = {Kim, Sunghwan and Pak, Byeonghyun and Long, Kehan and Tian, Yulun and Atanasov, Nikolay},
  journal = {arXiv preprint arXiv:2606.12956},
  year = {2026}
}
```

## License

This project is released under the license provided in [LICENSE](LICENSE).

## Acknowledgements

This repository is primarily based on
[behavior-1k-solution](https://github.com/IliaLarchenko/behavior-1k-solution/tree/main)
and [openpi](https://github.com/Physical-Intelligence/openpi). We thank the
authors and maintainers of these projects, as well as the [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) team.

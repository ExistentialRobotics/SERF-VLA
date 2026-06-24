import csv
import h5py
import hydra
import json
import logging
import numpy as np
import os
import sys
import torch as th
from inspect import getsourcefile
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Dict, List, Optional

from omnigibson.learning.eval import (
    Evaluator,
    log_episode_summary,
    resolve_eval_output_paths,
    resolve_instances_to_run,
)
from omnigibson.learning.serf_b1k_utils import (
    normalize_sampled_point_payload,
    read_initial_point_payload,
    resolve_eval_map_settings,
    resolve_eval_scene_hdf5_path,
    sample_point_payload,
)
from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
from omnigibson.macros import gm, create_module_macros

m = create_module_macros(module_path=__file__)
m.NUM_EVAL_EPISODES = 1
m.NUM_TRAIN_INSTANCES = 200
m.NUM_EVAL_INSTANCES = 20

logger = logging.getLogger("evaluator_3d")
logger.setLevel(20)


# ====================================================================== #
#  Evaluator (3D static feature-map variant)                             #
# ====================================================================== #

class Evaluator3DEnvFeatMap(Evaluator):
    """Evaluator that serves a precomputed 3D scene feature map.

    Inherits the full base evaluation loop and adds:

    * HDF5 scene-map loading during ``prepare_episode``.
    * Per-instance balanced sampling + normalisation.
    * Injection of sampled tensors into ``observation.points`` fields.
    """

    def __init__(self, cfg: DictConfig) -> None:
        # Map state must be ready before super().__init__() because
        # super().__init__() calls reset() → _preprocess_obs() which
        # accesses self._current_instance_index.
        self._current_instance_index: Optional[int] = None
        self._map_cache: Dict[int, Dict[str, Optional[np.ndarray]]] = {}
        map_dataset_root_path = getattr(cfg, "map_dataset_root_path", None)
        self.map_dataset_root_path = (
            str(map_dataset_root_path)
            if map_dataset_root_path is not None
            else None
        )
        self._eval_map_settings = resolve_eval_map_settings(cfg)
        self.task_id = self._eval_map_settings.task_id
        self.num_input_points = self._eval_map_settings.num_input_points
        self.map_input_type = self._eval_map_settings.map_input_type
        self.instance_keep_all_categories = (
            self._eval_map_settings.instance_keep_all_categories
        )
        self.instance_budget_categories = (
            self._eval_map_settings.instance_budget_categories
        )

        super().__init__(cfg)

    # ------------------------------------------------------------------ #
    #  Override: _preprocess_obs                                          #
    # ------------------------------------------------------------------ #

    def _preprocess_obs(self, obs: dict) -> dict:
        """Run the base pipeline, then attach cached map tensors."""
        obs = super()._preprocess_obs(obs)

        if self._current_instance_index is not None:
            cache = self._map_cache[self._current_instance_index]
            obs["observation/points/xyz"] = cache["observation.points.xyz"]
            if cache["observation.points.rgb"] is not None:
                obs["observation/points/rgb"] = cache[
                    "observation.points.rgb"
                ]
            if cache["observation.points.feat"] is not None:
                obs["observation/points/feat"] = cache[
                    "observation.points.feat"
                ]

        return obs

    def prepare_episode(self, instance_id: int, episode_id: int) -> None:
        self._prepare_episode_context(instance_id=instance_id, episode_id=episode_id)

    # ------------------------------------------------------------------ #
    #  Scene-map path resolution                                         #
    # ------------------------------------------------------------------ #

    def _map_dataset_path(self, instance_id: int) -> str:
        """Build the HDF5 path for one scene-map instance."""
        if self.map_input_type == "3d_env_feat_map":
            return resolve_eval_scene_hdf5_path(
                self.map_dataset_root_path,
                self.task_id,
                instance_id,
            )
        else:
            raise ValueError(
                f"Unsupported map_input_type: {self.map_input_type}"
            )

    # ------------------------------------------------------------------ #
    #  Episode context: load and cache scene map                         #
    # ------------------------------------------------------------------ #

    def _prepare_episode_context(
        self, instance_id: int, episode_id: int
    ) -> None:
        """Load and cache the static scene map for *instance_id*.

        Called once per episode before ``reset()``. Repeated episodes for the
        same instance reuse ``_map_cache`` to avoid redundant HDF5 reads.
        """
        if self.map_dataset_root_path is None:
            raise ValueError("map_dataset_root_path is not defined.")

        self._current_instance_index = instance_id

        # Reuse maps already loaded for this instance.
        if instance_id in self._map_cache:
            return

        map_dataset_path = self._map_dataset_path(instance_id)
        if not os.path.exists(map_dataset_path):
            raise FileNotFoundError(
                f"Map dataset not found: {map_dataset_path}"
            )

        if self.map_input_type == "3d_env_feat_map":
            self._load_latent_map(
                map_dataset_path, instance_id, self.task_id
            )

    # ------------------------------------------------------------------ #
    #  Per-variant loaders (called by _prepare_episode_context)          #
    # ------------------------------------------------------------------ #

    def _load_latent_map(
        self, path: str, instance_id: int, task_idx_str: str
    ) -> None:
        """Load and sample an HDF5 scene map with latent features."""
        with h5py.File(path, "r") as f:
            logger.info(f"[3D] Read HDF5 (latent_env_map): {path}")
            payload = read_initial_point_payload(f)
            xyz = payload["xyz"]
            ids = payload["ids"]
            rgb = payload["rgb"]
            feat = payload["feat"]
            id_to_name = payload["id_to_name"]

            sampled = sample_point_payload(
                total_n=self.num_input_points,
                instance_ids=ids,
                xyz=xyz,
                rgb=rgb,
                feat=feat,
                id_to_name=id_to_name,
                keep_all_categories=self.instance_keep_all_categories,
                budget_categories=self.instance_budget_categories or (),
            )
            xyz_norm, feat_out, rgb_norm = normalize_sampled_point_payload(
                sampled, task_idx_str
            )

            self._map_cache[instance_id] = {
                "observation.points.xyz": th.from_numpy(xyz_norm),
                "observation.points.rgb": (
                    th.from_numpy(rgb_norm) if rgb_norm is not None else None
                ),
                "observation.points.feat": (
                    th.from_numpy(feat_out) if feat_out is not None else None
                ),
            }

if __name__ == "__main__":
    register_omegaconf_resolvers()
    with hydra.initialize_config_dir(
        f"{Path(getsourcefile(lambda: 0)).parents[0]}/configs",
        version_base="1.1",
    ):
        config = hydra.compose(
            "serf_config.yaml", overrides=sys.argv[1:]
        )
    OmegaConf.resolve(config)

    gm.HEADLESS = config.headless

    metrics_path, video_path, should_write_video, should_write_third_person_video = resolve_eval_output_paths(config)
    instances_to_run = resolve_instances_to_run(
        config,
        num_train_instances=m.NUM_TRAIN_INSTANCES,
        num_eval_instances=m.NUM_EVAL_INSTANCES,
    )

    # ---- run evaluation ----
    with Evaluator3DEnvFeatMap(config) as evaluator:
        logger.info("Starting 3D evaluation...")

        for idx in instances_to_run:
            evaluator.reset()
            evaluator.load_task_instance(
                idx, test_hidden=config.test_hidden
            )
            logger.info(
                f"Starting task instance {idx} for 3D evaluation..."
            )

            for epi in range(m.NUM_EVAL_EPISODES):
                episode_outputs = evaluator.run_episode(
                    instance_id=idx,
                    episode_id=epi,
                    metrics_path=metrics_path,
                    video_path=video_path if (should_write_video or should_write_third_person_video) else None,
                )
                log_episode_summary(
                    logger,
                    evaluator,
                    episode_outputs,
                    write_video=should_write_video,
                    write_third_person_video=should_write_third_person_video,
                )

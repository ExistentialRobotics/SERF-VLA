"""4D evaluation with robot surface features included in the scene map.

Extends Evaluator4DEnvFeatMap by merging FK-posed robot surface samples and
their learned features into the tracked environment map. The environment map
updates when tracking commits a new snapshot, while robot FK is recomputed at
policy boundaries so robot geometry stays aligned with the current state.
"""

import csv
import hydra
import json
import logging
import numpy as np
import omnigibson as og
import os
import sys
import torch as th
import traceback
from hydra.utils import instantiate
from inspect import getsourcefile
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from signal import signal, SIGINT
from typing import Dict, Optional, Tuple

from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
from omnigibson.macros import gm, create_module_macros

from omnigibson.learning.eval_4d_env_feat_serf_b1k import Evaluator4DEnvFeatMap
from omnigibson.learning.eval import (
    log_episode_summary,
    resolve_eval_output_paths,
    resolve_instances_to_run,
)
from omnigibson.learning.serf_b1k_utils import (
    normalize_sampled_point_payload,
    resolve_eval_scene_hdf5_path,
    sample_robot_map_payload,
)


m = create_module_macros(module_path=__file__)
m.NUM_EVAL_EPISODES = 1
m.NUM_TRAIN_INSTANCES = 200
m.NUM_EVAL_INSTANCES = 20

logger = logging.getLogger("evaluator_4d_robot")
logger.setLevel(20)


def _import_robot_utils():
    from serf_b1k.mapping.utils import robot as robot_utils

    return robot_utils


class Evaluator4DEnvRobotFeatMap(Evaluator4DEnvFeatMap):
    """4D evaluator that augments the tracked scene map with robot samples."""

    def __init__(self, cfg: DictConfig) -> None:
        self._robot_sampler = None
        self._robot_features: Optional[np.ndarray] = None
        self._last_env_world: Optional[Dict[str, Optional[np.ndarray]]] = None
        self._last_merged: Optional[Dict[str, Optional[np.ndarray]]] = None
        self._last_robot_xyz: Optional[np.ndarray] = None
        self._robot_model_root_path: Optional[str] = getattr(
            cfg, "robot_model_root_path", None
        )
        super().__init__(cfg)
        if self._robot_model_root_path in (None, "???", "null", "None"):
            self._robot_model_root_path = (
                self._eval_map_settings.robot_model_root_path
            )

    # ------------------------------------------------------------------ #
    #  Override: _map_dataset_path                                        #
    # ------------------------------------------------------------------ #

    def _map_dataset_path(self, instance_id: int) -> str:
        if self.map_input_type == "4d_env_robot_feat_map":
            return resolve_eval_scene_hdf5_path(
                self.map_dataset_root_path,
                self.task_id,
                instance_id,
            )
        return super()._map_dataset_path(instance_id)

    # ------------------------------------------------------------------ #
    #  Override: prepare episode context                                  #
    # ------------------------------------------------------------------ #

    def _prepare_episode_context(self, instance_id: int, episode_id: int) -> None:
        """Initialize the tracker and robot-side map assets."""
        super()._prepare_episode_context(instance_id, episode_id)
        self._load_robot_assets()
        self._last_env_world = None
        self._last_merged = None

    def _load_robot_assets(self) -> None:
        """Load the robot surface sampler and learned features once per task."""
        if self._robot_sampler is not None:
            return

        robot_utils = _import_robot_utils()
        RobotSurfaceSampler = robot_utils.RobotSurfaceSampler

        if self._robot_model_root_path is None:
            raise ValueError(
                "robot_model_root_path must be set in config for 4d_env_robot_feat_map eval"
            )

        robot_dir = (
            Path(self._robot_model_root_path) / self.task_id / "neural_points" / "robot"
        )
        urdf_path = Path(self._robot_model_root_path).parent / "robot" / "urdf" / "r1pro.urdf"

        sampler_path = robot_dir / "sampler.npz"
        if not sampler_path.exists():
            raise FileNotFoundError(f"Robot sampler not found: {sampler_path}")
        if not urdf_path.exists():
            raise FileNotFoundError(f"Robot URDF not found: {urdf_path}")
        self._robot_sampler = RobotSurfaceSampler.load(
            str(sampler_path), str(urdf_path)
        )

        npt_path = robot_dir / "robot_neural_points.pt"
        if not npt_path.exists():
            raise FileNotFoundError(f"Robot feature checkpoint not found: {npt_path}")
        checkpoint = th.load(npt_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "features" in checkpoint:
            self._robot_features = checkpoint["features"].detach().numpy().astype(np.float32)
        else:
            state = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()
            if "features" not in state:
                raise ValueError(
                    "Robot feature checkpoint is missing required 'features' "
                    f"tensor: {npt_path}"
                )
            self._robot_features = state["features"].detach().numpy().astype(np.float32)

        if self._robot_features is None or self._robot_features.size == 0:
            raise ValueError(
                "Robot feature tensor is missing or empty for "
                f"4d_env_robot_feat_map eval: {npt_path}"
            )

        n_sampler = self._robot_sampler.local_points_homo.shape[0]
        n_features = self._robot_features.shape[0]
        assert n_features == n_sampler, (
            f"Robot feature count ({n_features}) != sampler point count ({n_sampler}). "
            f"Ensure feature and sampler checkpoints are from the same training run."
        )
        self._tracker.set_robot_visualization_assets(features=self._robot_features)

        logger.info(
            f"Loaded robot assets: sampler={n_sampler} pts, "
            f"features={self._robot_features.shape}"
        )

    def _is_policy_boundary(self) -> bool:
        boundary_fn = getattr(self, "_should_flush_tracker_for_policy", None)
        return bool(boundary_fn()) if callable(boundary_fn) else False

    def _get_env_map_world(self) -> Dict[str, Optional[np.ndarray]]:
        env_world = self._tracker.get_current_points_world()
        env_feat = env_world.get("feat")

        if self.map_input_type == "4d_env_robot_feat_map":
            if env_feat is None:
                raise ValueError(
                    "Env tracking output is missing latent features for "
                    "4d_env_robot_feat_map eval. Check the tracking HDF5 contains "
                    "initial_features / features / latent_features / latent."
                )
            if self._robot_features is None:
                raise ValueError(
                    "Robot features are missing for "
                    "4d_env_robot_feat_map eval."
                )

        return {
            "xyz": env_world["xyz"].astype(np.float32),
            "instance_ids": env_world["instance_ids"],
            "feat": env_feat.astype(np.float32) if env_feat is not None else None,
            "rgb": env_world.get("rgb"),
        }

    def _merge_env_and_robot_map(
        self,
        env_world: Dict[str, Optional[np.ndarray]],
        robot_xyz: Optional[np.ndarray],
    ) -> Dict[str, Optional[np.ndarray]]:
        sampled, robot_mask = sample_robot_map_payload(
            total_n=self.num_input_points,
            env_instance_ids=env_world["instance_ids"],
            env_xyz=env_world["xyz"],
            env_feat=env_world.get("feat"),
            env_rgb=env_world.get("rgb"),
            robot_xyz=robot_xyz,
            robot_feat=self._robot_features,
            id_to_name=self._tracker._training_id_map,
            keep_all_categories=self.instance_keep_all_categories,
            budget_categories=self.instance_budget_categories or (),
        )
        xyz_norm, feat_out, _ = normalize_sampled_point_payload(sampled, self.task_id)

        return {
            "xyz": xyz_norm,
            "feat": feat_out,
            "robot_mask": robot_mask,
        }

    # ------------------------------------------------------------------ #
    #  Override: _preprocess_obs                                         #
    # ------------------------------------------------------------------ #

    def _preprocess_obs(self, obs: dict) -> dict:
        """Merge robot FK samples into the tracked scene map.

        The parent handles tracker updates and writes the current environment
        map into the policy observation. This override refreshes the cached
        env snapshot when tracking commits,
        recomputes robot FK at policy boundaries, and overwrites the
        ``observation.points`` tensors with the robot-inclusive result.
        """
        flat_obs = super()._preprocess_obs(obs)

        if not self._tracker.initialized or self._robot_sampler is None:
            return flat_obs

        should_refresh_env = bool(
            getattr(self, "_tracker_flushed_for_policy", False)
        ) or (
            getattr(self, "_last_env_world", None) is None
            and self._last_merged is None
        )
        should_refresh_robot = (
            should_refresh_env
            or self._last_merged is None
            or self._is_policy_boundary()
        )

        if should_refresh_env:
            self._last_env_world = self._get_env_map_world()

        if should_refresh_robot:
            robot_state = flat_obs.get("robot_r1::proprio")
            if robot_state is None:
                return flat_obs
            if isinstance(robot_state, th.Tensor):
                robot_state = robot_state.cpu().numpy()
            robot_state = robot_state.astype(np.float32)

            robot_utils = _import_robot_utils()
            cfg = robot_utils.state_to_urdf_cfg(robot_state)
            base_tf = robot_utils.state_to_base_transform_matrix(robot_state)
            robot_xyz, _ = self._robot_sampler.get_points(cfg, base_tf)
            self._last_robot_xyz = robot_xyz.astype(np.float32)
            self._tracker.set_robot_visualization_points(self._last_robot_xyz)

        if should_refresh_env or should_refresh_robot:
            env_world = getattr(self, "_last_env_world", None)
            if env_world is not None:
                self._last_merged = self._merge_env_and_robot_map(
                    env_world=env_world,
                    robot_xyz=self._last_robot_xyz,
                )

        # Replace env-only tensors with the robot-inclusive map sample.
        if self._last_merged is not None:
            if self._last_robot_xyz is not None:
                self._tracker.set_robot_visualization_points(self._last_robot_xyz)
            flat_obs["observation/points/xyz"] = th.from_numpy(
                self._last_merged["xyz"]
            )
            if self._last_merged.get("feat") is not None:
                flat_obs["observation/points/feat"] = th.from_numpy(
                    self._last_merged["feat"]
                )
            if self._last_merged.get("robot_mask") is not None:
                flat_obs["observation/points/robot_mask"] = th.from_numpy(
                    self._last_merged["robot_mask"]
                )

        return flat_obs

    # ------------------------------------------------------------------ #
    #  Override: reset                                                    #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        self._last_env_world = None
        self._last_merged = None
        self._last_robot_xyz = None
        self._tracker.set_robot_visualization_points(None)
        super().reset()

if __name__ == "__main__":
    register_omegaconf_resolvers()
    with hydra.initialize_config_dir(
        f"{Path(getsourcefile(lambda: 0)).parents[0]}/configs", version_base="1.1"
    ):
        config = hydra.compose("serf_config.yaml", overrides=sys.argv[1:])
    OmegaConf.resolve(config)

    gm.HEADLESS = config.headless

    write_tracking_video = bool(
        getattr(config, "write_tracking_video", getattr(config, "tracking_save_2d_video", False))
    )
    metrics_path, video_path, should_write_video, should_write_third_person_video = resolve_eval_output_paths(
        config,
        include_tracking_video=write_tracking_video,
    )
    instances_to_run = resolve_instances_to_run(
        config,
        num_train_instances=m.NUM_TRAIN_INSTANCES,
        num_eval_instances=m.NUM_EVAL_INSTANCES,
    )

    # ---- run evaluation ----
    with Evaluator4DEnvRobotFeatMap(config) as evaluator:
        logger.info("Starting 4D robot-inclusive evaluation...")

        for idx in instances_to_run:
            evaluator.reset()
            evaluator.load_task_instance(idx, test_hidden=config.test_hidden)
            logger.info(f"Starting task instance {idx} for 4D robot evaluation...")

            for epi in range(m.NUM_EVAL_EPISODES):
                episode_outputs = evaluator.run_episode(
                    instance_id=idx,
                    episode_id=epi,
                    metrics_path=metrics_path,
                    video_path=video_path if (should_write_video or should_write_third_person_video or evaluator._tracker.config.save_tracking_video) else None,
                )
                log_episode_summary(
                    logger,
                    evaluator,
                    episode_outputs,
                    write_video=should_write_video,
                    write_third_person_video=should_write_third_person_video,
                )
                if episode_outputs["tracking_2d_video"] is not None:
                    logger.info(
                        f"Saved 2D tracking video to {episode_outputs['tracking_2d_video']}"
                    )
                if episode_outputs["tracking_3d_video"] is not None:
                    logger.info(
                        f"Saved 3D tracking video to {episode_outputs['tracking_3d_video']}"
                    )

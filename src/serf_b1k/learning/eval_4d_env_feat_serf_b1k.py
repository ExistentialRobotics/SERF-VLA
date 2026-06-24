import csv
import hydra
import json
import logging
import numpy as np
import omnigibson as og
import omnigibson.utils.transform_utils as T
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

from gello.robots.sim_robot.og_teleop_utils import (
    augment_rooms,
    load_available_tasks,
    generate_robot_config,
    get_task_relevant_room_types,
)
from gello.robots.sim_robot.og_teleop_cfg import DISABLED_TRANSITION_RULES
from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
from omnigibson.learning.utils.eval_utils import (
    ROBOT_CAMERA_NAMES,
    PROPRIOCEPTION_INDICES,
    CAMERA_INTRINSICS,
    HEAD_RESOLUTION,
    WRIST_RESOLUTION,
    generate_basic_environment_config,
    flatten_obs_dict,
    TASK_NAMES_TO_INDICES,
)
from omnigibson.macros import gm, create_module_macros

from omnigibson.learning.eval_3d_env_feat_serf_b1k import Evaluator3DEnvFeatMap
from omnigibson.learning.eval import (
    get_third_person_sensor_kwargs,
    log_episode_summary,
    resolve_eval_output_paths,
    resolve_instances_to_run,
)
from omnigibson.learning.serf_b1k_utils import (
    EvalPolicyBoundaryState,
    resolve_eval_map_settings,
    resolve_eval_scene_hdf5_path,
    task_name_to_task_id,
)
from omnigibson.learning.online_3d_tracker import (
    OnlineTracker,
    OnlineTrackerConfig,
)
from serf_b1k.mapping.tracking.config.tracking_config import (
    TrackingConfig as ExternalTrackingConfig,
    load_task_config,
)

m = create_module_macros(module_path=__file__)
m.NUM_EVAL_EPISODES = 1
m.NUM_TRAIN_INSTANCES = 200
m.NUM_EVAL_INSTANCES = 20

logger = logging.getLogger("evaluator_4d")
logger.setLevel(20)


# ====================================================================== #
#  Intrinsics helpers                                                    #
# ====================================================================== #

# Original camera resolutions that the intrinsic matrices correspond to.
_ORIGINAL_RESOLUTIONS: Dict[str, Tuple[int, int]] = {
    "head": HEAD_RESOLUTION,          # (720, 720)
    "left_wrist": WRIST_RESOLUTION,   # (480, 480)
    "right_wrist": WRIST_RESOLUTION,  # (480, 480)
}

# ``serf_b1k.mapping`` stores / consumes camera poses in OpenCV camera
# coordinates (x right, y down, z forward), while OmniGibson camera poses use
# x right, y up, -z forward. Convert the camera frame before unprojection.
_T_CV_OG = np.array(
    [
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float32,
)


def _intrinsics_tuple(cam_id: str) -> Tuple[float, float, float, float]:
    """Extract ``(fx, fy, cx, cy)`` from the 3x3 intrinsic matrix."""
    K = CAMERA_INTRINSICS["R1Pro"][cam_id]
    return float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])


def _scale_intrinsics(
    fx: float, fy: float, cx: float, cy: float,
    orig_h: int, orig_w: int, cur_h: int, cur_w: int,
) -> Tuple[float, float, float, float]:
    """Scale intrinsics from original to current (potentially lower) resolution."""
    sx = cur_w / orig_w
    sy = cur_h / orig_h
    return fx * sx, fy * sy, cx * sx, cy * sy


# ====================================================================== #
#  Sensor-name lookup                                                    #
# ====================================================================== #

# Mapping from short camera id to the OmniGibson sensor name (without the
# ``robot_r1::`` prefix that ROBOT_CAMERA_NAMES carries).
_SENSOR_NAMES: Dict[str, str] = {
    cam_id: full_name.split("::")[1]
    for cam_id, full_name in ROBOT_CAMERA_NAMES["R1Pro"].items()
}


def _task_tracking_config(task_name: str) -> ExternalTrackingConfig:
    task_id = task_name_to_task_id(task_name).removeprefix("task-")
    try:
        return load_task_config(task_id)
    except FileNotFoundError:
        config_dir = Path(__file__).resolve().parents[1] / "mapping" / "tracking" / "config"
        return ExternalTrackingConfig.from_yaml(
            str(config_dir / "tracking.yaml")
        )


def _required_tracking_buffer_size(
    execute_in_n_steps: int,
    frame_step: int,
) -> int:
    execute_in_n_steps = int(execute_in_n_steps)
    frame_step = max(1, int(frame_step))
    if execute_in_n_steps % frame_step != 0:
        raise ValueError(
            "execute_in_n_steps must be divisible by tracking_frame_step "
            f"for one-frame tracker overlap (got execute_in_n_steps={execute_in_n_steps}, "
            f"tracking_frame_step={frame_step})"
        )
    return 1 + (max(0, execute_in_n_steps) // frame_step)


def _build_online_tracker_config(cfg: DictConfig) -> OnlineTrackerConfig:
    external_cfg = _task_tracking_config(cfg.task.name)
    map_settings = resolve_eval_map_settings(cfg)

    def _cfg_value(*names: str, default):
        for name in names:
            value = getattr(cfg, name, None)
            if value is not None:
                return value
        return default

    buffer_size = getattr(cfg, "tracking_buffer_size", None)
    if buffer_size is None:
        buffer_size = getattr(cfg, "tracking_buffer_len", None)
    if buffer_size is None:
        buffer_size = external_cfg.buffer_size
    frame_step = int(
        _cfg_value(
            "tracking_frame_step",
            "tracking_update_stride",
            default=external_cfg.frame_step,
        )
    )
    execute_in_n_steps = getattr(cfg, "execute_in_n_steps", None)
    if execute_in_n_steps is not None:
        buffer_size = max(
            int(buffer_size),
            _required_tracking_buffer_size(
                execute_in_n_steps=int(execute_in_n_steps),
                frame_step=frame_step,
            ),
        )

    save_tracking_video = getattr(cfg, "write_tracking_video", None)
    if save_tracking_video is None:
        save_tracking_video = getattr(cfg, "tracking_save_2d_video", None)
    if save_tracking_video is None:
        save_tracking_video = False

    return OnlineTrackerConfig(
        device=str(_cfg_value("tracking_device", default=external_cfg.device)),
        views=tuple(
            str(view)
            for view in _cfg_value("tracking_views", default=external_cfg.views)
        ),
        buffer_size=int(buffer_size),
        frame_step=frame_step,
        target_instance_names=tuple(
            str(name)
            for name in _cfg_value(
                "tracking_target_instance_names",
                default=external_cfg.target_instance_names,
            )
        ),
        save_tracking_video=bool(save_tracking_video),
        tracking_video_fps=int(
            _cfg_value("tracking_video_fps", default=external_cfg.video_fps)
        ),
        num_keypoints_per_instance=int(
            _cfg_value(
                "tracking_num_keypoints_per_instance",
                default=external_cfg.max_total_points_per_instance,
            )
        ),
        visibility_threshold=float(
            _cfg_value(
                "tracking_visibility_threshold",
                default=external_cfg.visibility_threshold,
            )
        ),
        confidence_threshold=float(
            _cfg_value(
                "tracking_confidence_threshold",
                default=external_cfg.confidence_threshold,
            )
        ),
        fgr_threshold=float(
            _cfg_value("tracking_fgr_threshold", default=external_cfg.fgr_threshold)
        ),
        icp_threshold=float(
            _cfg_value("tracking_icp_threshold", default=external_cfg.icp_threshold)
        ),
        icp_min_points=int(
            _cfg_value("tracking_icp_min_points", default=external_cfg.icp_min_points)
        ),
        icp_max_iterations=int(
            _cfg_value(
                "tracking_icp_max_iterations",
                default=external_cfg.icp_max_iterations,
            )
        ),
        rescue_icp_max_iterations=int(
            _cfg_value(
                "tracking_rescue_icp_max_iterations",
                default=external_cfg.rescue_icp_max_iterations,
            )
        ),
        stationary_threshold=float(
            _cfg_value(
                "tracking_stationary_threshold",
                default=external_cfg.stationary_threshold,
            )
        ),
        min_observed_keypoints=int(
            _cfg_value(
                "tracking_min_observed_keypoints",
                default=external_cfg.min_observed_keypoints,
            )
        ),
        graph_refine_distance=float(
            _cfg_value(
                "tracking_graph_refine_distance",
                default=external_cfg.graph_refine_distance,
            )
        ),
        graph_refine_majority_ratio=float(
            _cfg_value(
                "tracking_graph_refine_majority_ratio",
                default=external_cfg.graph_refine_majority_ratio,
            )
        ),
        rescue_drift_threshold=float(
            _cfg_value(
                "tracking_rescue_drift_threshold",
                default=external_cfg.rescue_drift_threshold,
            )
        ),
        rescue_acceptance_threshold=float(
            _cfg_value(
                "tracking_rescue_acceptance_threshold",
                default=external_cfg.rescue_acceptance_threshold,
            )
        ),
        exclude_categories=tuple(
            str(name)
            for name in _cfg_value(
                "tracking_exclude_categories",
                default=external_cfg.exclude_categories,
            )
        ),
        erosion_iters=int(
            _cfg_value("tracking_erosion_iters", default=external_cfg.erosion_iters)
        ),
        kp_quality_level=float(
            _cfg_value(
                "tracking_kp_quality_level",
                default=external_cfg.kp_quality_level,
            )
        ),
        kp_min_distance=int(
            _cfg_value("tracking_kp_min_distance", default=external_cfg.kp_min_distance)
        ),
        kp_block_size=int(
            _cfg_value("tracking_kp_block_size", default=external_cfg.kp_block_size)
        ),
        fallback_lookback=int(
            _cfg_value(
                "tracking_fallback_lookback",
                default=external_cfg.fallback_lookback,
            )
        ),
        instance_keep_all_categories=map_settings.instance_keep_all_categories,
        instance_budget_categories=map_settings.instance_budget_categories,
        camera_extrinsic=external_cfg.camera_extrinsic,
    )


# ====================================================================== #
#  Evaluator4DEnvFeatMap                                                 #
# ====================================================================== #

class Evaluator4DEnvFeatMap(Evaluator3DEnvFeatMap):
    """Evaluator that updates the scene feature map with online tracking."""

    def __init__(self, cfg: DictConfig) -> None:
        # Create tracker *before* super().__init__ which calls reset().
        tracker_config = _build_online_tracker_config(cfg)
        self._tracker = OnlineTracker(config=tracker_config)
        self._last_info: Optional[dict] = None
        self._last_policy_map: Optional[Dict[str, Optional[np.ndarray]]] = None
        self._last_tracking_intrinsics: Optional[Dict[str, Tuple[float, float, float, float]]] = None
        self._tracker_flushed_for_policy: bool = False
        self._policy_boundary_state = EvalPolicyBoundaryState()

        super().__init__(cfg)
        self._ensure_tracker_policy_overlap()

    def _ensure_tracker_policy_overlap(self) -> None:
        policy = getattr(self, "policy", None)
        if policy is None or not hasattr(self._tracker, "ensure_buffer_size"):
            return

        execute_steps = self._get_execute_in_n_steps(policy)
        if execute_steps is None:
            return

        required_buffer_size = _required_tracking_buffer_size(
            execute_in_n_steps=execute_steps,
            frame_step=int(self._tracker.config.frame_step),
        )
        self._tracker.ensure_buffer_size(required_buffer_size)

    def _get_execute_in_n_steps(self, policy=None) -> Optional[int]:
        execute_steps = getattr(getattr(self, "cfg", None), "execute_in_n_steps", None)
        if execute_steps is None and policy is not None:
            execute_steps = getattr(getattr(policy, "config", None), "execute_in_n_steps", None)
        if execute_steps is None:
            return None
        return int(execute_steps)

    # ------------------------------------------------------------------ #
    #  Override: _map_dataset_path                                        #
    # ------------------------------------------------------------------ #

    def _map_dataset_path(self, instance_id: int) -> str:
        """Resolve the initial map for the 4D feature-map variant."""
        if self.map_input_type == "4d_env_feat_map":
            return resolve_eval_scene_hdf5_path(
                self.map_dataset_root_path,
                self.task_id,
                instance_id,
            )
        return super()._map_dataset_path(instance_id)

    # ------------------------------------------------------------------ #
    #  Override: load_env                                               #
    # ------------------------------------------------------------------ #

    def load_env(self, env_wrapper: DictConfig):
        """Create the eval environment with tracking observation modalities.

        ``obs_modalities`` must be set before ``og.Environment`` is created,
        so this mirrors the base environment construction and adds the depth
        and segmentation streams required by the tracker.
        """
        for rule in DISABLED_TRANSITION_RULES:
            rule.ENABLED = False

        available_tasks = load_available_tasks()
        task_name = self.cfg.task.name
        assert task_name in available_tasks, f"Got invalid task name: {task_name}"

        task_idx = TASK_NAMES_TO_INDICES[task_name]
        self.human_stats = {
            "length": [],
            "distance_traveled": [],
            "left_eef_displacement": [],
            "right_eef_displacement": [],
        }
        with open(
            os.path.join(
                gm.DATA_PATH,
                "2025-challenge-task-instances",
                "metadata",
                "episodes.jsonl",
            ),
            "r",
        ) as f:
            episodes = [json.loads(line) for line in f]
        for episode in episodes:
            if episode["episode_index"] // 1e4 == task_idx:
                for k in self.human_stats:
                    self.human_stats[k].append(episode[k])
        for k in self.human_stats:
            self.human_stats[k] = sum(self.human_stats[k]) / len(self.human_stats[k])

        task_cfg = available_tasks[task_name][0]
        robot_type = self.cfg.robot.type
        assert robot_type == "R1Pro", f"Got invalid robot type: {robot_type}"

        cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)
        if self.cfg.partial_scene_load:
            relevant_rooms = get_task_relevant_room_types(activity_name=task_name)
            relevant_rooms = augment_rooms(
                relevant_rooms, task_cfg["scene_model"], task_name
            )
            cfg["scene"]["load_room_types"] = relevant_rooms

        cfg["robots"] = [
            generate_robot_config(task_name=task_name, task_cfg=task_cfg)
        ]

        # Depth and instance segmentation are consumed by the online tracker.
        cfg["robots"][0]["obs_modalities"] = [
            "proprio", "rgb", "depth_linear", "seg_instance",
        ]
        cfg["robots"][0]["proprio_obs"] = list(
            PROPRIOCEPTION_INDICES["R1Pro"].keys()
        )

        if self.cfg.robot.controllers is not None:
            cfg["robots"][0]["controller_config"].update(self.cfg.robot.controllers)
        if self.cfg.max_steps is None:
            logger.info(
                f"Setting timeout to 2x average human demo length: "
                f"{int(self.human_stats['length'] * 2)}"
            )
            cfg["task"]["termination_config"]["max_steps"] = int(
                self.human_stats["length"] * 2
            )
        else:
            logger.info(f"Setting timeout to {self.cfg.max_steps} steps.")
            cfg["task"]["termination_config"]["max_steps"] = self.cfg.max_steps

        cfg["task"]["include_obs"] = False
        third_person_view_config = self.third_person_view_config

        cfg["env"]["external_sensors"] = [
            dict(
                sensor_type="VisionSensor",
                relative_prim_path=third_person_view_config["relative_prim_path"],
                name=third_person_view_config["name"],
                modalities=["rgb"],
                sensor_kwargs=get_third_person_sensor_kwargs(third_person_view_config),
                include_in_obs=False,
                position=list(third_person_view_config["position"]),
                orientation=list(third_person_view_config["orientation"])
            )
        ]

        env = og.Environment(configs=cfg)
        env = instantiate(env_wrapper, env=env)
        return env

    # ------------------------------------------------------------------ #
    #  Override: prepare episode context                                #
    # ------------------------------------------------------------------ #

    def _prepare_episode_context(self, instance_id: int, episode_id: int) -> None:
        """Initialise the online tracker from the episode's map file.

        ``_current_instance_index`` is intentionally left unset so the parent
        does not inject the static map; policy inputs come from the tracker.
        """
        if self.map_dataset_root_path is None:
            raise ValueError("map_dataset_root_path is not defined.")

        self._last_info = None

        hdf5_path = self._map_dataset_path(instance_id)
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"Map dataset not found: {hdf5_path}")

        task_idx_str = self.task_id

        self._tracker.initialize(
            hdf5_path=hdf5_path,
            task_idx_str=task_idx_str,
            map_input_type=self.map_input_type,
            num_points=self.num_input_points,
        )
        logger.info(
            f"[4D] Tracker initialised for instance {instance_id}, "
            f"task {task_idx_str}, type {self.map_input_type}"
        )

    def _after_env_step(self, info: dict) -> None:
        self._last_info = info
        self._policy_boundary_state.after_env_step(
            self._get_execute_in_n_steps(getattr(self, "policy", None))
        )

    # ------------------------------------------------------------------ #
    #  Override: _preprocess_obs                                       #
    # ------------------------------------------------------------------ #

    def _preprocess_obs(self, obs: dict) -> dict:
        self._ensure_tracker_policy_overlap()
        self._tracker_flushed_for_policy = False
        if self._tracker.initialized:

            # --- 1. Extract raw tracking inputs before observation flattening. ---
            rgb_maps, depth_maps, seg_maps = self._extract_tracking_inputs(obs)

            # --- 2. Update dynamic simulator-id to object-name mapping. ---
            seg_info = self._extract_seg_info()
            if seg_info:
                self._tracker.build_id_mapping(seg_info)

            # --- 3. Append the frame and flush tracker state at policy boundaries. ---
            camera_poses_c2w = self._get_camera_poses_c2w()
            intrinsics = self._get_scaled_intrinsics(depth_maps)
            required_views = tuple(self._tracker.config.views)
            has_complete_tracking_frame = all(
                view in rgb_maps
                and view in depth_maps
                and view in seg_maps
                and view in camera_poses_c2w
                and view in intrinsics
                for view in required_views
            )
            if has_complete_tracking_frame:
                self._tracker.append_observation(
                    rgb_maps, depth_maps, seg_maps, camera_poses_c2w
                )
                self._last_tracking_intrinsics = {
                    view: intrinsics[view] for view in required_views
                }
            flush_intrinsics = getattr(self, "_last_tracking_intrinsics", None)
            if (
                self._should_flush_tracker_for_policy()
                and flush_intrinsics is not None
            ):
                self._tracker_flushed_for_policy = self._tracker.flush_pending(
                    flush_intrinsics
                )
                if self._tracker_flushed_for_policy:
                    self._last_policy_map = self._tracker.get_current_points()

        # --- 4. Run the standard policy observation pipeline. ---
        flat_obs = super()._preprocess_obs(obs)

        # --- 5. Replace static map tensors with the latest tracker snapshot. ---
        if self._tracker.initialized and self._last_policy_map is not None:
            pts = self._last_policy_map
            flat_obs["observation/points/xyz"] = th.from_numpy(
                pts["observation.points.xyz"]
            )
            if pts["observation.points.rgb"] is not None:
                flat_obs["observation/points/rgb"] = th.from_numpy(
                    pts["observation.points.rgb"]
                )
            if pts["observation.points.feat"] is not None:
                flat_obs["observation/points/feat"] = th.from_numpy(
                    pts["observation.points.feat"]
                )

        return flat_obs

    def _should_flush_tracker_for_policy(self) -> bool:
        policy = getattr(self, "policy", None)
        execute_steps = self._get_execute_in_n_steps(policy)
        return self._policy_boundary_state.should_flush_for_policy(
            policy,
            execute_steps=execute_steps,
        )

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _extract_tracking_inputs(
        self, obs: dict
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
    ]:
        """Pull ``rgb``, ``depth_linear``, and ``seg_instance`` from raw obs.

        The raw observation from OmniGibson has the structure::

            obs["robot_r1"]["robot_r1:zed_link:Camera:0"]["depth_linear"]
        """
        rgb_maps: Dict[str, np.ndarray] = {}
        depth_maps: Dict[str, np.ndarray] = {}
        seg_maps: Dict[str, np.ndarray] = {}
        robot_obs = obs.get("robot_r1", {})

        for cam_id in self._tracker.config.views:
            sensor_name = _SENSOR_NAMES[cam_id]
            sensor_obs = robot_obs.get(sensor_name, {})

            rgb = sensor_obs.get("rgb")
            if rgb is not None:
                if isinstance(rgb, th.Tensor):
                    rgb_maps[cam_id] = rgb.cpu().numpy()[..., :3].astype(np.uint8)
                else:
                    rgb_maps[cam_id] = np.asarray(rgb)[..., :3].astype(np.uint8)

            d = sensor_obs.get("depth_linear")
            if d is not None:
                if isinstance(d, th.Tensor):
                    depth = d.cpu().numpy().astype(np.float32)
                else:
                    depth = np.asarray(d, dtype=np.float32)
                depth[~np.isfinite(depth)] = 0.0
                depth[depth < 0.0] = 0.0
                depth_maps[cam_id] = depth

            s = sensor_obs.get("seg_instance")
            if s is not None:
                if isinstance(s, th.Tensor):
                    seg_maps[cam_id] = s.cpu().numpy().astype(np.int64)
                else:
                    seg_maps[cam_id] = np.asarray(s, dtype=np.int64)

        return rgb_maps, depth_maps, seg_maps

    def _get_camera_poses_c2w(self) -> Dict[str, np.ndarray]:
        """Return camera-to-world 4×4 matrices in OpenCV camera convention.

        Follows the same camera-parameter extraction logic as the parent's
        ``_preprocess_obs`` (lines 336-345 of ``eval_serf_b1k.py``), but
        returns the absolute camera-to-world matrix rather than the relative
        pose used by the policy.

        The online tracker reuses ``serf_b1k.mapping`` utilities,
        which expect camera coordinates in OpenCV convention. OmniGibson's
        camera poses are in its native convention, so we apply the fixed
        camera-frame conversion here.
        """
        poses: Dict[str, np.ndarray] = {}
        for cam_id in self._tracker.config.views:
            sensor_name = _SENSOR_NAMES[cam_id]
            camera = self.robot.sensors[sensor_name]
            view_transform = camera.camera_parameters["cameraViewTransform"]
            if np.allclose(view_transform, np.zeros(16)):
                pos, orn = camera.get_position_orientation()
                c2w_og = T.pose2mat((pos, orn)).cpu().numpy().astype(np.float32)
            else:
                # cameraViewTransform is world→camera, stored column-major
                w2c = np.reshape(view_transform, [4, 4]).T
                c2w_og = np.linalg.inv(w2c).astype(np.float32)
            poses[cam_id] = (c2w_og @ _T_CV_OG).astype(np.float32)
        return poses

    def _get_scaled_intrinsics(
        self, depth_maps: Dict[str, np.ndarray]
    ) -> Dict[str, Tuple[float, float, float, float]]:
        """Scale the original intrinsics to match the actual depth-map size.

        Tracking renders may run at a lower square resolution than the
        calibration matrices were defined for (e.g. 720->480 for the head
        camera), so intrinsics must be scaled to the actual depth-map size.
        """
        result: Dict[str, Tuple[float, float, float, float]] = {}
        for cam_id in self._tracker.config.views:
            depth = depth_maps.get(cam_id)
            if depth is None:
                continue
            fx, fy, cx, cy = _intrinsics_tuple(cam_id)
            orig_h, orig_w = _ORIGINAL_RESOLUTIONS[cam_id]
            cur_h, cur_w = depth.shape[:2]
            result[cam_id] = _scale_intrinsics(
                fx, fy, cx, cy, orig_h, orig_w, cur_h, cur_w
            )
        return result

    def _extract_seg_info(self) -> Optional[Dict[int, str]]:
        """Extract ``{og_instance_id: object_name}`` from the last step info.

        The info dict returned by ``env.step`` contains::

            info["obs_info"]["robot_r1"][sensor_name]["seg_instance"]
                = {int_id: "object_name", ...}

        We merge all sensors into a single dict.
        """
        if self._last_info is None:
            return None
        
        obs_info = self._last_info.get("obs_info")
        while (
            isinstance(obs_info, dict)
            and "robot_r1" not in obs_info
            and "obs_info" in obs_info
        ):
            obs_info = obs_info["obs_info"]

        robot_info = obs_info.get("robot_r1", {}) if isinstance(obs_info, dict) else {}
        combined: Dict[int, str] = {}
        for sensor_name, sensor_info in robot_info.items():
            if isinstance(sensor_info, dict) and "seg_instance" in sensor_info:
                for id_val, name in sensor_info["seg_instance"].items():
                    id_int = int(id_val) if isinstance(id_val, str) else id_val
                    combined[id_int] = name
        return combined if combined else None

    # ------------------------------------------------------------------ #
    #  Override: reset                                                  #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset tracker state in addition to the parent reset.

        After the standard reset we attempt an extra ``env.get_obs()`` call
        to obtain the ``obs_info`` dict (which contains the seg-instance
        id→name mapping).  This allows the tracker ID mapping to be built
        during ``reset`` rather than waiting for the first ``step``.
        """
        if hasattr(self, "_tracker") and self._tracker.initialized:
            self._tracker.reset()
            self._last_policy_map = self._tracker.get_current_points()
        else:
            self._last_policy_map = None
        self._last_tracking_intrinsics = None
        self._last_info = None
        self._tracker_flushed_for_policy = False
        self._policy_boundary_state.reset()

        # Parent reset: env.reset() + _preprocess_obs (no tracking yet)
        super().reset()

        # Try to obtain obs_info for early ID-mapping construction.
        # env.get_obs() is cheap (no physics step) and returns (obs, info).
        if hasattr(self, "_tracker") and self._tracker.initialized:
            try:
                _, obs_info = self.env.get_obs()
                self._last_info = {"obs_info": obs_info}
                seg_info = self._extract_seg_info()
                if seg_info:
                    self._tracker.build_id_mapping(seg_info)
            except Exception:
                pass  # not critical — mapping will be built on first step()

    def open_episode_outputs(
        self,
        *,
        instance_id: int,
        episode_id: int,
        video_path: Optional[Path],
    ) -> Dict[str, Optional[str]]:
        output_paths = super().open_episode_outputs(
            instance_id=instance_id,
            episode_id=episode_id,
            video_path=video_path,
        )
        output_paths["tracking_2d_video"] = None
        output_paths["tracking_3d_video"] = None

        if video_path is not None and self._tracker.config.save_tracking_video:
            base = video_path / f"{self.cfg.task.name}_{instance_id}_{episode_id}"
            tracking_2d_name = str(base) + "_tracking_2d.mp4"
            tracking_3d_name = str(base) + "_tracking_3d.mp4"
            self._tracker.set_output_tracking_video_paths(
                path_2d=tracking_2d_name,
                path_3d=tracking_3d_name,
            )
            output_paths["tracking_2d_video"] = tracking_2d_name
            output_paths["tracking_3d_video"] = tracking_3d_name

        return output_paths

    def _close_additional_episode_outputs(self) -> None:
        if hasattr(self, "_tracker"):
            self._tracker.close()

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
    with Evaluator4DEnvFeatMap(config) as evaluator:
        logger.info("Starting 4D evaluation...")

        for idx in instances_to_run:
            evaluator.reset()
            evaluator.load_task_instance(idx, test_hidden=config.test_hidden)
            logger.info(f"Starting task instance {idx} for 4D evaluation...")

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

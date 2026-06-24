"""Tracking configuration dataclass for neural point tracking."""

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import yaml


# Directory containing per-task override YAML files
_TASKS_DIR = os.path.join(os.path.dirname(__file__), "tasks")
# Default base config path
_BASE_CONFIG = os.path.join(os.path.dirname(__file__), "tracking.yaml")


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (returns a new dict)."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_task_id(task: str | int) -> str:
    """Normalize task identifiers like ``21`` or ``"0021"`` to ``"0021"``."""
    return f"{int(task):04d}"


def get_task_config_path(task: str | int) -> str:
    """Return the task override YAML path for a tracking task."""
    return os.path.join(_TASKS_DIR, f"task_{_normalize_task_id(task)}.yaml")


def load_task_config_dict(
    task: str | int,
    base_config_path: str = _BASE_CONFIG,
) -> dict:
    """Load and merge the base tracking config with a task override YAML."""
    with open(base_config_path, "r") as f:
        base = yaml.safe_load(f) or {}

    task_file = get_task_config_path(task)
    if not os.path.exists(task_file):
        available = [
            f.replace("task_", "").replace(".yaml", "")
            for f in os.listdir(_TASKS_DIR)
            if f.startswith("task_") and f.endswith(".yaml")
        ]
        raise FileNotFoundError(
            f"No config found for task '{task}' at {task_file}. "
            f"Available tasks: {available}"
        )

    with open(task_file, "r") as f:
        task_override = yaml.safe_load(f) or {}

    return _deep_merge(base, task_override)


def load_task_config(
    task: str | int,
    base_config_path: str = _BASE_CONFIG,
) -> "TrackingConfig":
    """Load tracking config for a specific task number.

    Loads the base ``tracking.yaml``, then deep-merges the task-specific
    override from ``tasks/task_{task}.yaml`` on top.

    Parameters
    ----------
    task : str
        Task identifier, e.g. ``"0021"`` or ``"0026"``.
    base_config_path : str
        Path to the shared base config YAML.
    """
    merged = load_task_config_dict(task, base_config_path=base_config_path)
    return TrackingConfig.from_dict(merged)


@dataclass
class TrackingConfig:
    """Configuration for neural point tracking pipeline."""

    # --- Paths (required) ---
    data_path: str = ""
    run_dir: str = ""

    # --- Frame range ---
    start_frame: int = 500
    end_frame: int = 1500

    # --- Device ---
    device: str = "cuda"

    # --- Tracking parameters ---
    buffer_size: int = 10
    frame_step: int = 3
    erosion_iters: int = 2
    kp_quality_level: float = 0.01
    kp_min_distance: int = 20
    kp_block_size: int = 7
    max_total_points_per_instance: int = 30

    # --- CoTracker filtering ---
    visibility_threshold: float = 0.8
    confidence_threshold: float = 0.7

    # --- Registration parameters ---
    fgr_threshold: float = 0.05
    icp_threshold: float = 0.02
    icp_min_points: int = 200
    icp_max_iterations: int = 30
    rescue_icp_max_iterations: int = 60
    min_observed_keypoints: int = 15
    stationary_threshold: float = 0.015
    fallback_lookback: int = 20
    graph_refine_distance: float = 0.02
    graph_refine_majority_ratio: float = 0.8
    rescue_drift_threshold: float = 0.10
    rescue_acceptance_threshold: float = 0.04

    # --- Instance tracking ---
    target_instance_names: List[str] = field(
        default_factory=lambda: ["dice", "teddy_bear", "toy_train", "board_game"]
    )
    exclude_categories: List[str] = field(
        default_factory=lambda: [
            "background", "wall", "floors", "ceiling",
            "roof", "door", "window", "fence", "lawn",
        ]
    )

    # --- Camera views ---
    views: List[str] = field(
        default_factory=lambda: ["head", "left_wrist", "right_wrist"]
    )

    # --- Visualization ---
    vis_window_width: int = 1280
    vis_window_height: int = 720
    camera_extrinsic: Optional[np.ndarray] = None


    # --- Output ---
    save_video: bool = True
    output_video_path: str = "dynamic_pca_iterative.mp4"
    output_2d_video_path: str = "cotracker_2d_visualization.mp4"
    video_fps: int = 100
    save_all_instances: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrackingConfig":
        """Load config from a YAML file."""
        with open(yaml_path, "r") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict) -> "TrackingConfig":
        """Create TrackingConfig from a nested config dictionary.

        Expected structure (matches tracking/config/tracking.yaml)::

            data_path: ...
            run_dir: ...
            device: ...
            views: [...]
            frame_range:
                start_frame: ...
                end_frame: ...
            tracking:
                buffer_size: ...
                ...
            registration:
                fgr_threshold: ...
                ...
            instance_tracking:
                target_instance_names: [...]
                exclude_categories: [...]
            visualization:
                window_width: ...
                window_height: ...
                camera_extrinsic: [[...], ...]
            output:
                video_path: ...
                video_2d_path: ...
                video_fps: ...
        """
        frame_range = d.get("frame_range") or {}
        tracking = d.get("tracking") or {}
        registration = d.get("registration") or {}
        instance = d.get("instance_tracking") or {}
        vis = d.get("visualization") or {}
        output = d.get("output") or {}

        cam_ext = vis.get("camera_extrinsic")
        if cam_ext is not None:
            cam_ext = np.array(cam_ext, dtype=np.float64)

        return cls(
            data_path=d.get("data_path", ""),
            run_dir=d.get("run_dir") or "",
            start_frame=frame_range.get("start_frame", 500),
            end_frame=frame_range.get("end_frame", 1500),
            device=d.get("device", "cuda"),
            buffer_size=tracking.get("buffer_size", 10),
            frame_step=tracking.get("frame_step", 3),
            erosion_iters=tracking.get("erosion_iters", 2),
            kp_quality_level=tracking.get("kp_quality_level", 0.01),
            kp_min_distance=tracking.get("kp_min_distance", 20),
            kp_block_size=tracking.get("kp_block_size", 7),
            max_total_points_per_instance=tracking.get("max_total_points_per_instance", 30),
            visibility_threshold=tracking.get("visibility_threshold", 0.8),
            confidence_threshold=tracking.get("confidence_threshold", 0.7),
            fgr_threshold=registration.get("fgr_threshold", 0.05),
            icp_threshold=registration.get("icp_threshold", 0.02),
            icp_min_points=registration.get("icp_min_points", 200),
            icp_max_iterations=registration.get("icp_max_iterations", 30),
            rescue_icp_max_iterations=registration.get("rescue_icp_max_iterations", 60),
            min_observed_keypoints=registration.get("min_observed_keypoints", 15),
            stationary_threshold=registration.get("stationary_threshold", 0.015),
            fallback_lookback=registration.get("fallback_lookback", 20),
            graph_refine_distance=registration.get("graph_refine_distance", 0.02),
            graph_refine_majority_ratio=registration.get("graph_refine_majority_ratio", 0.8),
            rescue_drift_threshold=registration.get("rescue_drift_threshold", 0.10),
            rescue_acceptance_threshold=registration.get("rescue_acceptance_threshold", 0.04),
            target_instance_names=instance.get(
                "target_instance_names",
                ["dice", "teddy_bear", "toy_train", "board_game"],
            ),
            exclude_categories=instance.get(
                "exclude_categories",
                ["background", "wall", "floors", "ceiling",
                 "roof", "door", "window", "fence", "lawn"],
            ),
            views=d.get("views", ["head", "left_wrist", "right_wrist"]),
            vis_window_width=vis.get("window_width", 1280),
            vis_window_height=vis.get("window_height", 720),
            camera_extrinsic=cam_ext,

            save_video=output.get("save_video", True),
            output_video_path=output.get("video_path", "dynamic_pca_iterative.mp4"),
            output_2d_video_path=output.get("video_2d_path", "cotracker_2d_visualization.mp4"),
            video_fps=output.get("video_fps", 100),
            save_all_instances=output.get("save_all_instances", False),
        )

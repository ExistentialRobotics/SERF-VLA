"""LeRobot feature-map datasets for static, dynamic, and robot-augmented maps."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import h5py
import numpy as np
import open3d as o3d
import torch

from omnigibson.learning.datas.lerobot_dataset import BehaviorLeRobotDataset
from serf_b1k.utils import (
    canonicalize_neural_point_root,
    equal_per_instance_sampling,
    get_pc_norm_params,
    normalize_pc,
    normalize_rgb,
    sample_with_instance_filter,
)

logger = logging.getLogger("b1k")


def _import_robot_utils():
    from serf_b1k.mapping.utils import robot as robot_utils

    return robot_utils


def _resolve_scalar_frame_index(raw_frame_index, timestamp, fps: int) -> int:
    """Resolve one scalar lerobot frame index."""
    if raw_frame_index is None:
        ts = timestamp.item() if isinstance(timestamp, torch.Tensor) else timestamp
        return round(float(ts) * fps)

    if isinstance(raw_frame_index, torch.Tensor):
        if raw_frame_index.ndim != 0:
            raise ValueError(
                "4D map training expects a scalar frame_index. "
                "Vector frame_index inputs are unsupported."
            )
        return int(raw_frame_index.item())

    if isinstance(raw_frame_index, np.ndarray):
        if raw_frame_index.ndim != 0:
            raise ValueError(
                "4D map training expects a scalar frame_index. "
                "Vector frame_index inputs are unsupported."
            )
        return int(raw_frame_index.item())

    return int(raw_frame_index)


class _BaseFeatureMapDataset(BehaviorLeRobotDataset):
    """Shared setup, HDF5 reads, sampling, and output assignment."""

    def __init__(
        self,
        *args,
        map_dataset_root_path: Optional[str] = None,
        num_points: int = 24576,
        instance_keep_all_categories: Optional[tuple] = None,
        instance_budget_categories: Optional[tuple] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert map_dataset_root_path is not None, (
            "map_dataset_root_path must be specified"
        )
        assert len(self.task_indices) == 1, (
            "Only a single task is supported per dataset instance"
        )

        canonical_root = Path(canonicalize_neural_point_root(map_dataset_root_path))
        self.task_idx = f"task-{self.task_indices[0]:>04}"
        self.map_dataset_root_path = str(canonical_root / self.task_idx / "train")
        self.num_points = num_points
        self.instance_keep_all_categories = instance_keep_all_categories
        self.instance_budget_categories = instance_budget_categories

        self._current_ep_idx: Optional[int] = None
        self._cache: Optional[Dict] = None

    @staticmethod
    def _read_any(f: h5py.File, keys: tuple[str, ...]) -> Optional[np.ndarray]:
        for key in keys:
            if key in f:
                return f[key][:]
        return None

    def _episode_path(self, ep_idx: int) -> Path:
        return Path(self.map_dataset_root_path) / f"episode_{ep_idx:08d}.hdf5"

    def _read_common_hdf5(
        self,
        f: h5py.File,
        h5_path: Path,
        *,
        require_frame_indices: bool,
    ) -> Dict:
        required_keys = ["initial_points", "initial_instance_ids"]
        if require_frame_indices:
            required_keys.append("frame_indices")

        for key in required_keys:
            if key not in f:
                raise KeyError(f"{h5_path} missing required dataset: {key}")

        xyz = f["initial_points"][:].astype(np.float32)
        ids = f["initial_instance_ids"][:].astype(np.int64)

        feat = self._read_any(
            f, ("initial_features", "features", "latent_features", "latent")
        )
        if feat is not None:
            assert feat.shape[0] == xyz.shape[0], (
                f"Feature array length mismatch: "
                f"feat.shape[0]={feat.shape[0]}, xyz.shape[0]={xyz.shape[0]}"
            )
            feat = feat.astype(np.float32)

        rgb = self._read_any(f, ("rgbs", "colors", "rgb"))
        rgb = rgb.astype(np.float32) if rgb is not None else None

        id_to_name = None
        if "instance_id_to_name" in f.attrs:
            id_to_name = json.loads(f.attrs["instance_id_to_name"])

        payload = {
            "xyz": xyz,
            "ids": ids,
            "feat": feat,
            "rgb": rgb,
            "id_to_name": id_to_name,
        }
        if require_frame_indices:
            payload["frame_indices"] = f["frame_indices"][:].astype(np.int64)
        return payload

    def _ensure_episode_loaded(self, ep_idx: int) -> None:
        if ep_idx != self._current_ep_idx:
            self._load_episode_hdf5(ep_idx)
            self._current_ep_idx = ep_idx

    def _load_episode_hdf5(self, ep_idx: int) -> None:
        raise NotImplementedError

    def _sample_points(
        self,
        *,
        xyz: np.ndarray,
        ids: np.ndarray,
        feat: Optional[np.ndarray],
        rgb: Optional[np.ndarray],
        id_to_name: Optional[dict],
        total_n: Optional[int] = None,
        keep_instance_ids: bool = False,
    ) -> Dict[str, Optional[np.ndarray]]:
        total_n = self.num_points if total_n is None else total_n
        data: Dict[str, Optional[np.ndarray]] = {"xyz": xyz}
        if feat is not None:
            data["feat"] = feat
        if rgb is not None:
            data["rgb"] = rgb

        if id_to_name is not None and self.instance_keep_all_categories:
            sampled = sample_with_instance_filter(
                total_n=total_n,
                instance_ids=ids,
                data=data,
                id_to_name=id_to_name,
                keep_all_categories=self.instance_keep_all_categories,
                budget_categories=self.instance_budget_categories or (),
            )
        else:
            sampled = equal_per_instance_sampling(
                total_n=total_n,
                instance_ids=ids,
                data=data,
            )
            if not keep_instance_ids:
                sampled.pop("_instance_ids", None)

        return sampled

    def _normalize_sampled(
        self, sampled: Dict[str, Optional[np.ndarray]]
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        xyz_o3d = o3d.core.Tensor(sampled["xyz"], dtype=o3d.core.float32)
        xyz_norm = normalize_pc(xyz_o3d, self.task_idx).numpy().astype(np.float32)

        rgb_norm = None
        if sampled.get("rgb") is not None:
            rgb_norm = normalize_rgb(sampled["rgb"]).astype(np.float32)

        feat_out = sampled.get("feat")
        if feat_out is not None:
            feat_out = feat_out.astype(np.float32)

        return xyz_norm, feat_out, rgb_norm

    def _sample_and_normalize_env(
        self,
        xyz: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        sampled = self._sample_points(
            xyz=xyz,
            ids=self._cache["ids"],
            feat=self._cache["feat"],
            rgb=self._cache["rgb"],
            id_to_name=self._cache.get("id_to_name"),
        )
        return self._normalize_sampled(sampled)

    def _set_point_outputs(
        self,
        item: dict,
        xyz_norm: np.ndarray,
        feat_out: Optional[np.ndarray],
        rgb_norm: Optional[np.ndarray],
    ) -> None:
        item["observation.points.xyz"] = xyz_norm
        item["observation.points.feat"] = feat_out
        item["observation.points.rgb"] = rgb_norm

        offset, scale = get_pc_norm_params(self.task_idx)
        item["observation.points.pc_norm_offset"] = offset
        item["observation.points.pc_norm_scale"] = np.array([scale], dtype=np.float32)


class BehaviorLeRobotDataset_3D_EnvFeatMap(_BaseFeatureMapDataset):
    """Dataset for static 3D environment feature maps."""

    VALID_MAP_INPUT_TYPES = ("3d_env_feat_map",)

    def __init__(
        self,
        *args,
        map_input_type: str = "3d_env_feat_map",
        **kwargs,
    ):
        assert map_input_type in self.VALID_MAP_INPUT_TYPES, (
            f"map_input_type must be one of {self.VALID_MAP_INPUT_TYPES}, "
            f"got '{map_input_type}'"
        )
        self.map_input_type = map_input_type
        super().__init__(*args, **kwargs)

    def _load_episode_hdf5(self, ep_idx: int) -> None:
        h5_path = self._episode_path(ep_idx)
        if not h5_path.exists():
            raise FileNotFoundError(f"HDF5 not found: {h5_path}")

        with h5py.File(h5_path, "r") as f:
            self._cache = self._read_common_hdf5(
                f, h5_path, require_frame_indices=False
            )

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        ep_idx = int(item["episode_index"])
        self._ensure_episode_loaded(ep_idx)

        xyz_norm, feat_out, rgb_norm = self._sample_and_normalize_env(
            self._cache["xyz"]
        )
        self._set_point_outputs(item, xyz_norm, feat_out, rgb_norm)
        return item


class BehaviorLeRobotDataset_4D_EnvFeatMap(_BaseFeatureMapDataset):
    """Dataset for 4D environment feature maps with per-frame transforms."""

    def __init__(
        self,
        *args,
        apply_transforms: bool = True,
        frame_sync_mode: str = "nearest",
        **kwargs,
    ):
        assert frame_sync_mode in ("nearest", "floor"), (
            f"frame_sync_mode must be 'nearest' or 'floor', got '{frame_sync_mode}'"
        )
        self.apply_transforms = apply_transforms
        self.frame_sync_mode = frame_sync_mode
        super().__init__(*args, **kwargs)

    def _load_episode_hdf5(self, ep_idx: int) -> None:
        h5_path = self._episode_path(ep_idx)
        if not h5_path.exists():
            raise FileNotFoundError(
                f"4D HDF5 not found: {h5_path}\n"
                "Run neural_point_tracking.py first to generate tracking results."
            )

        with h5py.File(h5_path, "r") as f:
            cache = self._read_common_hdf5(
                f, h5_path, require_frame_indices=True
            )

            transforms: Dict[int, np.ndarray] = {}
            if "transforms" in f:
                for inst_id_str, ds in f["transforms"].items():
                    inst_id = int(inst_id_str)
                    transform_arr = ds[:].astype(np.float32)
                    assert transform_arr.shape[1:] == (4, 4), (
                        f"transforms[{inst_id_str}] has unexpected shape "
                        f"{transform_arr.shape}, expected (T, 4, 4)"
                    )
                    transforms[inst_id] = transform_arr

        unique_ids: Set[int] = set(cache["ids"].tolist())
        tracked_ids: Set[int] = set(transforms.keys())
        cache["transforms"] = transforms
        cache["static_ids"] = unique_ids - tracked_ids
        self._cache = cache

    def _find_frame_t(self, lerobot_frame_idx: int) -> int:
        frame_indices = self._cache["frame_indices"]
        if self.frame_sync_mode == "nearest":
            return int(np.argmin(np.abs(frame_indices - lerobot_frame_idx)))

        candidates = np.where(frame_indices <= lerobot_frame_idx)[0]
        return int(candidates[-1]) if len(candidates) > 0 else 0

    def _apply_transforms_at_t(self, t: int) -> np.ndarray:
        xyz = self._cache["xyz"]
        ids = self._cache["ids"]
        xyz_t = xyz.copy()

        for inst_id, transform_arr in self._cache["transforms"].items():
            t_clamped = min(t, transform_arr.shape[0] - 1)
            transform = transform_arr[t_clamped]
            rotation = transform[:3, :3]
            translation = transform[:3, 3]

            mask = ids == inst_id
            if not mask.any():
                continue

            xyz_t[mask] = xyz_t[mask] @ rotation.T + translation

        return xyz_t

    def _resolve_item_t(self, item: dict) -> int:
        lerobot_frame_idx = _resolve_scalar_frame_index(
            item.get("frame_index", None),
            item.get("timestamp"),
            self.fps,
        )
        return self._find_frame_t(lerobot_frame_idx)

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        ep_idx = int(item["episode_index"])
        self._ensure_episode_loaded(ep_idx)

        t = self._resolve_item_t(item)
        xyz = (
            self._apply_transforms_at_t(t)
            if self.apply_transforms
            else self._cache["xyz"].copy()
        )

        xyz_norm, feat_out, rgb_norm = self._sample_and_normalize_env(xyz)
        self._set_point_outputs(item, xyz_norm, feat_out, rgb_norm)

        if not self.apply_transforms:
            item["observation.points.transforms"] = {
                str(inst_id): transform_arr[min(t, transform_arr.shape[0] - 1)]
                for inst_id, transform_arr in self._cache["transforms"].items()
            }
            item["observation.points.t"] = t

        return item


class BehaviorLeRobotDataset_4D_EnvRobotFeatMap(BehaviorLeRobotDataset_4D_EnvFeatMap):
    """Dataset for 4D environment feature maps augmented with robot points."""

    def __init__(
        self,
        *args,
        robot_model_root_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert robot_model_root_path is not None, (
            "robot_model_root_path must be specified for 4d_env_robot_feat_map"
        )
        self.robot_model_root_path = robot_model_root_path
        self._robot_sampler = None
        self._robot_features: Optional[np.ndarray] = None

    def _ensure_robot_assets_loaded(self) -> None:
        if self._robot_sampler is not None:
            return

        robot_utils = _import_robot_utils()
        robot_dir = (
            Path(self.robot_model_root_path)
            / self.task_idx
            / "neural_points"
            / "robot"
        )
        urdf_path = (
            Path(self.robot_model_root_path).parent / "robot" / "urdf" / "r1pro.urdf"
        )

        sampler_path = robot_dir / "sampler.npz"
        if not sampler_path.exists():
            raise FileNotFoundError(f"Robot sampler not found: {sampler_path}")
        if not urdf_path.exists():
            raise FileNotFoundError(f"Robot URDF not found: {urdf_path}")
        self._robot_sampler = robot_utils.RobotSurfaceSampler.load(
            str(sampler_path), str(urdf_path)
        )

        npt_path = robot_dir / "robot_neural_points.pt"
        if not npt_path.exists():
            raise FileNotFoundError(f"Robot neural points not found: {npt_path}")
        checkpoint = torch.load(npt_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "features" in checkpoint:
            self._robot_features = (
                checkpoint["features"].detach().numpy().astype(np.float32)
            )
        else:
            state = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()
            self._robot_features = state["features"].detach().numpy().astype(np.float32)

        logger.info(
            f"Loaded robot assets for {self.task_idx}: "
            f"sampler={self._robot_sampler.local_points_homo.shape[0]} pts, "
            f"features={self._robot_features.shape}"
        )

    def _compute_robot_points(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        robot_utils = _import_robot_utils()
        cfg = robot_utils.state_to_urdf_cfg(state)
        base_tf = robot_utils.state_to_base_transform_matrix(state)
        robot_xyz, _ = self._robot_sampler.get_points(cfg, base_tf)
        return robot_xyz.astype(np.float32), self._robot_features

    def _sample_and_normalize_with_robot(
        self,
        xyz: np.ndarray,
        robot_xyz: np.ndarray,
        robot_feat: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        ids = self._cache["ids"]
        feat = self._cache["feat"]
        rgb = self._cache["rgb"]
        id_to_name = self._cache.get("id_to_name")

        if id_to_name is not None and self.instance_keep_all_categories:
            env_budget = max(0, self.num_points - robot_xyz.shape[0])
            if env_budget == 0:
                logger.warning(
                    f"Robot points ({robot_xyz.shape[0]}) >= num_points "
                    f"({self.num_points}). No env points will be sampled."
                )

            sampled_env = self._sample_points(
                xyz=xyz,
                ids=ids,
                feat=feat,
                rgb=rgb,
                id_to_name=id_to_name,
                total_n=env_budget,
            )

            sampled = {
                "xyz": np.concatenate([sampled_env["xyz"], robot_xyz], axis=0),
                "feat": None,
                "rgb": None,
            }
            if sampled_env.get("feat") is not None and robot_feat is not None:
                sampled["feat"] = np.concatenate(
                    [sampled_env["feat"], robot_feat], axis=0
                )
            robot_mask = np.concatenate(
                [
                    np.zeros(sampled_env["xyz"].shape[0], dtype=np.bool_),
                    np.ones(robot_xyz.shape[0], dtype=np.bool_),
                ]
            )
        else:
            robot_id = int(ids.max()) + 1 if len(ids) > 0 else 0
            robot_ids = np.full(robot_xyz.shape[0], robot_id, dtype=ids.dtype)

            all_xyz = np.concatenate([xyz, robot_xyz], axis=0)
            all_ids = np.concatenate([ids, robot_ids], axis=0)
            all_feat = (
                np.concatenate([feat, robot_feat], axis=0)
                if feat is not None and robot_feat is not None
                else None
            )

            sampled = self._sample_points(
                xyz=all_xyz,
                ids=all_ids,
                feat=all_feat,
                rgb=None,
                id_to_name=None,
                keep_instance_ids=True,
            )
            robot_mask = sampled.pop("_instance_ids") == robot_id

        xyz_norm, feat_out, rgb_norm = self._normalize_sampled(sampled)
        return xyz_norm, feat_out, rgb_norm, robot_mask

    def __getitem__(self, idx: int) -> dict:
        item = BehaviorLeRobotDataset.__getitem__(self, idx)
        ep_idx = int(item["episode_index"])
        self._ensure_episode_loaded(ep_idx)
        self._ensure_robot_assets_loaded()

        obs_state = item["observation.state"]
        if isinstance(obs_state, torch.Tensor):
            obs_state = obs_state.numpy()
        obs_state = obs_state.astype(np.float32)

        t = self._resolve_item_t(item)
        env_xyz = (
            self._apply_transforms_at_t(t)
            if self.apply_transforms
            else self._cache["xyz"].copy()
        )
        robot_xyz, robot_feat = self._compute_robot_points(obs_state)
        xyz_norm, feat_out, rgb_norm, robot_mask = self._sample_and_normalize_with_robot(
            env_xyz, robot_xyz, robot_feat
        )

        self._set_point_outputs(item, xyz_norm, feat_out, rgb_norm)
        item["observation.points.robot_mask"] = robot_mask
        return item

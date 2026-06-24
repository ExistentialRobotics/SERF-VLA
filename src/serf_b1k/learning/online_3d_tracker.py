import logging
from collections import deque
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import h5py
import numpy as np
import open3d as o3d
import torch

import imageio

from omnigibson.learning.serf_b1k_utils import (
    equal_per_instance_sampling,
    normalize_pc,
    normalize_rgb,
    read_instance_id_to_name,
    read_optional_dataset,
    sample_with_instance_filter,
)

from serf_b1k.mapping.tracking.utils import (
    extract_keypoints_automatically,
    get_dense_observation_cloud,
    render_2d_frame,
    run_cotracker_offline,
    sample_random_mask_points,
)
from serf_b1k.mapping.utils.geometry import (
    refine_registration_icp,
    solve_rigid_transform,
    unproject_depth_to_world,
)
from serf_b1k.mapping.utils.instance_ids import (
    # build_cross_episode_id_mapping,
    filter_ids_by_keywords,
    resolve_names_to_ids,
)
from serf_b1k.mapping.visualization.pca import (
    PCA_CHANNEL_LABELS,
    PCA_CHANNEL_PERMUTATIONS,
    compute_pca_colors_higher,
    permute_pca_colors,
)

logger = logging.getLogger("online_tracker")

RegistrationOutcome = Literal["failed", "stationary", "changed"]


@dataclass(frozen=True)
class CommitChunkResult:
    processed: bool
    geometry_changed: bool

# Match visualization/vis_multiple_pca.py default mode: PCs (2,3,4) + GRB order.
_DEFAULT_PCA_CHANNEL_ORDER = PCA_CHANNEL_PERMUTATIONS[
    PCA_CHANNEL_LABELS.index("GRB")
]
_FALLBACK_REQUIRED_VISIBILITY_VIEW_SUBSTRINGS = ("wrist",)


def _find_neighbors_chunked(
    points: torch.Tensor,
    distance: float,
    chunk_size: int = 4096,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    num_points = points.shape[0]
    if num_points == 0:
        return None, None

    radius_sq = float(distance * distance)
    rows = []
    cols = []
    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
        dist_sq = torch.cdist(points[start:end], points).square()
        mask = (dist_sq <= radius_sq) & (dist_sq > 0)
        idx = torch.nonzero(mask, as_tuple=False)
        if idx.numel() == 0:
            continue
        rows.append(idx[:, 0] + start)
        cols.append(idx[:, 1])

    if not rows:
        return None, None
    return torch.cat(rows), torch.cat(cols)


def _refine_instance_ids_by_graph(
    points: np.ndarray,
    instance_ids: np.ndarray,
    distance: float,
    majority_ratio: float,
    device: str,
) -> np.ndarray:
    if points.size == 0 or instance_ids.size == 0:
        return instance_ids

    pts = torch.from_numpy(points).to(device=device, dtype=torch.float32)
    ids = torch.from_numpy(instance_ids).to(device=device, dtype=torch.long)
    active_indices = torch.nonzero(ids >= 0, as_tuple=False).squeeze(1)
    if active_indices.numel() == 0:
        return instance_ids

    pts = pts[active_indices]
    ids = ids[active_indices]

    try:
        from torch_cluster import radius_graph

        edge_index = radius_graph(pts, r=distance, loop=False)
        neighbor_i, neighbor_j = edge_index[0], edge_index[1]
    except ImportError:
        neighbor_i, neighbor_j = _find_neighbors_chunked(pts, distance)

    if neighbor_i is None or neighbor_i.numel() == 0:
        return instance_ids

    neighbor_counts = torch.zeros(
        pts.shape[0], dtype=torch.long, device=pts.device
    )
    neighbor_counts.scatter_add_(0, neighbor_i, torch.ones_like(neighbor_i))

    unique_ids, inverse = ids.unique(return_inverse=True)
    num_unique = unique_ids.shape[0]
    id_counts = torch.zeros(
        pts.shape[0], num_unique, dtype=torch.long, device=pts.device
    )
    flat_idx = neighbor_i * num_unique + inverse[neighbor_j]
    id_counts.view(-1).scatter_add_(0, flat_idx, torch.ones_like(flat_idx))

    dominant_counts, dominant_compact = id_counts.max(dim=1)
    dominant_ids = unique_ids[dominant_compact]
    ratios = dominant_counts.float() / neighbor_counts.float().clamp(min=1)
    update_mask = (
        (dominant_ids != ids)
        & (ratios >= majority_ratio)
        & (neighbor_counts > 0)
    )
    if not bool(update_mask.any()):
        return instance_ids

    refined = torch.from_numpy(instance_ids.copy()).to(
        device=pts.device, dtype=torch.long
    )
    refined_active = ids.clone()
    refined_active[update_mask] = dominant_ids[update_mask]
    refined[active_indices] = refined_active
    return refined.cpu().numpy().astype(np.int64)

def _canonicalize_training_name(name: str) -> str:
    if name == "background":
        return "background"

    parts = name.strip("/").split("/")

    # e.g. /World/scene_0/bed_ivdnny_0/base_link/visuals
    # -> bed_ivdnny_0
    if len(parts) >= 3:
        return parts[2]

    return name

def build_cross_episode_id_mapping(
    online_id_map: Dict[str, str],
    training_id_map: Dict[str, str],
) -> Dict[int, int]:
    name_to_train_id = {}
    for tid, raw_name in training_id_map.items():
        canon_name = _canonicalize_training_name(raw_name)
        name_to_train_id[canon_name] = int(tid)

    mapping = {}
    for online_id_str, online_name in online_id_map.items():
        if online_name in name_to_train_id:
            mapping[int(online_id_str)] = name_to_train_id[online_name]
    return mapping

@dataclass
class OnlineTrackerConfig:
    """Configuration for OnlineTracker.

    NOTE: Field defaults here are fallbacks only. In production, all fields
    are set explicitly by ``_build_online_tracker_config()`` in
    ``eval_4d_env_feat_serf_b1k.py`` using ExternalTrackingConfig as the source of
    truth. Prefer ``from_external_config()`` for direct instantiation.
    """

    # Defaults match serf_b1k.mapping TrackingConfig.
    buffer_size: int = 8
    frame_step: int = 5
    device: str = "cuda"
    views: tuple[str, ...] = ("head", "left_wrist", "right_wrist")
    target_instance_names: tuple[str, ...] = ()
    save_tracking_video: bool = False
    output_tracking_2d_video_path: Optional[str] = None
    output_tracking_3d_video_path: Optional[str] = None
    tracking_video_fps: int = 20
    tracking_panel_size: tuple[int, int] = (720, 1280)
    num_keypoints_per_instance: int = 30
    visibility_threshold: float = 0.8
    confidence_threshold: float = 0.7
    fgr_threshold: float = 0.05
    icp_threshold: float = 0.02
    icp_min_points: int = 200
    icp_max_iterations: int = 30
    rescue_icp_max_iterations: int = 60
    stationary_threshold: float = 0.02
    fallback_lookback: int = 20
    min_observed_keypoints: int = 15
    instance_keep_all_categories: tuple[str, ...] | None = None
    instance_budget_categories: tuple[str, ...] | None = None
    graph_refine_distance: float = 0.02
    graph_refine_majority_ratio: float = 0.8
    rescue_drift_threshold: float = 0.10
    rescue_acceptance_threshold: float = 0.04
    exclude_categories: tuple[str, ...] = (
        "background",
        "wall",
        "floors",
        "ceiling",
        "roof",
        "door",
        "window",
        "fence",
        "lawn",
        "electric",
    )
    erosion_iters: int = 2
    kp_quality_level: float = 0.01
    kp_min_distance: int = 20
    kp_block_size: int = 7
    min_instance_obs_points: int = 50
    obs_downsample_factor: int = 4
    save_frame_min: Optional[int] = None
    save_frame_max: Optional[int] = None
    camera_extrinsic: Optional[np.ndarray] = None

    @classmethod
    def from_external_config(cls, ext: object) -> "OnlineTrackerConfig":
        """Create from an ExternalTrackingConfig, inheriting all defaults.

        Use this instead of ``OnlineTrackerConfig()`` to ensure defaults
        stay in sync with serf_b1k.mapping.
        """
        return cls(
            buffer_size=int(ext.buffer_size),
            frame_step=int(ext.frame_step),
            device=str(ext.device),
            views=tuple(str(v) for v in ext.views),
            target_instance_names=tuple(str(n) for n in ext.target_instance_names),
            tracking_video_fps=int(ext.video_fps),
            num_keypoints_per_instance=int(ext.max_total_points_per_instance),
            visibility_threshold=float(ext.visibility_threshold),
            confidence_threshold=float(ext.confidence_threshold),
            fgr_threshold=float(ext.fgr_threshold),
            icp_threshold=float(ext.icp_threshold),
            icp_min_points=int(ext.icp_min_points),
            icp_max_iterations=int(ext.icp_max_iterations),
            rescue_icp_max_iterations=int(ext.rescue_icp_max_iterations),
            stationary_threshold=float(ext.stationary_threshold),
            fallback_lookback=int(ext.fallback_lookback),
            min_observed_keypoints=int(ext.min_observed_keypoints),
            graph_refine_distance=float(ext.graph_refine_distance),
            graph_refine_majority_ratio=float(ext.graph_refine_majority_ratio),
            rescue_drift_threshold=float(ext.rescue_drift_threshold),
            rescue_acceptance_threshold=float(ext.rescue_acceptance_threshold),
            exclude_categories=tuple(str(n) for n in ext.exclude_categories),
            erosion_iters=int(ext.erosion_iters),
            kp_quality_level=float(ext.kp_quality_level),
            kp_min_distance=int(ext.kp_min_distance),
            kp_block_size=int(ext.kp_block_size),
            camera_extrinsic=ext.camera_extrinsic,
        )


class OnlineTracker:
    def __init__(self, config: OnlineTrackerConfig = None) -> None:
        self.config = config or OnlineTrackerConfig()
        self.config.buffer_size = max(2, int(self.config.buffer_size))
        self.config.frame_step = max(1, int(self.config.frame_step))
        self._device = self.config.device
        if self._device == "cuda" and not torch.cuda.is_available():
            self._device = "cpu"

        self._cotracker = None
        self.initialized = False

        self._initial_points: Optional[np.ndarray] = None
        self._initial_features: Optional[np.ndarray] = None
        self._initial_rgb: Optional[np.ndarray] = None
        self._initial_instance_ids: Optional[np.ndarray] = None
        self._current_points: Optional[np.ndarray] = None
        self._instance_point_indices: Dict[int, np.ndarray] = {}
        self._cumulative_transforms: Dict[int, np.ndarray] = {}
        self._committed_cumulative: Dict[int, np.ndarray] = {}
        self._pts_to_anchor: Dict[int, np.ndarray] = {}
        self._instance_centroids: Dict[int, np.ndarray] = {}
        self._last_observed_frames: Dict[int, int] = {}

        self._task_idx_str: Optional[str] = None
        self._map_input_type: Optional[str] = None
        self._num_points: int = 24576
        self._training_id_map: Dict[str, str] = {}
        self._online_to_train_id: Dict[int, int] = {}
        self._excluded_train_ids: set[int] = set()
        self._tracked_online_ids: list[int] = []
        self._target_online_ids: set[int] = set()
        self._instance_particles: Dict[int, np.ndarray] = {}

        self._rgb_buffer: Dict[str, deque] = {}
        self._depth_buffer: Dict[str, deque] = {}
        self._seg_buffer: Dict[str, deque] = {}
        self._pose_buffer: Dict[str, deque] = {}
        self._tracking_2d_writer = None
        self._tracking_3d_writer = None
        self._video_frame_count = 0
        self._current_output_2d_path: Optional[str] = None
        self._current_output_3d_path: Optional[str] = None
        self._pca_colors: Optional[np.ndarray] = None
        self._robot_pca_colors: Optional[np.ndarray] = None
        self._o3d_vis = None
        self._o3d_pcd = None
        self._o3d_mat = None
        self._active_mask: Optional[np.ndarray] = None
        self._robot_viz_features: Optional[np.ndarray] = None
        self._robot_viz_points: Optional[np.ndarray] = None
        self._step_count = 0
        self._sample_count = 0
        self._last_flush_sample_idx = -1
        self._pending_anchor_sample_idx = 0
        self._moved_instances: Dict[int, int] = {}
        self._iteration_count: int = 0

    def ensure_buffer_size(self, min_buffer_size: int) -> None:
        """Grow tracker buffers so the pending anchor cannot be evicted."""
        min_buffer_size = max(2, int(min_buffer_size))
        if min_buffer_size <= self.config.buffer_size:
            return

        self.config.buffer_size = min_buffer_size
        for buffer_name in (
            "_rgb_buffer",
            "_depth_buffer",
            "_seg_buffer",
            "_pose_buffer",
        ):
            buffers = getattr(self, buffer_name)
            for view, buf in list(buffers.items()):
                buffers[view] = deque(buf, maxlen=min_buffer_size)

        logger.info("[4D] Grew tracking buffer to %d frames", min_buffer_size)

    def _buffer_frame_indices(self) -> tuple[int, int]:
        """Return (anchor_t, target_t) indices into the current buffer.

        Returns ``(-1, -1)`` when the buffer has fewer than 2 frames,
        meaning no valid anchor/target pair exists.
        """
        buffer_len = len(next(iter(self._rgb_buffer.values()), ()))
        if buffer_len < 2:
            return -1, -1
        anchor_t = min(max(0, self.config.buffer_size - 2), buffer_len - 2)
        target_t = min(max(anchor_t + 1, self.config.buffer_size - 1), buffer_len - 1)
        return anchor_t, target_t

    def _pending_buffer_frame_indices(self) -> tuple[int, int]:
        """Return the pending flush window relative to the current buffers."""
        buffer_len = len(next(iter(self._rgb_buffer.values()), ()))
        if buffer_len < 2 or self._sample_count < 2:
            return -1, -1

        target_global = self._sample_count - 1
        first_global = max(0, self._sample_count - buffer_len)
        if self._pending_anchor_sample_idx < first_global:
            raise RuntimeError(
                "Pending anchor fell out of tracker buffer before flush."
            )
        anchor_global = self._pending_anchor_sample_idx
        if anchor_global >= target_global:
            return -1, -1
        return (
            self._sample_global_to_buffer_index(anchor_global, buffer_len),
            self._sample_global_to_buffer_index(target_global, buffer_len),
        )

    def _next_pending_anchor_after_flush(self, latest_sample_idx: int) -> int:
        if latest_sample_idx < 0:
            return 0

        if self._last_flush_sample_idx >= 0:
            new_samples = latest_sample_idx - self._last_flush_sample_idx
        else:
            new_samples = latest_sample_idx - self._pending_anchor_sample_idx + 1
        new_samples = max(1, int(new_samples))

        overlap_samples = max(1, int(self.config.buffer_size) - new_samples)
        return max(0, latest_sample_idx - overlap_samples + 1)

    def _consume_pending_window(self) -> None:
        if self._sample_count <= 0:
            return

        latest_sample_idx = self._sample_count - 1
        if latest_sample_idx > self._pending_anchor_sample_idx:
            self._pending_anchor_sample_idx = self._next_pending_anchor_after_flush(
                latest_sample_idx
            )
        self._last_flush_sample_idx = latest_sample_idx

    def _prune_moved_instances(self) -> None:
        cutoff = self._iteration_count - self.config.fallback_lookback
        if cutoff <= 0:
            return
        self._moved_instances = {
            iid: it for iid, it in self._moved_instances.items() if it >= cutoff
        }

    def _sample_global_to_buffer_index(
        self, sample_idx: int, buffer_len: int
    ) -> int:
        """Map a sampled-frame index to its current buffer index.

        The first sampled frame primes the full buffer with duplicates. Until we
        have collected ``buffer_len`` unique sampled frames, the unique samples
        are right-aligned in the buffer.
        """
        unique_samples_in_buffer = min(self._sample_count, buffer_len)
        first_global = self._sample_count - unique_samples_in_buffer
        if self._sample_count < buffer_len:
            prefix_len = buffer_len - self._sample_count
            return prefix_len + sample_idx
        return sample_idx - first_global

    def initialize(
        self,
        hdf5_path: str,
        task_idx_str: str,
        map_input_type: str,
        num_points: int,
    ) -> None:
        self._task_idx_str = task_idx_str
        self._map_input_type = map_input_type
        self._num_points = int(num_points)

        with h5py.File(hdf5_path, "r") as f:
            self._initial_points = f["initial_points"][:].astype(np.float32)
            self._initial_instance_ids = f["initial_instance_ids"][:].astype(np.int64)

            rgb = read_optional_dataset(f, ("rgbs", "colors", "rgb"))
            self._initial_rgb = rgb.astype(np.float32) if rgb is not None else None

            feat = read_optional_dataset(
                f,
                ("initial_features", "features", "latent_features", "latent"),
            )
            if feat is not None:
                feat = feat.astype(np.float32)
            self._initial_features = feat

            self._training_id_map = read_instance_id_to_name(f) or {}

        self._initial_instance_ids = _refine_instance_ids_by_graph(
            self._initial_points,
            self._initial_instance_ids,
            distance=self.config.graph_refine_distance,
            majority_ratio=self.config.graph_refine_majority_ratio,
            device=self._device,
        )

        self._instance_point_indices = {
            int(inst_id): np.flatnonzero(self._initial_instance_ids == inst_id)
            for inst_id in np.unique(self._initial_instance_ids)
        }

        self._load_cotracker()
        self.reset()
        self.initialized = True

        logger.info(
            "[4D] OnlineTracker initialized: %d points, %d instances, buffer=%d",
            len(self._initial_points),
            len(self._instance_point_indices),
            self.config.buffer_size,
        )

    def build_id_mapping(self, seg_info: Dict[int, str]) -> None:
        if not self.initialized:
            return

        online_id_map = {str(int(k)): v for k, v in seg_info.items()}
        target_online_ids: Optional[set[int]] = None
        if self.config.target_instance_names:
            target_online_ids = set(
                resolve_names_to_ids(
                    online_id_map,
                    list(self.config.target_instance_names),
                )
            )

        new_mapping = build_cross_episode_id_mapping(
            online_id_map,
            self._training_id_map,
        )
        for online_inst_id, train_inst_id in new_mapping.items():
            self._online_to_train_id.setdefault(online_inst_id, train_inst_id)

        self._excluded_train_ids = filter_ids_by_keywords(
            self._training_id_map,
            self.config.exclude_categories,
        )

        tracked_ids = []
        particles = {}
        for online_inst_id, train_inst_id in sorted(self._online_to_train_id.items()):
            if target_online_ids is not None and online_inst_id not in target_online_ids:
                continue
            if train_inst_id in self._excluded_train_ids:
                continue
            indices = self._instance_point_indices.get(train_inst_id)
            if indices is None or len(indices) == 0:
                continue
            tracked_ids.append(online_inst_id)
            particles[online_inst_id] = indices
            self._cumulative_transforms.setdefault(
                online_inst_id,
                np.eye(4, dtype=np.float64),
            )
            self._committed_cumulative.setdefault(
                online_inst_id,
                np.eye(4, dtype=np.float64),
            )
            self._pts_to_anchor.setdefault(
                online_inst_id,
                np.eye(4, dtype=np.float64),
            )
            if online_inst_id not in self._instance_centroids:
                self._instance_centroids[online_inst_id] = (
                    self._current_points[indices].mean(axis=0).astype(np.float64)
                )
            self._last_observed_frames.setdefault(online_inst_id, 0)

        self._tracked_online_ids = sorted(tracked_ids)
        self._target_online_ids = set(self._tracked_online_ids)
        self._instance_particles = particles

        logger.info(
            "[4D] Built online->train ID mapping for %d tracked instances "
            "(targets=%s)",
            len(self._tracked_online_ids),
            list(self.config.target_instance_names),
        )

    def update(
        self,
        rgb_maps: Dict[str, np.ndarray],
        depth_maps: Dict[str, np.ndarray],
        seg_maps: Dict[str, np.ndarray],
        camera_poses_c2w: Dict[str, np.ndarray],
        intrinsics: Dict[str, Tuple[float, float, float, float]],
    ) -> None:
        if not self.initialized:
            return

        required_views = tuple(self.config.views)
        if not all(
            view in rgb_maps
            and view in depth_maps
            and view in seg_maps
            and view in camera_poses_c2w
                and view in intrinsics
                for view in required_views
        ):
            return
        appended = self.append_observation(
            rgb_maps, depth_maps, seg_maps, camera_poses_c2w
        )
        if not appended:
            return
        self.flush_pending(intrinsics)

    def append_observation(
        self,
        rgb_maps: Dict[str, np.ndarray],
        depth_maps: Dict[str, np.ndarray],
        seg_maps: Dict[str, np.ndarray],
        camera_poses_c2w: Dict[str, np.ndarray],
    ) -> bool:
        if not self.initialized:
            return False

        required_views = tuple(self.config.views)
        if not all(
            view in rgb_maps
            and view in depth_maps
            and view in seg_maps
            and view in camera_poses_c2w
            for view in required_views
        ):
            return False

        should_sample = (self._step_count % self.config.frame_step) == 0
        self._step_count += 1
        if not should_sample:
            return False

        self._prime_or_append_buffers(
            rgb_maps=rgb_maps,
            depth_maps=depth_maps,
            seg_maps=seg_maps,
            camera_poses_c2w=camera_poses_c2w,
        )
        self._sample_count += 1
        return True

    def _count_registration_correspondences(
        self,
        *,
        chunk_results: Dict[str, Dict[str, object]],
        views: tuple[str, ...],
        intrinsics: Dict[str, Tuple[float, float, float, float]],
        inst_id: int,
        anchor_t: int,
        target_t: int,
    ) -> int:
        count = 0
        for view in views:
            res = chunk_results.get(view)
            if not res or not res.get("valid", False):
                continue
            required_keys = (
                "kp_inst_ids",
                "visibilities",
                "confidences",
                "chunk_depths",
                "tracks_xy",
            )
            if any(key not in res for key in required_keys):
                continue
            if view not in intrinsics:
                continue

            kp_inst_ids = res["kp_inst_ids"]
            idx_inst = torch.where(kp_inst_ids == inst_id)[0]
            if len(idx_inst) == 0:
                continue

            vis = res["visibilities"]
            conf = res["confidences"]
            depths = res["chunk_depths"]
            tracks = res["tracks_xy"]
            if (
                anchor_t >= vis.shape[0]
                or target_t >= vis.shape[0]
                or anchor_t >= depths.shape[0]
                or target_t >= depths.shape[0]
                or anchor_t >= tracks.shape[0]
                or target_t >= tracks.shape[0]
            ):
                continue

            vis_anchor = vis[anchor_t]
            vis_target = vis[target_t]
            conf_anchor = conf[anchor_t]
            conf_target = conf[target_t]
            visibility_mask = (
                (vis_anchor[idx_inst] >= self.config.visibility_threshold)
                & (vis_target[idx_inst] >= self.config.visibility_threshold)
                & (conf_anchor[idx_inst] >= self.config.confidence_threshold)
                & (conf_target[idx_inst] >= self.config.confidence_threshold)
            )
            valid_idx = idx_inst[visibility_mask]
            if len(valid_idx) == 0:
                continue

            depth_anchor = depths[anchor_t]
            depth_target = depths[target_t]
            h, w = depth_anchor.shape

            uvs_anchor = tracks[anchor_t][valid_idx]
            uvs_target = tracks[target_t][valid_idx]
            ui_a = torch.round(uvs_anchor[:, 0]).long()
            vi_a = torch.round(uvs_anchor[:, 1]).long()
            ui_t = torch.round(uvs_target[:, 0]).long()
            vi_t = torch.round(uvs_target[:, 1]).long()

            ui_a_s = torch.clamp(ui_a, 0, w - 1)
            vi_a_s = torch.clamp(vi_a, 0, h - 1)
            ui_t_s = torch.clamp(ui_t, 0, w - 1)
            vi_t_s = torch.clamp(vi_t, 0, h - 1)
            valid_anchor = (
                (ui_a >= 0)
                & (ui_a < w)
                & (vi_a >= 0)
                & (vi_a < h)
                & (depth_anchor[vi_a_s, ui_a_s] > 0)
            )
            valid_target = (
                (ui_t >= 0)
                & (ui_t < w)
                & (vi_t >= 0)
                & (vi_t < h)
                & (depth_target[vi_t_s, ui_t_s] > 0)
            )
            count += int((valid_anchor & valid_target).sum().item())
        return count

    @staticmethod
    def _chunk_seg_numpy(res: Dict[str, object], t: int) -> Optional[np.ndarray]:
        chunk_segs = res.get("chunk_segs")
        if chunk_segs is None or t >= len(chunk_segs):
            return None
        seg = chunk_segs[t]
        if torch.is_tensor(seg):
            seg = seg.detach().cpu().numpy()
        return np.asarray(seg)

    def _instance_visible_at_frame(
        self,
        chunk_results: Dict[str, Dict[str, object]],
        views: tuple[str, ...],
        inst_id: int,
        t: int,
    ) -> bool:
        for view in views:
            res = chunk_results.get(view) or {}
            seg = self._chunk_seg_numpy(res, t)
            if seg is None:
                continue
            if np.any(seg == inst_id):
                return True
        return False

    def _review_adjacent_registration_counts(
        self,
        *,
        chunk_results: Dict[str, Dict[str, object]],
        views: tuple[str, ...],
        intrinsics: Dict[str, Tuple[float, float, float, float]],
        local_start_anchor_t: int,
        local_target_t: int,
    ) -> Dict[int, Dict[tuple[int, int], int]]:
        counts: Dict[int, Dict[tuple[int, int], int]] = {}
        for inst_id in self._tracked_online_ids:
            inst_counts = {}
            for target_t in range(local_start_anchor_t + 1, local_target_t + 1):
                anchor_t = target_t - 1
                inst_counts[(anchor_t, target_t)] = (
                    self._count_registration_correspondences(
                        chunk_results=chunk_results,
                        views=views,
                        intrinsics=intrinsics,
                        inst_id=inst_id,
                        anchor_t=anchor_t,
                        target_t=target_t,
                    )
                )
            counts[inst_id] = inst_counts
        return counts

    def _retry_undertracked_instances_once(
        self,
        *,
        chunk_results: Dict[str, Dict[str, object]],
        required_views: tuple[str, ...],
        intrinsics: Dict[str, Tuple[float, float, float, float]],
        retry_instances: set[int],
        buffer_anchor_t: int,
        buffer_target_t: int,
        local_anchor_t: int,
    ) -> None:
        if not retry_instances:
            return

        doubled_budget = self.config.num_keypoints_per_instance * 2
        halved_min_distance = max(1, self.config.kp_min_distance // 2)
        logger.info(
            "[4D] Fallback retry for %d instances with %d points: %s",
            len(retry_instances),
            doubled_budget,
            sorted(retry_instances),
        )

        retry_views: set[str] = set()
        for view in required_views:
            res = chunk_results.get(view) or {}
            seg_anchor = self._chunk_seg_numpy(res, local_anchor_t)
            if seg_anchor is None:
                continue
            if any(np.any(seg_anchor == inst_id) for inst_id in retry_instances):
                retry_views.add(view)

        overrides = {
            iid: (doubled_budget, halved_min_distance) for iid in retry_instances
        }
        for view in retry_views:
            retry_res = self._track_view_buffer(
                view,
                intrinsics,
                budget_overrides=overrides,
                anchor_t=buffer_anchor_t,
                target_t=buffer_target_t,
            )
            if retry_res.get("valid", False):
                chunk_results[view] = retry_res

    def _commit_start_local_anchor_t(
        self,
        *,
        buffer_anchor_t: int,
        buffer_target_t: int,
    ) -> int:
        local_target_t = buffer_target_t - buffer_anchor_t
        if self._last_flush_sample_idx < 0:
            return 0

        buffer_len = len(next(iter(self._rgb_buffer.values()), ()))
        if buffer_len < 2:
            return 0

        first_global = max(0, self._sample_count - buffer_len)
        latest_global = self._sample_count - 1
        previous_latest = self._last_flush_sample_idx
        if previous_latest < first_global:
            return 0
        if previous_latest >= latest_global:
            return local_target_t

        previous_buffer_t = self._sample_global_to_buffer_index(
            previous_latest,
            buffer_len,
        )
        return min(
            max(0, previous_buffer_t - buffer_anchor_t),
            max(0, local_target_t - 1),
        )

    def _commit_chunk_results(
        self,
        chunk_results: Dict[str, Dict[str, object]],
        required_views: tuple[str, ...],
        intrinsics: Dict[str, Tuple[float, float, float, float]],
        buffer_anchor_t: int,
        buffer_target_t: int,
    ) -> CommitChunkResult:
        if not self._tracked_online_ids:
            return CommitChunkResult(processed=False, geometry_changed=False)

        local_target_t = buffer_target_t - buffer_anchor_t
        if local_target_t < 1:
            return CommitChunkResult(processed=False, geometry_changed=False)

        any_valid_keypoints = any(
            bool(res.get("valid", False)) for res in chunk_results.values()
        )

        local_commit_anchor_t = self._commit_start_local_anchor_t(
            buffer_anchor_t=buffer_anchor_t,
            buffer_target_t=buffer_target_t,
        )
        if local_commit_anchor_t >= local_target_t:
            return CommitChunkResult(processed=False, geometry_changed=False)

        retry_instances: set[int] = set()
        if any_valid_keypoints:
            adjacent_counts = self._review_adjacent_registration_counts(
                chunk_results=chunk_results,
                intrinsics=intrinsics,
                views=required_views,
                local_start_anchor_t=local_commit_anchor_t,
                local_target_t=local_target_t,
            )
            recent_moved = {
                iid
                for iid, last_iter in self._moved_instances.items()
                if self._iteration_count - last_iter <= self.config.fallback_lookback
            }
            fallback_visibility_views = tuple(
                view
                for view in required_views
                if any(
                    substring in view
                    for substring in _FALLBACK_REQUIRED_VISIBILITY_VIEW_SUBSTRINGS
                )
            )
            for inst_id in self._tracked_online_ids:
                if inst_id not in recent_moved:
                    continue
                if not any(
                    count < self.config.min_observed_keypoints
                    for count in adjacent_counts.get(inst_id, {}).values()
                ):
                    continue
                if not fallback_visibility_views:
                    continue
                if self._instance_visible_at_frame(
                    chunk_results,
                    fallback_visibility_views,
                    inst_id,
                    local_commit_anchor_t,
                ):
                    retry_instances.add(inst_id)

            if retry_instances:
                self._retry_undertracked_instances_once(
                    chunk_results=chunk_results,
                    required_views=required_views,
                    intrinsics=intrinsics,
                    retry_instances=retry_instances,
                    buffer_anchor_t=buffer_anchor_t,
                    buffer_target_t=buffer_target_t,
                    local_anchor_t=local_commit_anchor_t,
                )

        for inst_id in self._tracked_online_ids:
            self._pts_to_anchor[inst_id] = np.eye(4, dtype=np.float64)
            self._cumulative_transforms[inst_id] = self._committed_cumulative.get(
                inst_id, np.eye(4, dtype=np.float64)
            ).copy()

        processed = False
        geometry_changed = False
        instance_anchors = {
            inst_id: local_commit_anchor_t for inst_id in self._tracked_online_ids
        }
        for target_t in range(local_commit_anchor_t + 1, local_target_t + 1):
            for inst_id in self._tracked_online_ids:
                anchor_t = instance_anchors[inst_id]
                outcome = self._register_instance_for_step(
                    inst_id=inst_id,
                    anchor_t=anchor_t,
                    curr_t=target_t,
                    views=required_views,
                    chunk_results=chunk_results,
                    intrinsics=intrinsics,
                    skip_min_keypoints=(inst_id in retry_instances),
                )
                if outcome in ("changed", "stationary"):
                    processed = True
                    instance_anchors[inst_id] = target_t
                if outcome == "changed":
                    self._moved_instances[inst_id] = self._iteration_count
                    geometry_changed = True
            self._iteration_count += 1
            self._prune_moved_instances()
        return CommitChunkResult(
            processed=processed,
            geometry_changed=geometry_changed,
        )

    def flush_pending(
        self,
        intrinsics: Dict[str, Tuple[float, float, float, float]],
    ) -> bool:
        if not self.initialized:
            return False

        required_views = tuple(self.config.views)
        if not all(view in intrinsics for view in required_views):
            return False
        if not self._tracked_online_ids:
            self._consume_pending_window()
            return False

        anchor_t, target_t = self._pending_buffer_frame_indices()
        if anchor_t < 0 or target_t < 0:
            return False

        chunk_results: Dict[str, Dict[str, object]] = {}
        for view in required_views:
            chunk_results[view] = self._track_view_buffer(
                view,
                intrinsics,
                anchor_t=anchor_t,
                target_t=target_t,
            )

        commit_result = self._commit_chunk_results(
            chunk_results,
            required_views,
            intrinsics,
            anchor_t,
            target_t,
        )
        if not commit_result.processed:
            self._consume_pending_window()
            return False

        self._write_tracking_frame(chunk_results, required_views)
        self._consume_pending_window()
        return commit_result.geometry_changed

    def get_current_points(self) -> Dict[str, Optional[np.ndarray]]:
        xyz = self._current_points
        ids = self._initial_instance_ids
        feat = self._initial_features
        rgb = self._initial_rgb

        data_dict: Dict[str, Optional[np.ndarray]] = {"xyz": xyz}
        if feat is not None:
            data_dict["feat"] = feat
        if rgb is not None:
            data_dict["rgb"] = rgb

        if self._training_id_map and self.config.instance_keep_all_categories:
            sampled = sample_with_instance_filter(
                self._num_points, ids, data_dict, self._training_id_map,
                self.config.instance_keep_all_categories,
                self.config.instance_budget_categories or (),
            )
        else:
            sampled = equal_per_instance_sampling(self._num_points, ids, data_dict)

        xyz_o3d = o3d.core.Tensor(sampled["xyz"], dtype=o3d.core.float32)
        xyz_norm = normalize_pc(xyz_o3d, self._task_idx_str).numpy().astype(np.float32)

        rgb_norm = None
        if sampled.get("rgb") is not None:
            rgb_norm = normalize_rgb(sampled["rgb"]).astype(np.float32)

        feat_out = sampled.get("feat")
        if feat_out is not None:
            feat_out = feat_out.astype(np.float32)

        return {
            "observation.points.xyz": xyz_norm,
            "observation.points.rgb": rgb_norm,
            "observation.points.feat": feat_out,
        }

    def get_current_points_world(self) -> Dict[str, Optional[np.ndarray]]:
        return {
            "xyz": self._current_points.copy(),
            "instance_ids": self._initial_instance_ids.copy(),
            "rgb": None if self._initial_rgb is None else self._initial_rgb.copy(),
            "feat": (
                None if self._initial_features is None else self._initial_features.copy()
            ),
        }

    def reset(self) -> None:
        self.close()
        self._current_output_2d_path = None
        self._current_output_3d_path = None
        self._current_points = (
            None if self._initial_points is None else self._initial_points.copy()
        )
        self._cumulative_transforms = {}
        self._online_to_train_id = {}
        self._tracked_online_ids = []
        self._target_online_ids = set()
        self._instance_particles = {}
        self._committed_cumulative = {}
        self._pts_to_anchor = {}
        self._instance_centroids = {}
        self._last_observed_frames = {}

        self._rgb_buffer = {}
        self._depth_buffer = {}
        self._seg_buffer = {}
        self._pose_buffer = {}
        self._step_count = 0
        self._sample_count = 0
        self._last_flush_sample_idx = -1
        self._pending_anchor_sample_idx = 0
        self._moved_instances = {}
        self._iteration_count = 0
        self._pca_colors = None
        self._robot_pca_colors = None
        self._robot_viz_points = None

    def set_output_tracking_video_paths(
        self,
        path_2d: Optional[str] = None,
        path_3d: Optional[str] = None,
    ) -> None:
        changed = (
            path_2d != self._current_output_2d_path
            or path_3d != self._current_output_3d_path
        )
        if not changed:
            return
        self.close()
        self._current_output_2d_path = path_2d
        self._current_output_3d_path = path_3d

    def close(self) -> None:
        self._close_tracking_writer()

    def _load_cotracker(self) -> None:
        if self._cotracker is not None:
            return

        logger.info("[4D] Loading CoTracker model on %s", self._device)
        self._cotracker = torch.hub.load(
            "facebookresearch/co-tracker",
            "cotracker3_offline",
        ).to(self._device)
        self._cotracker.eval()

    def _prime_or_append_buffers(
        self,
        rgb_maps: Dict[str, np.ndarray],
        depth_maps: Dict[str, np.ndarray],
        seg_maps: Dict[str, np.ndarray],
        camera_poses_c2w: Dict[str, np.ndarray],
    ) -> None:
        for view in self.config.views:
            rgb = rgb_maps.get(view)
            depth = depth_maps.get(view)
            seg = seg_maps.get(view)
            pose = camera_poses_c2w.get(view)
            if rgb is None or depth is None or seg is None or pose is None:
                continue

            if view not in self._rgb_buffer:
                self._rgb_buffer[view] = deque(
                    maxlen=self.config.buffer_size
                )
                self._depth_buffer[view] = deque(
                    maxlen=self.config.buffer_size
                )
                self._seg_buffer[view] = deque(
                    maxlen=self.config.buffer_size
                )
                self._pose_buffer[view] = deque(
                    maxlen=self.config.buffer_size
                )

            rgb_arr = np.asarray(rgb, dtype=np.uint8).copy()
            depth_arr = np.asarray(depth, dtype=np.float32).copy()
            seg_arr = np.asarray(seg, dtype=np.int64).copy()
            pose_arr = np.asarray(pose, dtype=np.float32).copy()

            self._rgb_buffer[view].append(rgb_arr)
            self._depth_buffer[view].append(depth_arr)
            self._seg_buffer[view].append(seg_arr)
            self._pose_buffer[view].append(pose_arr)

    @staticmethod
    def _stack_window(buffer: deque, start: int, end: int) -> np.ndarray:
        return np.stack(tuple(islice(buffer, start, end + 1)), axis=0)

    def _track_view_buffer(
        self,
        view: str,
        intrinsics: Dict[str, Tuple[float, float, float, float]],
        budget_overrides: Optional[Dict[int, Tuple[int, int]]] = None,
        anchor_t: Optional[int] = None,
        target_t: Optional[int] = None,
    ) -> Dict[str, object]:
        if anchor_t is None or target_t is None:
            anchor_t, target_t = self._buffer_frame_indices()
        frames_np = self._stack_window(self._rgb_buffer[view], anchor_t, target_t)
        seg_window = self._stack_window(self._seg_buffer[view], anchor_t, target_t)
        local_anchor_t = 0
        seg_first = seg_window[local_anchor_t]
        chunk_depths = torch.from_numpy(
            self._stack_window(self._depth_buffer[view], anchor_t, target_t)
        ).to(self._device)
        chunk_segs = torch.from_numpy(seg_window).to(self._device)
        chunk_poses = self._stack_window(self._pose_buffer[view], anchor_t, target_t)[:, :3, :4]
        chunk_poses_c2w = torch.from_numpy(chunk_poses).to(self._device)

        view_keypoints = []
        view_kp_inst_ids = []
        for inst_id in self._tracked_online_ids:
            mask = (seg_first == inst_id).astype(np.uint8) * 255
            if np.sum(mask) == 0:
                continue
            if budget_overrides is not None and inst_id in budget_overrides:
                budget, min_dist = budget_overrides[inst_id]
            else:
                budget = self.config.num_keypoints_per_instance
                min_dist = self.config.kp_min_distance
            kps = extract_keypoints_automatically(
                frames_np[local_anchor_t],
                mask,
                max_points=budget,
                erosion_iters=self.config.erosion_iters,
                quality_level=self.config.kp_quality_level,
                min_distance=min_dist,
                block_size=self.config.kp_block_size,
            )
            remaining = budget - len(kps)
            if remaining > 0:
                kps.extend(
                    sample_random_mask_points(
                        mask,
                        remaining,
                        erosion_iters=self.config.erosion_iters,
                        min_distance=min_dist,
                        existing_points=kps,
                    )
                )
            view_keypoints.extend(kps)
            view_kp_inst_ids.extend([inst_id] * len(kps))

        if not view_keypoints:
            return {
                "valid": False,
                "frames_np": frames_np,
                "chunk_depths": chunk_depths,
                "chunk_segs": chunk_segs,
                "chunk_poses_c2w": chunk_poses_c2w,
            }

        tracks_xy, visibilities, confidences = run_cotracker_offline(
            frames_np,
            view_keypoints,
            self._cotracker,
            device=self._device,
        )

        return {
            "valid": True,
            "frames_np": frames_np,
            "chunk_depths": chunk_depths,
            "chunk_segs": chunk_segs,
            "chunk_poses_c2w": chunk_poses_c2w,
            "keypoints": view_keypoints,
            "kp_inst_ids": torch.tensor(view_kp_inst_ids, dtype=torch.long, device=self._device),
            "tracks_xy": tracks_xy,
            "visibilities": visibilities,
            "confidences": confidences,
            "colors_2d": np.random.randint(
                0, 255, (len(view_keypoints), 3), dtype=np.uint8
            ),
        }

    def _compute_pca_colors(self) -> None:
        """Compute PCA-based RGB colors from initial features (once per episode).

        Matches ``visualization/vis_multiple_pca.py``'s default display mode:
        higher-order PCs ``(2, 3, 4)`` with ``GRB`` channel ordering.
        """
        if self._initial_features is None or self._initial_instance_ids is None:
            self._pca_colors = None
            return

        active_mask = np.ones(len(self._initial_instance_ids), dtype=bool)
        if self._training_id_map and self.config.exclude_categories:
            excluded_ids = filter_ids_by_keywords(
                self._training_id_map,
                self.config.exclude_categories,
            )
            for eid in excluded_ids:
                active_mask &= self._initial_instance_ids != eid
        self._active_mask = active_mask

        colors = np.full((len(self._initial_instance_ids), 3), 0.5, dtype=np.float32)
        self._robot_pca_colors = None

        if active_mask.sum() > 0:
            fg_features = torch.from_numpy(self._initial_features[active_mask])
            if self._robot_viz_features is not None and self._robot_viz_features.size > 0:
                combined_features = torch.cat(
                    [
                        fg_features,
                        torch.from_numpy(self._robot_viz_features),
                    ],
                    dim=0,
                )
                combined_colors = compute_pca_colors_higher(
                    combined_features,
                    skip_components=2,
                )
                combined_colors = permute_pca_colors(
                    combined_colors,
                    _DEFAULT_PCA_CHANNEL_ORDER,
                ).astype(np.float32)
                fg_count = fg_features.shape[0]
                fg_colors = combined_colors[:fg_count]
                robot_colors = combined_colors[fg_count:]
                self._robot_pca_colors = robot_colors.astype(np.float32)
            else:
                fg_colors = compute_pca_colors_higher(
                    fg_features,
                    skip_components=2,
                )
                fg_colors = permute_pca_colors(
                    fg_colors,
                    _DEFAULT_PCA_CHANNEL_ORDER,
                ).astype(np.float32)

            colors[active_mask] = fg_colors.astype(np.float32)

        self._pca_colors = colors
        logger.info(
            "[4D] PCA colors computed: %d fg / %d total points",
            int(active_mask.sum()), len(colors),
        )

    def set_robot_visualization_assets(
        self,
        *,
        features: Optional[np.ndarray],
    ) -> None:
        self._robot_viz_features = (
            None if features is None else np.asarray(features, dtype=np.float32)
        )
        self._robot_pca_colors = None
        self._pca_colors = None

    def set_robot_visualization_points(
        self,
        points_xyz: Optional[np.ndarray],
    ) -> None:
        self._robot_viz_points = (
            None if points_xyz is None else np.asarray(points_xyz, dtype=np.float32)
        )

    def _init_o3d_vis(self) -> None:
        """Create a persistent Open3D offscreen renderer (headless / Docker safe).

        Uses ``OffscreenRenderer`` (EGL/OSMesa) instead of the legacy
        ``Visualizer`` (GLFW/X11) so that no display connection is required.
        """
        if self._o3d_vis is not None:
            return

        width = self.config.tracking_panel_size[1]
        height = self.config.tracking_panel_size[0]

        self._o3d_vis = o3d.visualization.rendering.OffscreenRenderer(width, height)
        self._o3d_vis.scene.set_background([0.1, 0.1, 0.1, 1.0])

        self._o3d_mat = o3d.visualization.rendering.MaterialRecord()
        self._o3d_mat.shader = "defaultUnlit"
        self._o3d_mat.point_size = 2.0

        fg_pts = self._current_points[self._active_mask].astype(np.float32)
        fg_colors = self._pca_colors[self._active_mask]
        if (
            self._robot_viz_points is not None
            and self._robot_pca_colors is not None
            and len(self._robot_viz_points) == len(self._robot_pca_colors)
        ):
            fg_pts = np.concatenate([fg_pts, self._robot_viz_points], axis=0)
            fg_colors = np.concatenate([fg_colors, self._robot_pca_colors], axis=0)

        self._o3d_pcd = o3d.geometry.PointCloud()
        self._o3d_pcd.points = o3d.utility.Vector3dVector(fg_pts)
        self._o3d_pcd.colors = o3d.utility.Vector3dVector(fg_colors)
        self._o3d_vis.scene.add_geometry("pcd", self._o3d_pcd, self._o3d_mat)

        self._o3d_setup_camera(fg_pts)

    def _o3d_setup_camera(self, pts: np.ndarray) -> None:
        """Point the offscreen camera at the point cloud.

        Task camera matrices are saved as Open3D world-to-camera extrinsics
        from ``PinholeCameraParameters.extrinsic``.  Preserve that convention
        when using ``OffscreenRenderer``.

        Falls back to a top-down view when ``camera_extrinsic`` is not set.
        """
        width = int(self.config.tracking_panel_size[1])
        height = int(self.config.tracking_panel_size[0])

        if self.config.camera_extrinsic is not None:
            ext = np.asarray(self.config.camera_extrinsic, dtype=np.float64)
            fov_y = np.deg2rad(60.0)
            focal = 0.5 * height / np.tan(0.5 * fov_y)
            intrinsic = np.array(
                [
                    [focal, 0.0, (width - 1) * 0.5],
                    [0.0, focal, (height - 1) * 0.5],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            self._o3d_vis.setup_camera(intrinsic, ext, width, height)
            return

        if pts.shape[0] > 0:
            center = pts.mean(axis=0).astype(np.float64)
        else:
            center = np.zeros(3)
        extent = (
            np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
            if pts.shape[0] > 1
            else 2.0
        )
        eye = center + np.array([0.0, 0.0, float(extent)])
        up = np.array([0.0, 1.0, 0.0])

        self._o3d_vis.setup_camera(
            60.0,                      # vertical fov degrees
            center.tolist(),
            eye.tolist(),
            up.tolist(),
        )

    def _destroy_o3d_vis(self) -> None:
        if self._o3d_vis is not None:
            # OffscreenRenderer has no destroy_window(); just release references.
            self._o3d_vis = None
            self._o3d_pcd = None
            self._o3d_mat = None

    def _render_3d_frame(self) -> Optional[np.ndarray]:
        """Render a single Open3D frame with PCA colors (headless / Docker safe).

        Uses ``OffscreenRenderer.render_to_image()`` instead of the legacy
        ``capture_screen_float_buffer()`` which requires an active X11 display.
        """
        if self._current_points is None:
            return None

        if self._pca_colors is None:
            self._compute_pca_colors()
        if self._pca_colors is None:
            return None

        self._init_o3d_vis()

        fg_pts = self._current_points[self._active_mask].astype(np.float32)
        fg_colors = self._pca_colors[self._active_mask]
        if (
            self._robot_viz_points is not None
            and self._robot_pca_colors is not None
            and len(self._robot_viz_points) == len(self._robot_pca_colors)
        ):
            fg_pts = np.concatenate([fg_pts, self._robot_viz_points], axis=0)
            fg_colors = np.concatenate([fg_colors, self._robot_pca_colors], axis=0)

        # Update geometry in-place: remove and re-add is the OffscreenRenderer API.
        self._o3d_pcd.points = o3d.utility.Vector3dVector(fg_pts)
        self._o3d_pcd.colors = o3d.utility.Vector3dVector(fg_colors)
        self._o3d_vis.scene.remove_geometry("pcd")
        self._o3d_vis.scene.add_geometry("pcd", self._o3d_pcd, self._o3d_mat)

        # Re-apply camera every frame so it follows config changes.
        self._o3d_setup_camera(fg_pts)

        img_o3d = self._o3d_vis.render_to_image()
        return np.asarray(img_o3d)

    def _open_writers_if_needed(self) -> None:
        if not self.config.save_tracking_video:
            return

        fps = self.config.tracking_video_fps

        if self._tracking_2d_writer is None:
            path_2d = (
                self._current_output_2d_path
                or self.config.output_tracking_2d_video_path
            )
            if path_2d:
                Path(path_2d).parent.mkdir(parents=True, exist_ok=True)
                self._tracking_2d_writer = imageio.get_writer(
                    path_2d, fps=fps
                )
                self._current_output_2d_path = path_2d

        if self._tracking_3d_writer is None:
            path_3d = (
                self._current_output_3d_path
                or self.config.output_tracking_3d_video_path
            )
            if path_3d:
                Path(path_3d).parent.mkdir(parents=True, exist_ok=True)
                self._tracking_3d_writer = imageio.get_writer(
                    path_3d, fps=fps
                )
                self._current_output_3d_path = path_3d

    def _write_tracking_frame(
        self,
        chunk_results: Dict[str, Dict[str, object]],
        views: tuple[str, ...],
    ) -> None:
        if not self.config.save_tracking_video or not views:
            return

        cfg = self.config
        if cfg.save_frame_min is not None and self._step_count < cfg.save_frame_min:
            return
        if cfg.save_frame_max is not None and self._step_count > cfg.save_frame_max:
            self._close_tracking_writer()
            return

        first_view = views[0]
        if first_view not in self._rgb_buffer or len(self._rgb_buffer[first_view]) < 2:
            return

        self._open_writers_if_needed()

        t_latest = len(chunk_results[first_view]["frames_np"]) - 1

        # --- 2D tracking frame ---
        if self._tracking_2d_writer is not None:
            frame_2d = render_2d_frame(
                t_latest,
                list(views),
                chunk_results,
                visibility_threshold=self.config.visibility_threshold,
                confidence_threshold=self.config.confidence_threshold,
            )
            self._tracking_2d_writer.append_data(frame_2d)

        # --- 3D PCA point cloud frame (Open3D) ---
        if self._tracking_3d_writer is not None:
            frame_3d = self._render_3d_frame()
            if frame_3d is not None:
                self._tracking_3d_writer.append_data(frame_3d)

        self._video_frame_count += 1

    def _close_tracking_writer(self) -> None:
        if self._tracking_2d_writer is not None:
            self._tracking_2d_writer.close()
            self._tracking_2d_writer = None
        if self._tracking_3d_writer is not None:
            self._tracking_3d_writer.close()
            self._tracking_3d_writer = None
        self._destroy_o3d_vis()
        self._video_frame_count = 0

    def _register_instance_for_step(
        self,
        inst_id: int,
        anchor_t: int,
        curr_t: int,
        views: tuple[str, ...],
        chunk_results: Dict[str, Dict[str, object]],
        intrinsics: Dict[str, Tuple[float, float, float, float]],
        skip_min_keypoints: bool = False,
    ) -> RegistrationOutcome:
        if self._current_points is None:
            return "failed"

        y_idx = self._instance_particles.get(inst_id)
        if y_idx is None or len(y_idx) == 0:
            return "failed"

        all_src = []
        all_tgt = []
        contributing_views = set()

        for view in views:
            res = chunk_results.get(view)
            if not res or not res.get("valid", False):
                continue

            kp_inst_ids = res["kp_inst_ids"]
            idx_inst = torch.where(kp_inst_ids == inst_id)[0]
            if len(idx_inst) == 0:
                continue

            vis_prev = res["visibilities"][anchor_t]
            vis_curr = res["visibilities"][curr_t]
            conf_prev = res["confidences"][anchor_t]
            conf_curr = res["confidences"][curr_t]
            vis_mask = (
                (vis_prev[idx_inst] >= self.config.visibility_threshold)
                & (vis_curr[idx_inst] >= self.config.visibility_threshold)
                & (conf_prev[idx_inst] >= self.config.confidence_threshold)
                & (conf_curr[idx_inst] >= self.config.confidence_threshold)
            )
            valid_idx = idx_inst[vis_mask]
            if len(valid_idx) == 0:
                continue

            depth_prev = res["chunk_depths"][anchor_t]
            depth_curr = res["chunk_depths"][curr_t]
            fx, fy, cx, cy = intrinsics[view]
            h, w = depth_prev.shape

            depth_batch = torch.stack([depth_prev, depth_curr])
            pose_c2w_batch = torch.stack(
                [
                    res["chunk_poses_c2w"][anchor_t],
                    res["chunk_poses_c2w"][curr_t],
                ]
            )
            world_batch = unproject_depth_to_world(
                depth_batch,
                pose_c2w_batch,
                fx,
                fy,
                cx,
                cy,
                original_height=h,
                original_width=w,
            )

            uvs_prev = res["tracks_xy"][anchor_t][valid_idx]
            uvs_curr = res["tracks_xy"][curr_t][valid_idx]
            ui_p = torch.round(uvs_prev[:, 0]).long()
            vi_p = torch.round(uvs_prev[:, 1]).long()
            ui_c = torch.round(uvs_curr[:, 0]).long()
            vi_c = torch.round(uvs_curr[:, 1]).long()

            ui_p_s = torch.clamp(ui_p, 0, w - 1)
            vi_p_s = torch.clamp(vi_p, 0, h - 1)
            ui_c_s = torch.clamp(ui_c, 0, w - 1)
            vi_c_s = torch.clamp(vi_c, 0, h - 1)
            valid_p = (
                (ui_p >= 0)
                & (ui_p < w)
                & (vi_p >= 0)
                & (vi_p < h)
                & (depth_prev[vi_p_s, ui_p_s] > 0)
            )
            valid_c = (
                (ui_c >= 0)
                & (ui_c < w)
                & (vi_c >= 0)
                & (vi_c < h)
                & (depth_curr[vi_c_s, ui_c_s] > 0)
            )
            both_valid = valid_p & valid_c
            if not both_valid.any():
                continue

            pts_prev = world_batch[0, :, vi_p_s[both_valid], ui_p_s[both_valid]].T
            pts_curr = world_batch[1, :, vi_c_s[both_valid], ui_c_s[both_valid]].T
            all_src.append(pts_prev)
            all_tgt.append(pts_curr)
            contributing_views.add(view)

        current_pts_np = np.asarray(self._current_points[y_idx], dtype=np.float64)
        if current_pts_np.shape[0] == 0:
            return "failed"
        center_src = current_pts_np.mean(axis=0)

        dense_obs, _ = get_dense_observation_cloud(
            chunk_results=chunk_results,
            t=curr_t,
            inst_id=inst_id,
            views=list(views),
            intrinsics=intrinsics,
        )
        has_dense = dense_obs is not None and dense_obs.shape[0] > self.config.icp_min_points
        observed_centroid = (
            dense_obs.mean(dim=0).detach().cpu().numpy() if has_dense else None
        )

        def _run_centroid_initialized_icp(
            label: str,
            max_iterations: int,
        ) -> Optional[tuple[np.ndarray, np.ndarray]]:
            try:
                r_candidate, t_candidate, fitness, rmse = refine_registration_icp(
                    current_pts_np,
                    dense_obs,
                    np.eye(3, dtype=np.float64),
                    (observed_centroid - center_src).astype(np.float64),
                    threshold=self.config.icp_threshold,
                    max_iterations=max_iterations,
                    return_metrics=True,
                )
            except Exception as exc:
                logger.warning(
                    "[4D][%s] ICP failed for instance %s: %s",
                    label,
                    inst_id,
                    exc,
                )
                return None

            center_after = r_candidate @ center_src + t_candidate
            residual = np.linalg.norm(center_after - observed_centroid)
            if residual > self.config.rescue_acceptance_threshold:
                logger.info(
                    "[4D][%s] inst %s rejected: residual=%.4fm fitness=%.3f rmse=%.4f",
                    label,
                    inst_id,
                    residual,
                    fitness,
                    rmse,
                )
                return None
            logger.info(
                "[4D][%s] inst %s accepted: residual=%.4fm fitness=%.3f rmse=%.4f",
                label,
                inst_id,
                residual,
                fitness,
                rmse,
            )
            return np.asarray(r_candidate, dtype=np.float64), np.asarray(
                t_candidate,
                dtype=np.float64,
            )

        use_rescue = False
        if has_dense and inst_id in self._instance_centroids:
            drift = np.linalg.norm(
                observed_centroid - self._instance_centroids[inst_id]
            )
            if drift > self.config.rescue_drift_threshold:
                use_rescue = True

        if use_rescue:
            logger.info(
                "[4D][RESCUE] inst %s: drift=%.4fm > %.4fm, attempting rescue ICP",
                inst_id,
                drift,
                self.config.rescue_drift_threshold,
            )
            icp_result = _run_centroid_initialized_icp(
                "RESCUE",
                self.config.rescue_icp_max_iterations,
            )
            if icp_result is None:
                return "failed"
            r_mat, t_vec = icp_result
        else:
            use_fgr = True
            if not all_src:
                if skip_min_keypoints and has_dense:
                    logger.info(
                        "[4D][FALLBACK] inst %s: no valid correspondences, attempting centroid-init ICP",
                        inst_id,
                    )
                    icp_result = _run_centroid_initialized_icp(
                        "FALLBACK",
                        self.config.icp_max_iterations,
                    )
                    if icp_result is None:
                        return "failed"
                    r_mat, t_vec = icp_result
                    use_fgr = False
                else:
                    return "failed"
            else:
                all_src = torch.cat(all_src, dim=0)
                all_tgt = torch.cat(all_tgt, dim=0)
                if len(all_src) < 3:
                    if skip_min_keypoints and has_dense:
                        logger.info(
                            "[4D][FALLBACK] inst %s: only %d correspondences, attempting centroid-init ICP",
                            inst_id,
                            len(all_src),
                        )
                        icp_result = _run_centroid_initialized_icp(
                            "FALLBACK",
                            self.config.icp_max_iterations,
                        )
                        if icp_result is None:
                            return "failed"
                        r_mat, t_vec = icp_result
                        use_fgr = False
                    else:
                        return "failed"
                elif len(all_src) < self.config.min_observed_keypoints:
                    if skip_min_keypoints and has_dense:
                        logger.info(
                            "[4D][FALLBACK] inst %s: %d correspondences below threshold %d, attempting centroid-init ICP",
                            inst_id,
                            len(all_src),
                            self.config.min_observed_keypoints,
                        )
                        icp_result = _run_centroid_initialized_icp(
                            "FALLBACK",
                            self.config.icp_max_iterations,
                        )
                        if icp_result is None:
                            return "failed"
                        r_mat, t_vec = icp_result
                        use_fgr = False
                    else:
                        return "failed"

            if use_fgr:
                src_np = all_src.detach().cpu().numpy()
                tgt_np = all_tgt.detach().cpu().numpy()
                src_extent = src_np.max(axis=0) - src_np.min(axis=0)
                tgt_extent = tgt_np.max(axis=0) - tgt_np.min(axis=0)
                if src_extent.max() < 1e-6 or tgt_extent.max() < 1e-6:
                    return "failed"

                try:
                    r_fgr, t_fgr = solve_rigid_transform(
                        src_np,
                        tgt_np,
                        threshold=self.config.fgr_threshold,
                    )
                except Exception as exc:
                    logger.warning("[4D] FGR failed for instance %s: %s", inst_id, exc)
                    return "failed"

                t_init = np.eye(4, dtype=np.float64)
                t_init[:3, :3] = np.asarray(r_fgr, dtype=np.float64)
                t_init[:3, 3] = np.asarray(t_fgr, dtype=np.float64)
                t_init = t_init @ self._pts_to_anchor.get(
                    inst_id,
                    np.eye(4, dtype=np.float64),
                )
                r_mat = t_init[:3, :3]
                t_vec = t_init[:3, 3]

                if has_dense:
                    try:
                        r_mat, t_vec = refine_registration_icp(
                            current_pts_np,
                            dense_obs,
                            r_mat,
                            t_vec,
                            threshold=self.config.icp_threshold,
                            max_iterations=self.config.icp_max_iterations,
                        )
                    except Exception as exc:
                        logger.warning("[4D] ICP failed for instance %s: %s", inst_id, exc)
                        return "failed"

        t_reg = np.eye(4, dtype=np.float64)
        t_reg[:3, :3] = np.asarray(r_mat, dtype=np.float64)
        t_reg[:3, 3] = np.asarray(t_vec, dtype=np.float64)

        center_tgt_pred = r_mat @ center_src + t_vec
        trans_mag = np.linalg.norm(center_tgt_pred - center_src)
        if trans_mag < self.config.stationary_threshold:
            self._last_observed_frames[inst_id] = curr_t
            self._cumulative_transforms[inst_id] = self._committed_cumulative.get(
                inst_id,
                np.eye(4, dtype=np.float64),
            ).copy()
            self._pts_to_anchor[inst_id] = np.eye(4, dtype=np.float64)
            return "stationary"

        self._last_observed_frames[inst_id] = curr_t
        self._cumulative_transforms[inst_id] = t_reg @ self._committed_cumulative.get(
            inst_id,
            np.eye(4, dtype=np.float64),
        )
        self._committed_cumulative[inst_id] = self._cumulative_transforms[
            inst_id
        ].copy()
        self._pts_to_anchor[inst_id] = np.eye(4, dtype=np.float64)

        new_pts = (current_pts_np @ r_mat.T + t_vec).astype(np.float32)
        self._current_points[y_idx] = new_pts
        self._instance_centroids[inst_id] = new_pts.mean(axis=0).astype(np.float64)
        return "changed"

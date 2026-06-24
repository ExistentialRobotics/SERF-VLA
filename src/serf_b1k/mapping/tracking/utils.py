"""Tracking-specific utilities for neural point tracking."""

import json
import os
import sys
from pathlib import Path
from typing import Literal, Optional

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F

from serf_b1k.mapping.tracking.config.tracking_config import TrackingConfig
from serf_b1k.mapping.utils.geometry import (
    refine_registration_icp,
    solve_rigid_transform,
    unproject_depth_to_world,
)
from serf_b1k.mapping.utils.visualization import apply_camera_extrinsic, compute_pca_colors

RegistrationOutcome = Literal["failed", "stationary", "changed"]

def _ensure_cotracker_on_path() -> None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "external" / "co-tracker"
        if (candidate / "cotracker").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.append(candidate_str)
            return


_ensure_cotracker_on_path()

from cotracker.models.core.model_utils import get_points_on_a_grid


def get_exported_neural_points_task_dir(run_dir: str) -> str:
    """
    Resolve the canonical exported neural point directory for a task run.

    Example:
        /tmp/points/models/task-0021 -> /tmp/points/exported_neural_points/task-0021
    """
    run_dir = run_dir.rstrip("/")
    task_name = os.path.basename(run_dir)
    root_dir = os.path.dirname(os.path.dirname(run_dir))
    return os.path.join(root_dir, "exported_neural_points", task_name)


# --- Keypoint Extraction --- #

def extract_keypoints_automatically(
    rgb_image: np.ndarray,
    mask: np.ndarray = None,
    max_points: int = 500,
    erosion_iters: int = 2,
    quality_level: float = 0.03,
    min_distance: int = 15,
    block_size: int = 7,
) -> list[tuple[float, float]]:
    """Extract keypoints using Shi-Tomasi corner detection.

    Args:
        rgb_image: (H, W, 3) uint8 RGB image.
        mask: Optional (H, W) uint8 mask (255 = valid region).
        max_points: Maximum number of corners to detect.
        erosion_iters: Erosion iterations for mask border removal.
        quality_level: Minimum corner quality (eigenvalue ratio).
        min_distance: Minimum pixel distance between keypoints.
        block_size: Corner detection block size.

    Returns:
        List of (u, v) keypoint coordinates.
    """
    if mask is not None and erosion_iters > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=erosion_iters)

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_points,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=block_size,
        mask=mask,
    )

    if corners is None:
        return []

    return [(float(corners[i, 0, 0]), float(corners[i, 0, 1])) for i in range(corners.shape[0])]


def sample_random_mask_points(
    mask: np.ndarray,
    num_points: int,
    erosion_iters: int = 2,
    min_distance: int = 0,
    existing_points: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """Sample random points from a binary mask with minimum distance constraint.

    Args:
        mask: (H, W) uint8 mask (255 = valid region).
        num_points: Maximum number of points to sample.
        erosion_iters: Erosion iterations for mask border removal.
        min_distance: Minimum pixel distance between sampled points (0 = no constraint).
        existing_points: Optional list of (u, v) points to maintain distance from
            (e.g. previously extracted keypoints). These are not included in output.

    Returns:
        List of (u, v) point coordinates.
    """
    if num_points <= 0:
        return []
    if mask is not None and erosion_iters > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=erosion_iters)
    ys, xs = np.where(mask == 255)
    if len(ys) == 0:
        return []

    if min_distance <= 0 and not existing_points:
        n = min(num_points, len(ys))
        idx = np.random.choice(len(ys), size=n, replace=False)
        return [(float(xs[i]), float(ys[i])) for i in idx]

    # Greedy sampling with minimum distance constraint
    candidates = np.stack([xs, ys], axis=1).astype(np.float32)
    perm = np.random.permutation(len(candidates))
    candidates = candidates[perm]

    # Seed with existing points so distance checks include them
    n_existing = len(existing_points) if existing_points else 0
    max_arr_size = n_existing + num_points
    selected_arr = np.empty((max_arr_size, 2), dtype=np.float32)
    for i, (u, v) in enumerate(existing_points or []):
        selected_arr[i] = [u, v]
    n_total = n_existing  # tracks entries in selected_arr

    selected: list[tuple[float, float]] = []
    min_dist_sq = float(min_distance * min_distance) if min_distance > 0 else 0.0

    for pt in candidates:
        if len(selected) >= num_points:
            break
        if n_total > 0 and min_dist_sq > 0:
            diffs = selected_arr[:n_total] - pt
            if (diffs * diffs).sum(axis=1).min() < min_dist_sq:
                continue
        selected_arr[n_total] = pt
        n_total += 1
        selected.append((float(pt[0]), float(pt[1])))

    return selected


# --- CoTracker3 Tracking --- #

def run_cotracker_offline(
    frames_np: np.ndarray,
    keypoints: list[tuple[float, float]],
    cotracker_model,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run CoTracker3 in offline mode, returning separate visibility and confidence.

    Bypasses the predictor wrapper to access raw model outputs (visibility and
    confidence as continuous [0, 1] scores) instead of the binarized combined
    score that the default predictor returns.

    Args:
        frames_np: (T, H, W, 3) uint8 RGB frames.
        keypoints: List of (u, v) on the first frame.
        cotracker_model: Loaded CoTrackerPredictor (offline) instance.
        device: Torch device string.

    Returns:
        tracks_xy: (T, N, 2) tracked positions (GPU tensor).
        vis: (T, N) visibility scores in [0, 1] (GPU tensor).
        conf: (T, N) confidence scores in [0, 1] (GPU tensor).
    """
    if len(keypoints) == 0:
        raise ValueError("No keypoints provided to run_cotracker_offline.")

    video = torch.from_numpy(frames_np).permute(0, 3, 1, 2)[None].float().to(device)
    B, T, C, H, W = video.shape

    # Access predictor internals
    model = cotracker_model.model
    interp_shape = cotracker_model.interp_shape
    support_grid_size = cotracker_model.support_grid_size

    # Build and scale queries to model resolution
    queries_list = [[0.0, float(u), float(v)] for (u, v) in keypoints]
    queries = torch.tensor([queries_list], dtype=torch.float32, device=device)
    N = queries.shape[1]

    scaled_queries = queries.clone()
    scaled_queries[:, :, 1:] *= scaled_queries.new_tensor([
        (interp_shape[1] - 1) / (W - 1),
        (interp_shape[0] - 1) / (H - 1),
    ])

    # Add support grid (same as predictor)
    grid_pts = get_points_on_a_grid(support_grid_size, interp_shape, device=device)
    grid_pts = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)
    full_queries = torch.cat([scaled_queries, grid_pts], dim=1)

    # Resize video to model resolution
    video_resized = video.reshape(B * T, C, H, W)
    video_resized = F.interpolate(video_resized, tuple(interp_shape), mode="bilinear", align_corners=True)
    video_resized = video_resized.reshape(B, T, 3, interp_shape[0], interp_shape[1])

    with torch.inference_mode():
        pred_tracks, pred_vis, pred_conf, _ = model(
            video=video_resized, queries=full_queries, iters=6,
        )

    # Strip support grid points
    pred_tracks = pred_tracks[:, :, :N]
    pred_vis = pred_vis[:, :, :N]
    pred_conf = pred_conf[:, :, :N]

    # Scale tracks back to original resolution
    pred_tracks = pred_tracks * pred_tracks.new_tensor([
        (W - 1) / (interp_shape[1] - 1),
        (H - 1) / (interp_shape[0] - 1),
    ])

    tracks_xy = pred_tracks[0].detach()  # (T, N, 2)

    vis = pred_vis[0].detach()
    if vis.dim() == 3:
        vis = vis[..., 0]
    # (T, N) continuous [0, 1]

    conf = pred_conf[0].detach()
    if conf.dim() == 3:
        conf = conf[..., 0]
    # (T, N) continuous [0, 1]

    return tracks_xy, vis, conf


def run_cotracker_online_multi(
    frames_np: np.ndarray,
    keypoints: list[tuple[float, float]],
    cotracker_model,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run CoTracker3 in online mode, returning separate visibility and confidence.

    Bypasses the predictor wrapper to access raw model outputs (visibility and
    confidence as continuous [0, 1] scores) instead of the binarized combined
    score that the default predictor returns.

    Args:
        frames_np: (T, H, W, 3) uint8 RGB frames.
        keypoints: List of (u, v) on the first frame.
        cotracker_model: Loaded CoTrackerOnlinePredictor instance.
        device: Torch device string.

    Returns:
        tracks_xy: (T, N, 2) tracked positions (GPU tensor).
        vis: (T, N) visibility scores in [0, 1] (GPU tensor).
        conf: (T, N) confidence scores in [0, 1] (GPU tensor).
    """
    if len(keypoints) == 0:
        raise ValueError("No keypoints provided to run_cotracker_online_multi.")

    video = torch.from_numpy(frames_np).permute(0, 3, 1, 2)[None].float().to(device)
    B, T, C, H, W = video.shape

    # Access predictor internals
    model = cotracker_model.model
    interp_shape = cotracker_model.interp_shape
    step = cotracker_model.step
    support_grid_size = cotracker_model.support_grid_size

    # Build and scale queries to model resolution
    queries_list = [[0.0, float(u), float(v)] for (u, v) in keypoints]
    queries = torch.tensor([queries_list], dtype=torch.float32, device=device)
    N = queries.shape[1]

    scaled_queries = queries.clone()
    scaled_queries[:, :, 1:] *= scaled_queries.new_tensor([
        (interp_shape[1] - 1) / (W - 1),
        (interp_shape[0] - 1) / (H - 1),
    ])

    # Add support grid (same as predictor)
    grid_pts = get_points_on_a_grid(support_grid_size, interp_shape, device=device)
    grid_pts = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)
    full_queries = torch.cat([scaled_queries, grid_pts], dim=1)

    # Resize helper
    def _resize(chunk: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = chunk.shape
        chunk = chunk.reshape(b * t, c, h, w)
        chunk = F.interpolate(chunk, tuple(interp_shape), mode="bilinear", align_corners=True)
        return chunk.reshape(b, t, 3, interp_shape[0], interp_shape[1])

    with torch.inference_mode():
        model.init_video_online_processing()

        if T <= 2 * step:
            resized = _resize(video)
            pred_tracks, pred_vis, pred_conf, _ = model(
                video=resized, queries=full_queries, iters=6, is_online=True,
            )
        else:
            pred_tracks = pred_vis = pred_conf = None
            for ind in range(0, T - step, step):
                chunk = video[:, ind : ind + 2 * step]
                resized = _resize(chunk)
                pred_tracks, pred_vis, pred_conf, _ = model(
                    video=resized, queries=full_queries, iters=6, is_online=True,
                )

    # Strip support grid points
    pred_tracks = pred_tracks[:, :, :N]
    pred_vis = pred_vis[:, :, :N]
    pred_conf = pred_conf[:, :, :N]

    # Scale tracks back to original resolution
    pred_tracks = pred_tracks * pred_tracks.new_tensor([
        (W - 1) / (interp_shape[1] - 1),
        (H - 1) / (interp_shape[0] - 1),
    ])

    tracks_xy = pred_tracks[0].detach()  # (T, N, 2)

    vis = pred_vis[0].detach()
    if vis.dim() == 3:
        vis = vis[..., 0]
    # (T, N) continuous [0, 1]

    conf = pred_conf[0].detach()
    if conf.dim() == 3:
        conf = conf[..., 0]
    # (T, N) continuous [0, 1]

    return tracks_xy, vis, conf


# --- Active Foreground Indices --- #

def get_active_foreground_indices(
    neural_points,
    excluded_train_ids_tensor: torch.Tensor,
) -> torch.Tensor:
    """Get active point indices excluding background/deleted points.

    Args:
        neural_points: NeuralPointMap instance.
        excluded_train_ids_tensor: Tensor of instance IDs to exclude.

    Returns:
        GPU tensor of active foreground point indices.
    """
    active_mask = (
        (neural_points.buffer_pt_index != neural_points.EMPTY_KEY)
        & (neural_points.buffer_pt_index != neural_points.DELETED_KEY)
    )
    active_indices = neural_points.buffer_pt_index[active_mask]
    vis_mask = ~torch.isin(
        neural_points.map_instance_ids[active_indices], excluded_train_ids_tensor
    )
    return active_indices[vis_mask]


# --- 2D Rendering --- #

def _render_single_view(
    view: str,
    t: int,
    chunk_results: dict,
    default_shape: tuple[int, int],
    visibility_threshold: float,
    confidence_threshold: float,
) -> np.ndarray:
    """Render a single view with keypoint overlay."""
    res = chunk_results.get(view, {})
    has_valid = res.get("valid", False)
    has_frames = "frames_np" in res

    if has_valid:
        img_view = res["frames_np"][t].copy()
        tracks = res["tracks_xy"]
        visib = res["visibilities"]
        colors = res["colors_2d"]
        conf = res.get("confidences")

        if "_tracks_np" not in res:
            res["_tracks_np"] = tracks.cpu().numpy() if isinstance(tracks, torch.Tensor) else tracks
            res["_vis_np"] = visib.cpu().numpy() if isinstance(visib, torch.Tensor) else visib
            if conf is not None:
                res["_conf_np"] = conf.cpu().numpy() if isinstance(conf, torch.Tensor) else conf
        tracks = res["_tracks_np"]
        visib = res["_vis_np"]
        conf_np = res.get("_conf_np")

        for n in range(tracks.shape[1]):
            vis_ok = visib[t, n] >= visibility_threshold
            conf_ok = conf_np[t, n] >= confidence_threshold if conf_np is not None else True
            if vis_ok and conf_ok:
                x, y = tracks[t, n]
                c = (int(colors[n, 0]), int(colors[n, 1]), int(colors[n, 2]))
                cv2.circle(img_view, (int(x), int(y)), 3, c, -1)
                start_trail = max(0, t - 20)
                for k in range(start_trail, t):
                    vk = visib[k, n] >= visibility_threshold and visib[k + 1, n] >= visibility_threshold
                    ck = True
                    if conf_np is not None:
                        ck = conf_np[k, n] >= confidence_threshold and conf_np[k + 1, n] >= confidence_threshold
                    if vk and ck:
                        pt1 = (int(tracks[k, n, 0]), int(tracks[k, n, 1]))
                        pt2 = (int(tracks[k + 1, n, 0]), int(tracks[k + 1, n, 1]))
                        cv2.line(img_view, pt1, pt2, c, 1)

        cv2.putText(img_view, view, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    elif has_frames:
        img_view = res["frames_np"][t].copy()
        cv2.putText(img_view, view, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    else:
        h, w = default_shape
        img_view = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(img_view, f"{view}: No Data", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    return img_view


def render_2d_frame(
    t: int,
    views: list[str],
    chunk_results: dict,
    default_shape: tuple[int, int] = (480, 480),
    visibility_threshold: float = 0.9,
    confidence_threshold: float = 0.85,
) -> np.ndarray:
    """Render a 1080x720 2D visualization: head on left, wrists stacked on right.

    Layout (when 3 views: head + 2 wrists):
        ┌────────────┬──────┐
        │            │wristL│  360x360
        │    head    ├──────┤
        │  720x720   │wristR│  360x360
        └────────────┴──────┘
              720       360    = 1080 x 720

    Falls back to horizontal concatenation with height alignment for other view counts.

    Args:
        t: Frame index within the current chunk.
        views: List of view names.
        chunk_results: Dict mapping view -> tracking results.
        default_shape: (H, W) fallback image shape when no frames are available.
        visibility_threshold: Minimum visibility score to draw a point.
        confidence_threshold: Minimum confidence score to draw a point.

    Returns:
        (720, 1080, 3) uint8 composite image.
    """
    imgs = {}
    for view in views:
        imgs[view] = _render_single_view(
            view, t, chunk_results, default_shape,
            visibility_threshold, confidence_threshold,
        )

    head_views = [v for v in views if "head" in v]
    wrist_views = [v for v in views if "wrist" in v]

    if len(head_views) == 1 and len(wrist_views) >= 1:
        head_img = cv2.resize(imgs[head_views[0]], (720, 720), interpolation=cv2.INTER_LINEAR)
        half_h = 720 // len(wrist_views)

        wrist_resized = []
        for wv in wrist_views:
            wrist_resized.append(cv2.resize(imgs[wv], (360, half_h), interpolation=cv2.INTER_LINEAR))
        right_col = np.concatenate(wrist_resized, axis=0)

        if right_col.shape[0] != 720:
            right_col = cv2.resize(right_col, (360, 720), interpolation=cv2.INTER_LINEAR)

        return np.concatenate([head_img, right_col], axis=1)

    # Fallback: horizontal concatenation with height alignment
    all_imgs = [imgs[v] for v in views]
    max_h = max(img.shape[0] for img in all_imgs)
    aligned = []
    for img in all_imgs:
        if img.shape[0] < max_h:
            img = cv2.resize(img, (img.shape[1], max_h), interpolation=cv2.INTER_LINEAR)
        aligned.append(img)
    return np.concatenate(aligned, axis=1)


# --- 3D Rendering --- #

def render_3d_frame(
    vis,
    pcd,
    neural_points,
    pca_colors_arr: np.ndarray,
    excluded_train_ids_tensor: torch.Tensor,
    camera_extrinsic,
    geometry_changed: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Update the Open3D visualizer and capture a rendered frame."""
    import open3d as o3d

    if geometry_changed:
        active_indices = get_active_foreground_indices(neural_points, excluded_train_ids_tensor)
        active_indices_np = active_indices.cpu().numpy()

        if len(pca_colors_arr) != neural_points.map_points.shape[0]:
            diff = neural_points.map_points.shape[0] - len(pca_colors_arr)
            if diff > 0:
                pad = np.full((diff, 3), 0.5, dtype=np.float32)
                pca_colors_arr = np.concatenate([pca_colors_arr, pad], axis=0)

        pcd.points = o3d.utility.Vector3dVector(neural_points.map_points[active_indices].cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(pca_colors_arr[active_indices_np])
        vis.update_geometry(pcd)

    vis.poll_events()
    vis.update_renderer()
    apply_camera_extrinsic(vis, camera_extrinsic)

    img = (np.asarray(vis.capture_screen_float_buffer(do_render=True)) * 255).astype(np.uint8)
    return img, pca_colors_arr


# --- Dense Observation Cloud --- #

def get_dense_observation_cloud(
    chunk_results: dict,
    t: int,
    inst_id: int,
    views: list[str],
    intrinsics: dict[str, tuple[float, float, float, float]],
) -> tuple[Optional[torch.Tensor], set[str]]:
    """Extract dense point cloud for ``inst_id`` at chunk-time ``t`` from all views.

    Args:
        chunk_results: Per-view tracking results dict.
        t: Frame index within the current chunk.
        inst_id: Instance ID to extract points for.
        views: List of view names.
        intrinsics: Per-view (fx, fy, cx, cy) tuples.

    Returns:
        (cloud, contributing_views): Tensor (N, 3) or None, and the set of
        view names that contributed points.
    """
    clouds = []
    dense_views: set[str] = set()
    for view in views:
        res = chunk_results.get(view)
        if res is None:
            continue
        if (
            "chunk_segs" not in res
            or "chunk_depths" not in res
            or "chunk_poses_c2w" not in res
        ):
            continue

        seg = res["chunk_segs"][t]
        depth = res["chunk_depths"][t]
        fx, fy, cx, cy = intrinsics[view]

        mask = seg == inst_id
        if not mask.any():
            continue

        H, W = depth.shape
        depth_t = depth.unsqueeze(0)
        pose_c2w = res["chunk_poses_c2w"][t].unsqueeze(0)

        coords_world = unproject_depth_to_world(
            depth_t, pose_c2w, fx, fy, cx, cy,
            original_height=H, original_width=W,
        )

        valid = mask & (depth > 0)
        p_world = coords_world[0, :, valid].T
        if p_world.shape[0] > 0:
            clouds.append(p_world)
            dense_views.add(view)

    if not clouds:
        return None, dense_views
    return torch.cat(clouds, dim=0), dense_views


# --- Per-Instance Registration --- #

def register_instance_for_frame(
    inst_id: int,
    anchor_t: int,
    t: int,
    views: list[str],
    chunk_results: dict,
    intrinsics: dict[str, tuple[float, float, float, float]],
    instance_particles: dict[int, torch.Tensor],
    neural_points,
    committed_cumulative: dict[int, np.ndarray],
    instance_centroids: dict[int, np.ndarray],
    cfg: TrackingConfig,
    device: str,
    is_fallback_retry: bool = False,
) -> RegistrationOutcome:
    """Register a single instance from its latest well-observed chunk frame.

    Collects 2D-to-3D correspondences from the most recent sufficiently
    observed frame in the current chunk to frame ``t``, solves a rigid transform via FGR,
    optionally refines with ICP against the dense observation cloud, and
    updates the neural point map.

    Returns:
        "changed" when the point map was modified, "stationary" when
        registration succeeded but motion was below threshold, otherwise
        "failed".
    """
    all_src, all_tgt = [], []
    contributing_views = set()

    for view in views:
        res = chunk_results.get(view, {})
        if not res.get("valid", False):
            continue

        kp_inst_ids = res["kp_inst_ids"]
        idx_inst = torch.where(kp_inst_ids == inst_id)[0]
        if len(idx_inst) == 0:
            continue

        vis_anchor = res["visibilities"][anchor_t]
        vis_curr = res["visibilities"][t]
        conf_anchor = res["confidences"][anchor_t]
        conf_curr = res["confidences"][t]
        vis_mask = (
            (vis_anchor[idx_inst] >= cfg.visibility_threshold)
            & (vis_curr[idx_inst] >= cfg.visibility_threshold)
            & (conf_anchor[idx_inst] >= cfg.confidence_threshold)
            & (conf_curr[idx_inst] >= cfg.confidence_threshold)
        )
        valid_idx = idx_inst[vis_mask]
        if len(valid_idx) == 0:
            continue

        depth_anchor = res["chunk_depths"][anchor_t]
        depth_curr = res["chunk_depths"][t]
        fx, fy, cx, cy = intrinsics[view]
        H, W = depth_anchor.shape

        depth_batch = torch.stack([depth_anchor, depth_curr])
        pose_c2w_batch = torch.stack([
            res["chunk_poses_c2w"][anchor_t],
            res["chunk_poses_c2w"][t],
        ])
        world_batch = unproject_depth_to_world(
            depth_batch, pose_c2w_batch, fx, fy, cx, cy,
            original_height=H, original_width=W,
        )

        uvs_anchor = res["tracks_xy"][anchor_t][valid_idx]
        uvs_curr = res["tracks_xy"][t][valid_idx]
        ui_a = torch.round(uvs_anchor[:, 0]).long()
        vi_a = torch.round(uvs_anchor[:, 1]).long()
        ui_c = torch.round(uvs_curr[:, 0]).long()
        vi_c = torch.round(uvs_curr[:, 1]).long()

        ui_a_s = torch.clamp(ui_a, 0, W - 1)
        vi_a_s = torch.clamp(vi_a, 0, H - 1)
        ui_c_s = torch.clamp(ui_c, 0, W - 1)
        vi_c_s = torch.clamp(vi_c, 0, H - 1)
        valid_a = (ui_a >= 0) & (ui_a < W) & (vi_a >= 0) & (vi_a < H) & (depth_anchor[vi_a_s, ui_a_s] > 0)
        valid_c = (ui_c >= 0) & (ui_c < W) & (vi_c >= 0) & (vi_c < H) & (depth_curr[vi_c_s, ui_c_s] > 0)
        both_valid = valid_a & valid_c
        if not both_valid.any():
            continue

        pts_anchor = world_batch[0, :, vi_a[both_valid], ui_a[both_valid]].T
        pts_curr = world_batch[1, :, vi_c[both_valid], ui_c[both_valid]].T
        all_src.append(pts_anchor)
        all_tgt.append(pts_curr)
        contributing_views.add(view)

    # --- Prepare particle state (needed for both paths) --- #
    Y_idx = instance_particles[inst_id]
    if len(Y_idx) == 0:
        return "failed"
    current_pts = neural_points.map_points[Y_idx]
    center_src = current_pts.detach().cpu().numpy().mean(axis=0)

    # --- Get dense observation cloud (needed for both ICP and rescue) --- #
    dense_obs, _ = get_dense_observation_cloud(chunk_results, t, inst_id, views, intrinsics)
    has_dense = dense_obs is not None and dense_obs.shape[0] > cfg.icp_min_points
    observed_centroid = dense_obs.mean(dim=0).cpu().numpy() if has_dense else None

    def _run_centroid_initialized_icp(label: str, max_iterations: int):
        init_R = np.eye(3, dtype=np.float64)
        init_t = (observed_centroid - center_src).astype(np.float64)
        R, t_vec, fitness, rmse = refine_registration_icp(
            current_pts.detach(), dense_obs,
            init_R, init_t,
            threshold=cfg.icp_threshold,
            max_iterations=max_iterations,
            return_metrics=True,
        )
        center_after = R @ center_src + t_vec
        residual = np.linalg.norm(center_after - observed_centroid)
        if residual > cfg.rescue_acceptance_threshold:
            print(
                f"[{label}] inst {inst_id}: REJECTED "
                f"(residual={residual:.4f}m, fitness={fitness:.3f}, rmse={rmse:.4f})"
            )
            return None
        print(
            f"[{label}] inst {inst_id}: ACCEPTED "
            f"(residual={residual:.4f}m, fitness={fitness:.3f}, rmse={rmse:.4f})"
        )
        return R, t_vec

    # --- Drift check: decide rescue vs normal --- #
    use_rescue = False
    if has_dense and inst_id in instance_centroids:
        stored_centroid = instance_centroids[inst_id]
        drift = np.linalg.norm(observed_centroid - stored_centroid)
        if drift > cfg.rescue_drift_threshold:
            use_rescue = True

    if use_rescue:
        # --- Rescue path: centroid-init ICP, skip FGR --- #
        print(
            f"[RESCUE] inst {inst_id}: drift={drift:.4f}m > "
            f"{cfg.rescue_drift_threshold}m, attempting rescue ICP"
        )
        icp_result = _run_centroid_initialized_icp("RESCUE", cfg.rescue_icp_max_iterations)
        if icp_result is None:
            return "failed"
        R, t_vec = icp_result
    else:
        # --- Normal path: existing gates + FGR + optional ICP --- #
        use_fgr = True
        if not all_src:
            if is_fallback_retry and has_dense:
                print(f"[FALLBACK] inst {inst_id}: no valid correspondences, attempting centroid-init ICP")
                icp_result = _run_centroid_initialized_icp("FALLBACK", cfg.icp_max_iterations)
                if icp_result is None:
                    return "failed"
                R, t_vec = icp_result
                use_fgr = False
            else:
                return "failed"
        else:
            all_src = torch.cat(all_src, dim=0)
            all_tgt = torch.cat(all_tgt, dim=0)
            if len(all_src) < 3:
                if is_fallback_retry and has_dense:
                    print(
                        f"[FALLBACK] inst {inst_id}: only {len(all_src)} "
                        "correspondences, attempting centroid-init ICP"
                    )
                    icp_result = _run_centroid_initialized_icp("FALLBACK", cfg.icp_max_iterations)
                    if icp_result is None:
                        return "failed"
                    R, t_vec = icp_result
                    use_fgr = False
                else:
                    return "failed"
            elif len(all_src) < cfg.min_observed_keypoints:
                if is_fallback_retry and has_dense:
                    print(
                        f"[FALLBACK] inst {inst_id}: {len(all_src)} correspondences "
                        f"below threshold {cfg.min_observed_keypoints}, attempting centroid-init ICP"
                    )
                    icp_result = _run_centroid_initialized_icp("FALLBACK", cfg.icp_max_iterations)
                    if icp_result is None:
                        return "failed"
                    R, t_vec = icp_result
                    use_fgr = False
                else:
                    return "failed"

        if use_fgr:
            # Guard against degenerate point clouds (zero extent crashes FGR)
            src_np = all_src.detach().cpu().numpy()
            tgt_np = all_tgt.detach().cpu().numpy()
            src_extent = src_np.max(axis=0) - src_np.min(axis=0)
            tgt_extent = tgt_np.max(axis=0) - tgt_np.min(axis=0)
            if src_extent.max() < 1e-6 or tgt_extent.max() < 1e-6:
                return "failed"

            # FGR registration
            R, t_vec = solve_rigid_transform(all_src, all_tgt, threshold=cfg.fgr_threshold)

            # ICP refinement with dense observation cloud
            if has_dense:
                R, t_vec = refine_registration_icp(
                    current_pts.detach(), dense_obs,
                    R, t_vec, threshold=cfg.icp_threshold,
                    max_iterations=cfg.icp_max_iterations,
                )

    # --- Common commit path (normal + rescue) --- #
    T_reg = np.eye(4, dtype=np.float64)
    T_reg[:3, :3] = R
    T_reg[:3, 3] = t_vec

    # Stationary check
    center_tgt_pred = R @ center_src + t_vec
    trans_mag = np.linalg.norm(center_tgt_pred - center_src)
    if trans_mag < cfg.stationary_threshold:
        return "stationary"

    committed_cumulative[inst_id] = T_reg @ committed_cumulative[inst_id]

    R_f = torch.tensor(R, dtype=torch.float32, device=device)
    t_f = torch.tensor(t_vec, dtype=torch.float32, device=device)
    new_pts = current_pts @ R_f.T + t_f

    neural_points.remove_points_from_hash(Y_idx)
    neural_points.map_points[Y_idx] = new_pts
    new_grid_coords = torch.floor(new_pts / neural_points.voxel_size).to(torch.int64)
    neural_points._hash_table_insert(new_grid_coords, Y_idx)

    # Update centroid atomically with commit
    instance_centroids[inst_id] = new_pts.detach().cpu().numpy().mean(axis=0)

    return "changed"


def count_visible_keypoints_per_instance(
    chunk_results: dict,
    views: list[str],
    instance_ids: list[int],
    anchor_t: int,
    target_t: int,
    visibility_threshold: float,
    confidence_threshold: float,
) -> dict[int, int]:
    """Count per-instance keypoints passing visibility+confidence at two chunk frames.

    Used to detect instances that need a fallback retry with more keypoints.

    Args:
        chunk_results: Per-view tracking results from CoTracker.
        views: List of view names.
        instance_ids: Instance IDs to count.
        anchor_t: Anchor frame index inside the chunk.
        target_t: Target/current frame index inside the chunk.
        visibility_threshold: Minimum visibility score.
        confidence_threshold: Minimum confidence score.

    Returns:
        Mapping from instance ID to count of valid keypoints across all views.
    """
    counts: dict[int, int] = {iid: 0 for iid in instance_ids}
    for view in views:
        res = chunk_results.get(view, {})
        if not res.get("valid", False):
            continue
        kp_inst_ids = res["kp_inst_ids"]
        vis = res["visibilities"]   # (T, N)
        conf = res["confidences"]   # (T, N)
        t_anchor = min(anchor_t, vis.shape[0] - 1)
        t_target = min(target_t, vis.shape[0] - 1)
        for inst_id in instance_ids:
            idx = torch.where(kp_inst_ids == inst_id)[0]
            if len(idx) == 0:
                continue
            valid = (
                (vis[t_anchor, idx] >= visibility_threshold)
                & (vis[t_target, idx] >= visibility_threshold)
                & (conf[t_anchor, idx] >= confidence_threshold)
                & (conf[t_target, idx] >= confidence_threshold)
            )
            counts[inst_id] += valid.sum().item()
    return counts


def count_registration_correspondences_per_instance(
    chunk_results: dict,
    views: list[str],
    instance_ids: list[int],
    anchor_t: int,
    target_t: int,
    visibility_threshold: float,
    confidence_threshold: float,
) -> dict[int, int]:
    """Count per-instance correspondences usable for anchor-to-target registration."""
    counts: dict[int, int] = {iid: 0 for iid in instance_ids}
    for view in views:
        res = chunk_results.get(view, {})
        if not res.get("valid", False):
            continue
        required_keys = ("kp_inst_ids", "visibilities", "confidences", "chunk_depths", "tracks_xy")
        if any(key not in res for key in required_keys):
            continue

        kp_inst_ids = res["kp_inst_ids"]
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
        depth_anchor = depths[anchor_t]
        depth_target = depths[target_t]
        h, w = depth_anchor.shape

        for inst_id in instance_ids:
            idx_inst = torch.where(kp_inst_ids == inst_id)[0]
            if len(idx_inst) == 0:
                continue
            valid_idx = idx_inst[
                (vis_anchor[idx_inst] >= visibility_threshold)
                & (vis_target[idx_inst] >= visibility_threshold)
                & (conf_anchor[idx_inst] >= confidence_threshold)
                & (conf_target[idx_inst] >= confidence_threshold)
            ]
            if len(valid_idx) == 0:
                continue

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
            counts[inst_id] += int((valid_anchor & valid_target).sum().item())
    return counts


def select_fallback_retry_instances_and_views(
    *,
    views: list[str],
    inst_ids_to_check: list[int],
    kp_counts: dict[int, int],
    min_observed_keypoints: int,
    anchor_segmentations: dict[str, np.ndarray | torch.Tensor],
    required_visibility_view_substrings: tuple[str, ...] | None = None,
) -> tuple[set[int], set[str]]:
    """Select fallback retry instances and views from anchor-frame segmentations."""
    seg_cache: dict[str, np.ndarray] = {}

    def get_seg(view: str) -> np.ndarray | None:
        seg = anchor_segmentations.get(view)
        if seg is None:
            return None
        if view not in seg_cache:
            if isinstance(seg, torch.Tensor):
                seg = seg.detach().cpu().numpy()
            seg_cache[view] = np.asarray(seg)
        return seg_cache[view]

    required_views: list[str] = []
    if required_visibility_view_substrings:
        required_views = [
            view for view in views
            if any(substr in view for substr in required_visibility_view_substrings)
        ]

    retry_instances: set[int] = set()
    for inst_id in inst_ids_to_check:
        if kp_counts.get(inst_id, 0) >= min_observed_keypoints:
            continue
        if required_views:
            visible_in_required_view = False
            for view in required_views:
                seg = get_seg(view)
                if seg is not None and np.any(seg == inst_id):
                    visible_in_required_view = True
                    break
            if not visible_in_required_view:
                continue
        elif required_visibility_view_substrings:
            continue
        for view in views:
            seg = get_seg(view)
            if seg is not None and np.any(seg == inst_id):
                retry_instances.add(inst_id)
                break

    retry_views: set[str] = set()
    if retry_instances:
        for view in views:
            seg = get_seg(view)
            if seg is not None and any(np.any(seg == inst_id) for inst_id in retry_instances):
                retry_views.add(view)

    return retry_instances, retry_views


# --- Chunk Loading and Tracking --- #

def load_chunk_and_track(
    t_start: int,
    t_end: int,
    views: list[str],
    hdf5_file: h5py.File,
    instance_particles: dict[int, torch.Tensor],
    intrinsics: dict[str, tuple[float, float, float, float]],
    cotracker: torch.nn.Module,
    frame_step: int,
    device: str,
    erosion_iters: int = 2,
    kp_quality_level: float = 0.03,
    kp_min_distance: int = 15,
    kp_block_size: int = 7,
    max_total_points_per_instance: int = 50,
    visibility_threshold: float = 0.9,
    confidence_threshold: float = 0.85,
    min_observed_keypoints: int = 15,
    moved_instance_ids: set[int] | None = None,
) -> tuple[dict, bool, set[int]]:
    """Load a time-chunk from HDF5, extract keypoints, and run CoTracker.

    Extracts up to ``max_total_points_per_instance`` Shi-Tomasi keypoints per
    instance. If fewer are found, fills the remaining budget with random mask
    points that respect ``random_min_distance`` from existing keypoints.

    After the initial tracking pass, instances with fewer than
    ``min_observed_keypoints`` visible keypoints are retried with doubled point
    budget and halved minimum distance to improve tracking coverage.

    Args:
        t_start: Start frame index (inclusive).
        t_end: End frame index (exclusive).
        views: List of view names.
        hdf5_file: Open HDF5 file handle.
        instance_particles: Mapping from instance ID to tracked point indices.
        intrinsics: Per-view (fx, fy, cx, cy) tuples.
        cotracker: Loaded CoTracker model.
        frame_step: Frame subsampling step.
        device: Torch device string.
        erosion_iters: Erosion iterations for mask border removal.
        kp_quality_level: Shi-Tomasi minimum corner quality.
        kp_min_distance: Shi-Tomasi minimum pixel distance between keypoints.
        kp_block_size: Shi-Tomasi corner detection block size.
        max_total_points_per_instance: Total point budget per instance.
        visibility_threshold: Minimum visibility score for fallback check.
        confidence_threshold: Minimum confidence score for fallback check.
        min_observed_keypoints: Minimum valid keypoints per instance before
            triggering a fallback retry.
        moved_instance_ids: Set of instance IDs that have previously exceeded
            stationary_threshold. Only these are eligible for fallback retry.

    Returns:
        chunk_results: Per-view tracking results dict.
        any_valid_keypoints: Whether any view had valid keypoints.
        retry_instances: Set of instance IDs that were retried with fallback.
    """
    chunk_results = {}
    any_valid_keypoints = False
    retry_instances: set[int] = set()
    chunk_len = len(range(t_start, t_end, frame_step))
    anchor_t = max(0, chunk_len - 2)
    target_t = max(0, chunk_len - 1)

    for view in views:
        grp = hdf5_file[view]
        frames_np = grp["rgb"][t_start:t_end:frame_step]
        chunk_depths_np = grp["depth"][t_start:t_end:frame_step].astype(np.float32) / 1000.0
        chunk_poses_np = grp["poses"][t_start:t_end:frame_step].astype(np.float32)
        chunk_segs_np = grp["seg_instance_id"][t_start:t_end:frame_step]

        # Batch convert to GPU tensors (single transfer per chunk)
        chunk_depths = torch.from_numpy(chunk_depths_np).to(device)
        chunk_poses = torch.from_numpy(chunk_poses_np).to(device)
        chunk_poses_c2w = torch.linalg.inv(chunk_poses)[:, :3, :4]
        chunk_segs = torch.from_numpy(chunk_segs_np.astype(np.int64)).to(device)

        seg_first = chunk_segs_np[0]
        view_keypoints = []
        view_kp_inst_ids = []

        for inst_id in instance_particles:
            mask = np.zeros_like(seg_first, dtype=np.uint8)
            mask[seg_first == inst_id] = 255
            if np.sum(mask) == 0:
                continue
            kps = extract_keypoints_automatically(
                frames_np[0], mask, max_points=max_total_points_per_instance,
                erosion_iters=erosion_iters, quality_level=kp_quality_level,
                min_distance=kp_min_distance, block_size=kp_block_size,
            )
            remaining = max_total_points_per_instance - len(kps)
            if remaining > 0:
                rand_kps = sample_random_mask_points(
                    mask, remaining,
                    erosion_iters=erosion_iters,
                    min_distance=kp_min_distance,
                    existing_points=kps,
                )
                kps = kps + rand_kps
            view_keypoints.extend(kps)
            view_kp_inst_ids.extend([inst_id] * len(kps))

        if not view_keypoints:
            chunk_results[view] = {
                "valid": False,
                "frames_np": frames_np,
                "chunk_depths": chunk_depths,
                "chunk_poses": chunk_poses,
                "chunk_poses_c2w": chunk_poses_c2w,
                "chunk_segs": chunk_segs,
            }
            continue

        any_valid_keypoints = True
        tracks_xy, visibilities, confidences = run_cotracker_offline(frames_np, view_keypoints, cotracker, device=device)
        chunk_results[view] = {
            "valid": True,
            "frames_np": frames_np,
            "chunk_depths": chunk_depths,
            "chunk_poses": chunk_poses,
            "chunk_poses_c2w": chunk_poses_c2w,
            "chunk_segs": chunk_segs,
            "keypoints": view_keypoints,
            "kp_inst_ids": torch.tensor(view_kp_inst_ids, dtype=torch.long, device=device),
            "tracks_xy": tracks_xy,
            "visibilities": visibilities,
            "confidences": confidences,
            "colors_2d": np.random.randint(0, 255, (len(view_keypoints), 3), dtype=np.uint8),
        }

    # --- Fallback: retry views with under-tracked instances using 2x budget --- #
    # Only retry instances that have previously exceeded stationary_threshold.
    _moved = moved_instance_ids or set()
    if any_valid_keypoints and _moved:
        inst_ids_to_check = [iid for iid in instance_particles if iid in _moved]
        kp_counts = count_registration_correspondences_per_instance(
            chunk_results, views, inst_ids_to_check,
            anchor_t, target_t,
            visibility_threshold, confidence_threshold,
        )
        seg_anchor_cache: dict[str, np.ndarray] = {}
        for inst_id in inst_ids_to_check:
            if kp_counts[inst_id] >= min_observed_keypoints:
                continue
            visible = False
            for view in views:
                if view not in seg_anchor_cache:
                    seg_anchor_cache[view] = chunk_results[view]["chunk_segs"][anchor_t].detach().cpu().numpy()
                if np.any(seg_anchor_cache[view] == inst_id):
                    visible = True
                    break
            if visible:
                retry_instances.add(inst_id)

        if retry_instances:
            doubled_budget = max_total_points_per_instance * 2
            halved_min_distance = max(1, kp_min_distance // 2)
            print(f"[FALLBACK] Retrying {len(retry_instances)} under-tracked instances "
                  f"with {doubled_budget} points (min_distance={halved_min_distance}): {sorted(retry_instances)}")

            retry_views: set[str] = set()
            for view in views:
                if view not in seg_anchor_cache:
                    seg_anchor_cache[view] = chunk_results[view]["chunk_segs"][anchor_t].detach().cpu().numpy()
                seg = seg_anchor_cache[view]
                for inst_id in retry_instances:
                    if np.any(seg == inst_id):
                        retry_views.add(view)
                        break

            for view in retry_views:
                res = chunk_results[view]
                frames_np = res["frames_np"]
                seg_first = res["chunk_segs"][0].detach().cpu().numpy()

                view_keypoints = []
                view_kp_inst_ids = []

                for inst_id in instance_particles:
                    mask = np.zeros_like(seg_first, dtype=np.uint8)
                    mask[seg_first == inst_id] = 255
                    if np.sum(mask) == 0:
                        continue
                    is_retry = inst_id in retry_instances
                    budget = doubled_budget if is_retry else max_total_points_per_instance
                    min_dist = halved_min_distance if is_retry else kp_min_distance
                    kps = extract_keypoints_automatically(
                        frames_np[0], mask, max_points=budget,
                        erosion_iters=erosion_iters, quality_level=kp_quality_level,
                        min_distance=min_dist, block_size=kp_block_size,
                    )
                    remaining = budget - len(kps)
                    if remaining > 0:
                        rand_kps = sample_random_mask_points(
                            mask, remaining,
                            erosion_iters=erosion_iters,
                            min_distance=min_dist,
                            existing_points=kps,
                        )
                        kps = kps + rand_kps
                    view_keypoints.extend(kps)
                    view_kp_inst_ids.extend([inst_id] * len(kps))

                if view_keypoints:
                    tracks_xy, visibilities, confidences = run_cotracker_offline(
                        frames_np, view_keypoints, cotracker, device=device,
                    )
                    chunk_results[view] = {
                        **{k: v for k, v in res.items()
                           if k not in ("valid", "keypoints", "kp_inst_ids",
                                        "tracks_xy", "visibilities", "confidences",
                                        "colors_2d", "_tracks_np", "_vis_np", "_conf_np")},
                        "valid": True,
                        "keypoints": view_keypoints,
                        "kp_inst_ids": torch.tensor(view_kp_inst_ids, dtype=torch.long, device=device),
                        "tracks_xy": tracks_xy,
                        "visibilities": visibilities,
                        "confidences": confidences,
                        "colors_2d": np.random.randint(0, 255, (len(view_keypoints), 3), dtype=np.uint8),
                    }

    return chunk_results, any_valid_keypoints, retry_instances


# --- HDF5 Export --- #

def write_neural_points_to_hdf5(
    hf: h5py.File,
    points: np.ndarray,
    features: np.ndarray,
    instance_ids: np.ndarray,
    instance_id_map: dict,
    exclude_ids: set[int] | None = None,
) -> None:
    """Write base neural point data to an open HDF5 file.

    Filters out excluded instance IDs before writing (if provided).
    Both ``export_neural_points.py`` and ``neural_point_tracking.py`` call
    this to ensure an identical base format.

    Args:
        hf: Open HDF5 file handle (write mode).
        points: (N, 3) float32 point positions.
        features: (N, D) float32 per-point features.
        instance_ids: (N,) int64 per-point instance IDs.
        instance_id_map: Instance ID to name mapping.
        exclude_ids: Optional set of instance IDs to filter out.
    """
    if exclude_ids:
        mask = ~np.isin(instance_ids, list(exclude_ids))
        points = points[mask]
        features = features[mask]
        instance_ids = instance_ids[mask]

    unique_ids = sorted(set(instance_ids.tolist()))

    hf.create_dataset("initial_points", data=points)
    hf.create_dataset("initial_features", data=features)
    hf.create_dataset("initial_instance_ids", data=instance_ids)
    hf.attrs["num_instances"] = len(unique_ids)
    hf.attrs["instance_ids"] = json.dumps(unique_ids)
    hf.attrs["instance_id_to_name"] = json.dumps(
        {str(k): v for k, v in instance_id_map.items()}
    )



# --- Export Tracking Results --- #

def export_tracking_results(
    cfg: TrackingConfig,
    episode_name: str,
    initial_points: np.ndarray,
    initial_features: np.ndarray,
    initial_instance_ids: np.ndarray,
    frame_indices: list[int],
    frame_transforms: dict[int, list[np.ndarray]],
    training_id_map: dict[str, str],
    exclude_ids: set[int],
) -> None:
    """Save tracking results (transforms, initial state) to HDF5."""
    output_dir = os.path.join(get_exported_neural_points_task_dir(cfg.run_dir), "train")
    os.makedirs(output_dir, exist_ok=True)
    output_hdf5_path = os.path.join(output_dir, f"{episode_name}.hdf5")

    with h5py.File(output_hdf5_path, "w") as hf:
        # Base data (identical to export_neural_points.py)
        write_neural_points_to_hdf5(
            hf, initial_points, initial_features, initial_instance_ids,
            training_id_map, exclude_ids=exclude_ids,
        )
        # Tracking-specific data
        hf.create_dataset("frame_indices", data=np.array(frame_indices, dtype=np.int64))
        grp = hf.create_group("transforms")
        for iid, tf_list in frame_transforms.items():
            grp.create_dataset(str(iid), data=np.stack(tf_list).astype(np.float32))
        hf.attrs["num_frames"] = len(frame_indices)

    print(f"[DONE] Tracking results saved to {output_hdf5_path}")

from __future__ import annotations

from typing import Optional

import numpy as np
import open3d as o3d

from .point_normalization import get_pc_norm_params, normalize_pc, normalize_rgb
from .point_sampling import equal_per_instance_sampling, sample_with_instance_filter


def sample_point_payload(
    *,
    total_n: int,
    instance_ids: np.ndarray,
    xyz: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    feat: Optional[np.ndarray] = None,
    id_to_name: Optional[dict[str, str]] = None,
    keep_all_categories: tuple[str, ...] | None = None,
    budget_categories: tuple[str, ...] = (),
) -> dict[str, np.ndarray | None]:
    data: dict[str, np.ndarray | None] = {"xyz": xyz}
    if rgb is not None:
        data["rgb"] = rgb
    if feat is not None:
        data["feat"] = feat

    if id_to_name is not None and keep_all_categories:
        sampled = sample_with_instance_filter(
            total_n=total_n,
            instance_ids=instance_ids,
            data=data,
            id_to_name=id_to_name,
            keep_all_categories=keep_all_categories,
            budget_categories=budget_categories,
        )
    else:
        sampled = equal_per_instance_sampling(
            total_n=total_n,
            instance_ids=instance_ids,
            data=data,
        )
        sampled.pop("_instance_ids", None)
    _assert_sample_count(sampled, total_n)
    return sampled


def normalize_sampled_point_payload(
    sampled: dict[str, np.ndarray | None],
    task_idx: str,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    xyz_o3d = o3d.core.Tensor(sampled["xyz"], dtype=o3d.core.float32)
    xyz_norm = normalize_pc(xyz_o3d, task_idx).numpy().astype(np.float32)

    rgb_norm: Optional[np.ndarray] = None
    if sampled.get("rgb") is not None:
        rgb_norm = normalize_rgb(sampled["rgb"]).astype(np.float32)

    feat_out: Optional[np.ndarray] = sampled.get("feat")
    if feat_out is not None:
        feat_out = feat_out.astype(np.float32)

    return xyz_norm, feat_out, rgb_norm


def sample_robot_map_payload(
    *,
    total_n: int,
    env_instance_ids: np.ndarray,
    env_xyz: np.ndarray,
    env_feat: Optional[np.ndarray],
    env_rgb: Optional[np.ndarray] = None,
    robot_xyz: Optional[np.ndarray] = None,
    robot_feat: Optional[np.ndarray] = None,
    id_to_name: Optional[dict[str, str]] = None,
    keep_all_categories: tuple[str, ...] | None = None,
    budget_categories: tuple[str, ...] = (),
) -> tuple[dict[str, np.ndarray | None], np.ndarray]:
    env_ids = env_instance_ids.reshape(-1)
    if robot_xyz is None:
        sampled = sample_point_payload(
            total_n=total_n,
            instance_ids=env_ids,
            xyz=env_xyz,
            rgb=env_rgb,
            feat=env_feat,
            id_to_name=id_to_name,
            keep_all_categories=keep_all_categories,
            budget_categories=budget_categories,
        )
        return sampled, np.zeros(total_n, dtype=np.bool_)

    if robot_feat is None:
        raise ValueError("robot_feat is required when robot_xyz is provided.")
    if robot_xyz.shape[0] != robot_feat.shape[0]:
        raise ValueError(
            f"robot_xyz count ({robot_xyz.shape[0]}) != robot_feat count ({robot_feat.shape[0]})."
        )

    if id_to_name is not None and keep_all_categories:
        robot_count = robot_xyz.shape[0]
        env_budget = max(0, total_n - robot_count)
        env_sampled = sample_point_payload(
            total_n=env_budget,
            instance_ids=env_ids,
            xyz=env_xyz,
            rgb=env_rgb,
            feat=env_feat,
            id_to_name=id_to_name,
            keep_all_categories=keep_all_categories,
            budget_categories=budget_categories,
        )
        sampled_xyz = np.concatenate([env_sampled["xyz"], robot_xyz], axis=0)
        sampled_feat = None
        if env_sampled.get("feat") is not None:
            sampled_feat = np.concatenate([env_sampled["feat"], robot_feat], axis=0)
        sampled = {"xyz": sampled_xyz, "feat": sampled_feat, "rgb": None}
        robot_mask = np.concatenate(
            [
                np.zeros(env_sampled["xyz"].shape[0], dtype=np.bool_),
                np.ones(robot_count, dtype=np.bool_),
            ]
        )
    else:
        robot_id = int(env_ids.max()) + 1 if env_ids.size > 0 else 0
        robot_ids = np.full(robot_xyz.shape[0], robot_id, dtype=env_ids.dtype)
        all_xyz = np.concatenate([env_xyz, robot_xyz], axis=0)
        all_ids = np.concatenate([env_ids, robot_ids], axis=0)
        all_feat = None
        if env_feat is not None:
            all_feat = np.concatenate([env_feat, robot_feat], axis=0)
        sampled = equal_per_instance_sampling(
            total_n=total_n,
            instance_ids=all_ids,
            data={"xyz": all_xyz, "feat": all_feat},
        )
        robot_mask = sampled.pop("_instance_ids") == robot_id

    _assert_sample_count(sampled, total_n)
    if robot_mask.shape[0] != total_n:
        raise ValueError(f"robot_mask has {robot_mask.shape[0]} points, expected {total_n}.")
    return sampled, robot_mask


def attach_pc_norm_stats(item: dict, task_idx: str) -> None:
    offset, scale = get_pc_norm_params(task_idx)
    item["observation.points.pc_norm_offset"] = offset
    item["observation.points.pc_norm_scale"] = np.array([scale], dtype=np.float32)


def _assert_sample_count(sampled: dict[str, np.ndarray | None], total_n: int) -> None:
    actual = sampled["xyz"].shape[0]
    if actual != total_n:
        raise ValueError(f"Sampled point count {actual} does not match expected {total_n}.")

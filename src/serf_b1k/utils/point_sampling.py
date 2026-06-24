from typing import Dict, Optional, Tuple

import numpy as np
import open3d as o3d

from serf_b1k.mapping.utils.category_utils import extract_category_from_name


def random_sampling(t_pcd, total_n):
    t_pcd_len = len(t_pcd.point.positions)
    indices = np.random.choice(t_pcd_len, total_n, replace=False)
    indices = o3d.core.Tensor(indices, dtype=o3d.core.int64)
    return t_pcd.select_by_index(indices)


def equal_per_instance_sampling(
    total_n: int,
    instance_ids: np.ndarray,
    data: Dict[str, Optional[np.ndarray]],
    instance_weights: Optional[Dict[int, float]] = None,
) -> Dict[str, Optional[np.ndarray]]:
    if instance_ids is None:
        raise ValueError("instance_ids is required and cannot be None.")

    instance_ids = instance_ids.reshape(-1)
    n_points = instance_ids.shape[0]
    if n_points == 0:
        raise ValueError("instance_ids is empty.")

    unique_ids = np.unique(instance_ids)
    if len(unique_ids) == 0:
        raise ValueError("No valid instance IDs found.")

    for key, value in data.items():
        if value is None:
            continue
        if not isinstance(value, np.ndarray):
            raise TypeError(f"data['{key}'] must be np.ndarray or None, got {type(value)}")
        if value.shape[0] != n_points:
            raise ValueError(
                f"Length mismatch for key '{key}': data['{key}'].shape[0]={value.shape[0]} "
                f"but instance_ids.shape[0]={n_points}"
            )

    if instance_weights is not None:
        weights = np.array([instance_weights.get(int(uid), 1.0) for uid in unique_ids])
    else:
        weights = np.ones(len(unique_ids))

    raw_alloc = weights / weights.sum() * total_n
    alloc = np.floor(raw_alloc).astype(int)
    rem = total_n - alloc.sum()
    if rem > 0:
        alloc[np.argsort(-(raw_alloc - alloc))[:rem]] += 1

    picked = []
    for index, inst_id in enumerate(unique_ids):
        candidates = np.where(instance_ids == inst_id)[0]
        cur_n = alloc[index]
        replace = len(candidates) < cur_n
        picked.append(np.random.choice(candidates, cur_n, replace=replace))

    picked = np.concatenate(picked)
    np.random.shuffle(picked)

    sampled_data: Dict[str, Optional[np.ndarray]] = {}
    for key, value in data.items():
        sampled_data[key] = None if value is None else value[picked]
    sampled_data["_instance_ids"] = instance_ids[picked]
    return sampled_data


def sample_with_instance_filter(
    total_n: int,
    instance_ids: np.ndarray,
    data: Dict[str, Optional[np.ndarray]],
    id_to_name: Dict[str, str],
    keep_all_categories: Tuple[str, ...],
    budget_categories: Tuple[str, ...],
) -> Dict[str, Optional[np.ndarray]]:
    instance_ids = instance_ids.reshape(-1)
    keep_all_set = set(keep_all_categories)
    budget_set = set(budget_categories)
    keep_all_ids: set[int] = set()
    budget_ids: set[int] = set()

    for id_str, name in id_to_name.items():
        cat = extract_category_from_name(name)
        inst_id = int(id_str)
        if cat in keep_all_set:
            keep_all_ids.add(inst_id)
        elif cat in budget_set:
            budget_ids.add(inst_id)

    keep_mask = (
        np.isin(instance_ids, list(keep_all_ids))
        if keep_all_ids
        else np.zeros(len(instance_ids), dtype=bool)
    )
    keep_indices = np.where(keep_mask)[0]
    n_keep = len(keep_indices)
    n_budget = max(0, total_n - n_keep)

    if n_budget > 0 and budget_ids:
        budget_mask = np.isin(instance_ids, list(budget_ids))
        budget_indices = np.where(budget_mask)[0]
        if len(budget_indices) > 0:
            budget_sub_ids = instance_ids[budget_indices]
            budget_sub_data = {
                key: value[budget_indices] if value is not None else None
                for key, value in data.items()
            }
            sampled_budget = equal_per_instance_sampling(
                total_n=n_budget,
                instance_ids=budget_sub_ids,
                data=budget_sub_data,
            )

            result: Dict[str, Optional[np.ndarray]] = {}
            for key, value in data.items():
                result[key] = (
                    None
                    if value is None
                    else np.concatenate([value[keep_indices], sampled_budget[key]], axis=0)
                )

            sample_len = result[next(key for key, value in result.items() if value is not None)].shape[0]
            perm = np.random.permutation(sample_len)
            for key, value in result.items():
                if value is not None:
                    result[key] = value[perm]
            return result

    if n_keep == 0:
        return equal_per_instance_sampling(total_n, instance_ids, data)

    if n_keep > total_n:
        chosen = np.random.choice(keep_indices, total_n, replace=False)
    elif n_keep < total_n:
        extra = np.random.choice(keep_indices, total_n - n_keep, replace=True)
        chosen = np.concatenate([keep_indices, extra])
    else:
        chosen = keep_indices

    np.random.shuffle(chosen)
    return {key: None if value is None else value[chosen] for key, value in data.items()}

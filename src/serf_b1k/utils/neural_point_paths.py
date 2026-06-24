from __future__ import annotations

from pathlib import Path


_LEGACY_NEURAL_POINT_ROOTS = {
    "tracking_results",
    "exported_points",
}
_CANONICAL_NEURAL_POINT_ROOT = "exported_neural_points"
_CANONICAL_MAP_DATASET_ROOT = "SERF-BEHAVIOR-1K-MAP"


def canonicalize_neural_point_root(map_dataset_root_path: str | Path) -> str:
    root = Path(map_dataset_root_path)
    if root.name in _LEGACY_NEURAL_POINT_ROOTS:
        root = root.parent / _CANONICAL_NEURAL_POINT_ROOT
    elif root.name == _CANONICAL_MAP_DATASET_ROOT:
        root = root / _CANONICAL_NEURAL_POINT_ROOT
    elif root.name != _CANONICAL_NEURAL_POINT_ROOT and (
        root / _CANONICAL_NEURAL_POINT_ROOT
    ).exists():
        root = root / _CANONICAL_NEURAL_POINT_ROOT
    return str(root)


def resolve_eval_scene_hdf5_path(
    map_dataset_root_path: str | Path,
    task_idx_str: str,
    instance_id: int,
) -> str:
    root = Path(canonicalize_neural_point_root(map_dataset_root_path))
    return str(root / task_idx_str / "eval" / f"scene_{int(instance_id)}.hdf5")

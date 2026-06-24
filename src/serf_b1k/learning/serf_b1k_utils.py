from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


def _find_repo_src() -> Optional[Path]:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        candidate = parent / "src"
        if (candidate / "serf_b1k" / "utils" / "__init__.py").exists():
            return candidate
    return None


_repo_src = _find_repo_src()
if _repo_src is not None and str(_repo_src) not in sys.path:
    sys.path.insert(0, str(_repo_src))

from serf_b1k.utils import (  # noqa: E402
    EvalPolicyBoundaryState,
    EvalMapSettings,
    canonicalize_neural_point_root,
    collect_initial_predicate_states,
    compute_step_q_score,
    decode_latent_features,
    equal_per_instance_sampling,
    ensure_repo_src_on_path,
    get_task_point_category_settings,
    get_pc_norm_params,
    jsonable_step_q_score,
    load_mlp_checkpoint,
    normalize_pc,
    normalize_rgb,
    normalize_sampled_point_payload,
    point_cloud_stats,
    random_sampling,
    read_initial_point_payload,
    read_instance_id_to_name,
    read_optional_dataset,
    resolve_eval_map_config,
    resolve_eval_map_settings,
    resolve_eval_scene_hdf5_path,
    sample_point_payload,
    sample_robot_map_payload,
    sample_with_instance_filter,
    should_record_step_q_score,
    task_name_to_task_id,
)

__all__ = [
    "canonicalize_neural_point_root",
    "collect_initial_predicate_states",
    "compute_step_q_score",
    "EvalPolicyBoundaryState",
    "EvalMapSettings",
    "decode_latent_features",
    "equal_per_instance_sampling",
    "random_sampling",
    "sample_with_instance_filter",
    "get_task_point_category_settings",
    "get_pc_norm_params",
    "jsonable_step_q_score",
    "load_mlp_checkpoint",
    "normalize_pc",
    "normalize_rgb",
    "normalize_sampled_point_payload",
    "point_cloud_stats",
    "ensure_repo_src_on_path",
    "read_initial_point_payload",
    "read_optional_dataset",
    "read_instance_id_to_name",
    "resolve_eval_map_config",
    "resolve_eval_map_settings",
    "resolve_eval_scene_hdf5_path",
    "sample_point_payload",
    "sample_robot_map_payload",
    "should_record_step_q_score",
    "task_name_to_task_id",
]

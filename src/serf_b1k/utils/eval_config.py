from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any

from serf_b1k.utils.task_metadata import (
    extract_task_id_from_config_name,
    get_task_point_category_settings,
    normalize_task_id,
    task_name_to_task_id,
)


@dataclasses.dataclass(frozen=True)
class EvalMapSettings:
    num_input_points: int
    map_input_type: str
    task_id: str
    instance_keep_all_categories: tuple[str, ...]
    instance_budget_categories: tuple[str, ...]
    robot_model_root_path: str | None = None


def resolve_eval_map_settings(cfg: Any) -> EvalMapSettings:
    """Resolve eval map settings from task-scoped metadata and optional preset."""
    task_name = _resolve_eval_task_name(cfg)
    eval_task_id = task_name_to_task_id(task_name)

    eval_map_input_type = _none_if_missing(getattr(cfg, "map_input_type", None))
    policy_config_name = _none_if_missing(getattr(cfg, "policy_config_name", None))
    explicit_num_points = _none_if_missing(getattr(cfg, "num_input_points", None))
    robot_model_root_path = _none_if_missing(
        getattr(cfg, "robot_model_root_path", None)
    )

    if policy_config_name is None:
        raise ValueError(
            "policy_config_name must be set for map evaluation so train/eval "
            "map settings stay in sync."
        )

    train_settings = _load_training_map_settings(str(policy_config_name))
    model_map_input_type = train_settings.map_input_type
    if eval_map_input_type is None:
        eval_map_input_type = model_map_input_type
    elif eval_map_input_type != model_map_input_type:
        raise ValueError(
            "Eval map_input_type does not match policy_config_name: "
            f"eval={eval_map_input_type}, "
            f"{policy_config_name}.model.map_input_type={model_map_input_type}"
        )

    if train_settings.task_id != eval_task_id:
        raise ValueError(
            "Eval cfg.task.name does not match policy_config_name task: "
            f"cfg.task.name={task_name!r} -> {eval_task_id}, "
            f"policy_config_name={policy_config_name!r} -> {train_settings.task_id}."
        )

    if explicit_num_points is not None and int(explicit_num_points) != int(
        train_settings.num_input_points
    ):
        raise ValueError(
            "Eval num_input_points does not match policy_config_name: "
            f"eval={int(explicit_num_points)}, "
            f"{policy_config_name}.model.num_input_points={int(train_settings.num_input_points)}"
        )

    task_settings = get_task_point_category_settings(eval_task_id)
    if robot_model_root_path is None:
        robot_model_root_path = train_settings.robot_model_root_path

    return EvalMapSettings(
        num_input_points=int(train_settings.num_input_points),
        map_input_type=str(eval_map_input_type),
        task_id=task_settings.task_id,
        instance_keep_all_categories=task_settings.instance_keep_all_categories,
        instance_budget_categories=task_settings.instance_budget_categories,
        robot_model_root_path=robot_model_root_path,
    )


def resolve_eval_map_config(cfg: Any) -> tuple[int, str]:
    """Resolve eval map input type and count from the served training config."""
    settings = resolve_eval_map_settings(cfg)
    return settings.num_input_points, settings.map_input_type


def _resolve_eval_task_name(cfg: Any) -> str:
    task_cfg = getattr(cfg, "task", None)
    task_name = _none_if_missing(getattr(task_cfg, "name", None))
    if task_name is None:
        raise ValueError("cfg.task.name must be set for map evaluation.")
    return str(task_name)


def _none_if_missing(value):
    if value is None:
        return None
    if isinstance(value, str) and value in ("???", "null", "None"):
        return None
    return value


def _load_training_map_settings(config_name: str) -> EvalMapSettings:
    try:
        from serf_b1k.training import config as train_config
    except ImportError as exc:
        parsed = _parse_training_config_file(config_name)
        if parsed is None:
            raise ImportError(
                "Could not import serf_b1k.training.config or parse training/config.py "
                f"for policy_config_name={config_name!r}."
            ) from exc
        return parsed

    resolved = train_config.get_config(config_name)
    base_config = resolved.data.base_config
    task_settings = get_task_point_category_settings(resolved.task_id)
    return EvalMapSettings(
        num_input_points=int(resolved.model.num_input_points),
        map_input_type=str(
            _none_if_missing(getattr(resolved.model, "map_input_type", None))
        ),
        task_id=task_settings.task_id,
        instance_keep_all_categories=task_settings.instance_keep_all_categories,
        instance_budget_categories=task_settings.instance_budget_categories,
        robot_model_root_path=_none_if_missing(
            getattr(base_config, "robot_model_root_path", None)
        ),
    )


def _parse_training_config_file(config_name: str) -> EvalMapSettings | None:
    config_path = Path(__file__).resolve().parents[1] / "training" / "config.py"
    if not config_path.exists():
        return None

    text = config_path.read_text()
    block = _find_training_config_block(text, config_name)
    if block is None:
        return None

    type_match = re.search(r'map_input_type="([^"]+)"', block)
    num_match = re.search(r"num_input_points\s*=\s*([0-9_]+)", block)
    robot_root_match = re.search(r'robot_model_root_path="([^"]+)"', block)
    task_id_match = re.search(r'task_id="([^"]+)"', block)

    map_input_type = type_match.group(1) if type_match else None
    if map_input_type is None:
        return None

    config_task_id = extract_task_id_from_config_name(config_name)
    parsed_task_id = (
        normalize_task_id(task_id_match.group(1))
        if task_id_match is not None
        else config_task_id
    )
    if parsed_task_id is None:
        return None
    if config_task_id is not None and config_task_id != parsed_task_id:
        raise ValueError(
            f"Parsed task_id {parsed_task_id!r} does not match config name "
            f"suffix {config_task_id!r} for {config_name!r}."
        )

    task_settings = get_task_point_category_settings(parsed_task_id)
    num_points = int(num_match.group(1).replace("_", "")) if num_match else 24576
    return EvalMapSettings(
        num_input_points=num_points,
        map_input_type=map_input_type,
        task_id=task_settings.task_id,
        instance_keep_all_categories=task_settings.instance_keep_all_categories,
        instance_budget_categories=task_settings.instance_budget_categories,
        robot_model_root_path=(
            robot_root_match.group(1) if robot_root_match is not None else None
        ),
    )


def _find_training_config_block(text: str, config_name: str) -> str | None:
    name_match = re.search(rf'name="{re.escape(config_name)}"', text)
    if name_match is None:
        config_task_id = extract_task_id_from_config_name(config_name)
        if config_task_id is None:
            return None
        generated_prefix = config_name[: -len(config_task_id)]
        generated_name = f'name=f"{generated_prefix}{{task_id}}"'
        name_match = re.search(re.escape(generated_name), text)
        if name_match is None:
            return None

    train_start = text.rfind("TrainConfig(", 0, name_match.start())
    if train_start < 0:
        return None
    next_train = text.find("TrainConfig(", name_match.end())
    return text[train_start : next_train if next_train >= 0 else len(text)]

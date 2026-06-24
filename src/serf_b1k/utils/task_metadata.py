from __future__ import annotations

import dataclasses
import re

from serf_b1k.utils.task_catalog import (
    get_behavior_task_index,
    get_behavior_task_name,
)


_TASK_ID_PATTERN = re.compile(r"^task-(\d{4})$")


@dataclasses.dataclass(frozen=True)
class TaskPointCategorySettings:
    task_id: str
    task_index: int
    task_name: str
    instance_keep_all_categories: tuple[str, ...]
    instance_budget_categories: tuple[str, ...]


_TASK_POINT_CATEGORY_DATA = {
    "task-0021": dict(
        instance_keep_all_categories=(
            "dice",
            "teddy_bear",
            "toy_train",
            "board_game",
        ),
        instance_budget_categories=(
            "bed",
            "bookcase",
            "desk",
        ),
    ),
    "task-0022": dict(
        instance_keep_all_categories=(
            "gym_shoe",
            "sandal",
        ),
        instance_budget_categories=(
            "hall_tree",
        ),
    ),
    "task-0026": dict(
        instance_keep_all_categories=(
            "bow",
            "swiss_cheese",
            "pillar_candle",
            "butter_cookie",
            "wicker_basket",
        ),
        instance_budget_categories=(
            "bottom_cabinet",
            "coffee_table",
        ),
    ),
}


def normalize_task_id(task_id: str) -> str:
    match = _TASK_ID_PATTERN.fullmatch(str(task_id))
    if match is None:
        raise ValueError(
            f"Invalid task_id {task_id!r}. Expected canonical format 'task-XXXX'."
        )
    return f"task-{int(match.group(1)):04d}"


def extract_task_id_from_config_name(config_name: str) -> str | None:
    match = re.search(r"(task-\d{4})$", str(config_name))
    if match is None:
        return None
    return normalize_task_id(match.group(1))


def task_id_to_index(task_id: str) -> int:
    canonical_task_id = normalize_task_id(task_id)
    return int(canonical_task_id.removeprefix("task-"))


def task_id_to_task_name(task_id: str) -> str:
    return get_behavior_task_name(task_id_to_index(task_id))


def task_name_to_task_id(task_name: str) -> str:
    task_index = get_behavior_task_index(task_name)
    return f"task-{task_index:04d}"


def get_task_point_category_settings(task_id: str) -> TaskPointCategorySettings:
    canonical_task_id = normalize_task_id(task_id)
    try:
        raw = _TASK_POINT_CATEGORY_DATA[canonical_task_id]
    except KeyError as exc:
        known_task_ids = ", ".join(sorted(_TASK_POINT_CATEGORY_DATA))
        raise ValueError(
            f"No map task metadata registered for {canonical_task_id!r}. "
            f"Known task_ids: {known_task_ids or '<none>'}."
        ) from exc

    task_index = task_id_to_index(canonical_task_id)
    task_name = task_id_to_task_name(canonical_task_id)
    return TaskPointCategorySettings(
        task_id=canonical_task_id,
        task_index=task_index,
        task_name=task_name,
        instance_keep_all_categories=tuple(raw["instance_keep_all_categories"]),
        instance_budget_categories=tuple(raw["instance_budget_categories"]),
    )

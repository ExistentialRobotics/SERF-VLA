from __future__ import annotations

from typing import Sequence


BEHAVIOR_TASK_NAME_TO_INDEX = {
    "turning_on_radio": 0,
    "picking_up_trash": 1,
    "putting_away_Halloween_decorations": 2,
    "cleaning_up_plates_and_food": 3,
    "can_meat": 4,
    "setting_mousetraps": 5,
    "hiding_Easter_eggs": 6,
    "picking_up_toys": 7,
    "rearranging_kitchen_furniture": 8,
    "putting_up_Christmas_decorations_inside": 9,
    "set_up_a_coffee_station_in_your_kitchen": 10,
    "putting_dishes_away_after_cleaning": 11,
    "preparing_lunch_box": 12,
    "loading_the_car": 13,
    "carrying_in_groceries": 14,
    "bringing_in_wood": 15,
    "moving_boxes_to_storage": 16,
    "bringing_water": 17,
    "tidying_bedroom": 18,
    "outfit_a_basic_toolbox": 19,
    "sorting_vegetables": 20,
    "collecting_childrens_toys": 21,
    "putting_shoes_on_rack": 22,
    "boxing_books_up_for_storage": 23,
    "storing_food": 24,
    "clearing_food_from_table_into_fridge": 25,
    "assembling_gift_baskets": 26,
    "sorting_household_items": 27,
    "getting_organized_for_work": 28,
    "clean_up_your_desk": 29,
    "setting_the_fire": 30,
    "clean_boxing_gloves": 31,
    "wash_a_baseball_cap": 32,
    "wash_dog_toys": 33,
    "hanging_pictures": 34,
    "attach_a_camera_to_a_tripod": 35,
    "clean_a_patio": 36,
    "clean_a_trumpet": 37,
    "spraying_for_bugs": 38,
    "spraying_fruit_trees": 39,
    "make_microwave_popcorn": 40,
    "cook_cabbage": 41,
    "chop_an_onion": 42,
    "slicing_vegetables": 43,
    "chopping_wood": 44,
    "cook_hot_dogs": 45,
    "cook_bacon": 46,
    "freeze_pies": 47,
    "canning_food": 48,
    "make_pizza": 49,
}

BEHAVIOR_TASK_NAMES = tuple(
    task_name
    for task_name, _task_index in sorted(
        BEHAVIOR_TASK_NAME_TO_INDEX.items(), key=lambda item: item[1]
    )
)


def get_behavior_task_name(task_id: int) -> str:
    try:
        return BEHAVIOR_TASK_NAMES[int(task_id)]
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"Unknown BEHAVIOR task index {task_id!r}. "
            f"Expected 0 <= task_id < {len(BEHAVIOR_TASK_NAMES)}."
        ) from exc


def get_behavior_task_index(task_name: str) -> int:
    try:
        return BEHAVIOR_TASK_NAME_TO_INDEX[str(task_name)]
    except KeyError as exc:
        raise ValueError(f"Unknown BEHAVIOR task name {task_name!r}.") from exc


def select_behavior_task_names(task_ids: Sequence[int] | None) -> list[str]:
    if task_ids is None:
        return list(BEHAVIOR_TASK_NAMES)
    return [get_behavior_task_name(task_id) for task_id in task_ids]

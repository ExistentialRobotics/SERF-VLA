from __future__ import annotations

from typing import Any, Sequence


def should_record_step_q_score(cfg: Any) -> bool:
    return bool(getattr(cfg, "record_step_q_score", True))


def collect_initial_predicate_states(
    cfg: Any,
    ground_goal_state_options: Sequence[Sequence[Any]],
) -> list[list[bool]] | None:
    if not should_record_step_q_score(cfg):
        return None
    return [[pred.evaluate() for pred in option] for option in ground_goal_state_options]


def compute_step_q_score(
    *,
    cfg: Any,
    task_success: bool,
    ground_goal_state_options: Sequence[Sequence[Any]],
    initial_predicate_states: Sequence[Sequence[bool]] | None,
) -> float | None:
    if not should_record_step_q_score(cfg):
        return None
    if task_success:
        return 1.0
    if initial_predicate_states is None:
        return float("nan")
    return max(
        sum(
            int((not initially_true) and pred.evaluate())
            for pred, initially_true in zip(option, option_previous_state)
        )
        / len(option)
        for option, option_previous_state in zip(
            ground_goal_state_options,
            initial_predicate_states,
        )
    )


def jsonable_step_q_score(q_score: float | None) -> float | None:
    if q_score is None:
        return None
    return float(q_score)

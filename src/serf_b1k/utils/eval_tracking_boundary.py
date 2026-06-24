from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EvalPolicyBoundaryState:
    """Track eval-side policy execution windows for policies without chunk state."""

    steps_since_boundary: int = 0
    boundary_ready: bool = False

    def reset(self) -> None:
        self.steps_since_boundary = 0
        self.boundary_ready = False

    def after_env_step(self, execute_steps: int | None) -> None:
        self.boundary_ready = False
        if execute_steps is None:
            self.boundary_ready = True
            self.steps_since_boundary = 0
            return

        execute_steps = int(execute_steps)
        if execute_steps <= 0:
            self.boundary_ready = True
            self.steps_since_boundary = 0
            return

        self.steps_since_boundary += 1
        if self.steps_since_boundary >= execute_steps:
            self.boundary_ready = True
            self.steps_since_boundary = 0

    def should_flush_for_policy(
        self,
        policy: Any,
        *,
        execute_steps: int | None,
    ) -> bool:
        if policy is None:
            return True

        chunk_policy = _find_policy_with_chunk_state(policy)
        if chunk_policy is None:
            if execute_steps is None:
                return True
            return self.boundary_ready

        if getattr(chunk_policy, "last_actions", None) is None:
            return True
        if execute_steps is None:
            return True
        return int(chunk_policy.action_index) >= int(execute_steps)


def _find_policy_with_chunk_state(policy: Any) -> Any | None:
    current = policy
    seen_ids: set[int] = set()
    while current is not None and id(current) not in seen_ids:
        seen_ids.add(id(current))
        if hasattr(current, "last_actions") and hasattr(current, "action_index"):
            return current
        current = getattr(current, "policy", None)
    return None

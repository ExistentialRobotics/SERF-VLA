#!/usr/bin/env python3
"""Patch BEHAVIOR-1K task 21 goal definition for SERF evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path


GOAL = """    (:goal 
        (and 
            (exists
                (?bookcase.n.01 - bookcase.n.01)
                (and 
                    (forall
                        (?die.n.01 - die.n.01)
                        (inside ?die.n.01 ?bookcase.n.01)
                    )
                    (forall
                        (?teddy.n.01 - teddy.n.01)
                        (inside ?teddy.n.01 ?bookcase.n.01)         
                    )
                    (forall
                        (?board_game.n.01 - board_game.n.01)
                        (inside ?board_game.n.01 ?bookcase.n.01)
                    )
                    (forall
                        (?train_set.n.01 - train_set.n.01)
                        (inside ?train_set.n.01 ?bookcase.n.01)
                    )
                )
            )
        )
    )"""


RELATIVE_PROBLEM_PATH = Path(
    "bddl/bddl/activity_definitions/collecting_childrens_toys/problem0.bddl"
)


def find_goal_span(text: str) -> tuple[int, int]:
    start = text.find("(:goal")
    if start < 0:
        raise ValueError("Could not find a (:goal ...) block.")

    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return start, index + 1

    raise ValueError("Found (:goal but could not find its closing parenthesis.")


def normalize_for_compare(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def patch_file(problem_path: Path, *, check: bool) -> bool:
    text = problem_path.read_text()
    start, end = find_goal_span(text)
    current_goal = text[start:end]

    if normalize_for_compare(current_goal) == normalize_for_compare(GOAL):
        print(f"[INFO] BDDL goal already patched: {problem_path}")
        return False

    if check:
        raise SystemExit(f"[ERROR] BDDL goal is not patched: {problem_path}")

    backup_path = problem_path.with_suffix(problem_path.suffix + ".serf_backup")
    if not backup_path.exists():
        backup_path.write_text(text)

    patched = text[:start] + GOAL + text[end:]
    problem_path.write_text(patched)
    print(f"[INFO] Patched BDDL goal: {problem_path}")
    print(f"[INFO] Backup: {backup_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--behavior-root",
        type=Path,
        default=Path("BEHAVIOR-1K"),
        help="Path to the BEHAVIOR-1K repository root.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify that the goal is already patched.",
    )
    args = parser.parse_args()

    problem_path = args.behavior_root / RELATIVE_PROBLEM_PATH
    if not problem_path.exists():
        raise SystemExit(f"[ERROR] BDDL file not found: {problem_path}")

    patch_file(problem_path, check=args.check)


if __name__ == "__main__":
    main()

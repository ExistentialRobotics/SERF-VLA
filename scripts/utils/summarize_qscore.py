#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Optional, List, Tuple


FILENAME_RE = re.compile(r"^(?P<task>.+)_(?P<episode>\d+)_0\.json$")


def extract_episode_index(path: Path) -> Optional[int]:
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    return int(m.group("episode"))


def to_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def extract_q_final(obj: dict) -> Optional[float]:
    """
    Expected:
      "q_score": { "final": 0.222 }
    """
    if not isinstance(obj, dict):
        return None
    if "q_score" not in obj:
        return None
    qs = obj["q_score"]
    if not isinstance(qs, dict):
        return None
    return to_float(qs.get("final"))


def make_text_table(rows: List[Tuple[int, float, str]]) -> str:
    headers = ["episode_index", "q_score_final", "filename"]
    data = [
        [str(ep), f"{q:.6f}", fname]
        for ep, q, fname in rows
    ]

    # column width 계산
    cols = list(zip(*([headers] + data)))
    widths = [max(len(cell) for cell in col) for col in cols]

    def sep():
        return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def row(cells):
        return "| " + " | ".join(cell.ljust(w) for cell, w in zip(cells, widths)) + " |"

    lines = []
    lines.append(sep())
    lines.append(row(headers))
    lines.append(sep())
    for r in data:
        lines.append(row(r))
    lines.append(sep())

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str, help="JSON 파일들이 있는 폴더")
    ap.add_argument("-o", "--output", default="qscore_summary.txt")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    rows = []
    skipped = 0

    for p in sorted(root.rglob("*.json")):
        ep = extract_episode_index(p)
        if ep is None:
            skipped += 1
            continue

        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            skipped += 1
            continue

        qf = extract_q_final(obj)
        if qf is None:
            skipped += 1
            continue

        rows.append((ep, qf, p.name))

    rows.sort(key=lambda x: x[0])

    mean_q = None
    std_q = None
    if rows:
        mean_q = sum(r[1] for r in rows) / len(rows)
        std_q = (sum((r[1] - mean_q)**2 for r in rows) / len(rows))**0.5
    

    table = make_text_table(rows)

    out_lines = [
        "Q-score Summary",
        f"Root folder: {root}",
        f"Valid episodes: {len(rows)}",
        f"Skipped files: {skipped}",
        "",
        table,
        "",
        f"Mean q_score_final: {mean_q:.6f}" if mean_q is not None else "Mean q_score_final: N/A",
        f"Std q_score_final: {std_q:.6f}" if std_q is not None else "Std q_score_final: N/A",
        "",
    ]

    output_path = Path(root, args.output).resolve()
    output_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"[OK] Saved to {output_path}")


if __name__ == "__main__":
    main()

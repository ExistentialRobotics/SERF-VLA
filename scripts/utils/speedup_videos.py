#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Speed up all videos under a folder recursively, and write outputs under video_xN/ preserving structure.

Example:
  python speedup_videos.py --in_dir video --speed 2.0
  python speedup_videos.py --in_dir video --speed 4 --out_dir video_x4 --ext mp4
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import List


VIDEO_EXTS_DEFAULT = ["mp4", "mov", "mkv", "webm", "avi", "m4v"]


def which_or_die(bin_name: str) -> str:
    p = shutil.which(bin_name)
    if not p:
        raise SystemExit(f"[ERROR] '{bin_name}' not found in PATH. Please install it and retry.")
    return p


def atempo_chain(speed: float) -> str:
    """
    ffmpeg atempo supports only [0.5, 2.0].
    For speed-up (>1), we factor into multiple atempo filters so that each is within [0.5,2.0].

    Example:
      speed=4.0 -> "atempo=2.0,atempo=2.0"
      speed=3.0 -> "atempo=2.0,atempo=1.5"
    """
    if speed <= 0:
        raise ValueError("speed must be > 0")

    # For speed < 0.5 or between 0.5~2.0, still handle (though user asked speed-up)
    filters: List[float] = []
    remaining = speed

    # If speeding up
    while remaining > 2.0 + 1e-9:
        filters.append(2.0)
        remaining /= 2.0

    # If slowing down a lot (not typical here), break into 0.5 steps
    while remaining < 0.5 - 1e-9:
        filters.append(0.5)
        remaining /= 0.5

    filters.append(remaining)

    # Clamp tiny floating errors
    def clamp(x: float) -> float:
        if x < 0.5:
            return 0.5
        if x > 2.0:
            return 2.0
        return x

    filters = [clamp(x) for x in filters]
    return ",".join(f"atempo={x:.8f}".rstrip("0").rstrip(".") for x in filters)


def run(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout)


def has_audio_stream(ffprobe: str, video_path: Path) -> bool:
    # Check if there is any audio stream.
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        # If ffprobe fails, assume audio might exist; we’ll try with audio and fallback.
        return True
    return proc.stdout.strip() != ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir", default="video", help="Input root folder (default: video)")
    ap.add_argument("--speed", type=float, required=True, help="Speed factor (e.g., 2.0 means 2x faster)")
    ap.add_argument("--out_dir", default=None, help="Output root folder (default: video_xN)")
    ap.add_argument("--ext", default=None, help="Force output extension (e.g., mp4). Default: keep input ext.")
    ap.add_argument("--preset", default="veryfast", help="ffmpeg libx264 preset (default: veryfast)")
    ap.add_argument("--crf", type=int, default=23, help="Quality (lower=better, larger=smaller). Default: 23")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    ap.add_argument("--dry_run", action="store_true", help="Print commands without running")
    ap.add_argument("--exts", nargs="*", default=VIDEO_EXTS_DEFAULT, help="Video extensions to include")
    args = ap.parse_args()

    if args.speed <= 0:
        raise SystemExit("[ERROR] --speed must be > 0")

    ffmpeg = which_or_die("ffmpeg")
    ffprobe = which_or_die("ffprobe")

    in_dir = Path(args.in_dir).resolve()
    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"[ERROR] Input folder not found: {in_dir}")

    # Default output dir name: video_x{N} (use clean formatting)
    if args.out_dir is None:
        # Keep a readable folder name (2 -> "2", 2.5 -> "2p5")
        n = args.speed
        if abs(n - round(n)) < 1e-9:
            tag = str(int(round(n)))
        else:
            tag = str(n).replace(".", "p")
        out_dir = in_dir.parent / f"{in_dir.name}_x{tag}"
    else:
        out_dir = Path(args.out_dir).resolve()

    exts = {e.lower().lstrip(".") for e in args.exts}

    inputs: List[Path] = []
    for p in in_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower().lstrip(".") in exts:
            inputs.append(p)

    if not inputs:
        raise SystemExit(f"[ERROR] No video files found under {in_dir} with exts={sorted(exts)}")

    out_dir.mkdir(parents=True, exist_ok=True)

    setpts = f"setpts=PTS/{args.speed:.10f}".rstrip("0").rstrip(".")
    atempo = atempo_chain(args.speed)

    print(f"[INFO] in_dir : {in_dir}")
    print(f"[INFO] out_dir: {out_dir}")
    print(f"[INFO] speed  : {args.speed}x")
    print(f"[INFO] files  : {len(inputs)}")
    print("")

    done = 0
    skipped = 0
    failed = 0

    for src in inputs:
        rel = src.relative_to(in_dir)
        dst_parent = out_dir / rel.parent
        dst_parent.mkdir(parents=True, exist_ok=True)

        out_ext = args.ext if args.ext else src.suffix.lstrip(".")
        dst = (dst_parent / rel.name).with_suffix("." + out_ext)

        if dst.exists() and not args.overwrite:
            skipped += 1
            continue

        # Build ffmpeg command
        # Video: libx264 + preset + crf
        # Audio: aac + atempo chain (if audio exists)
        # Note: if you want max speed and don't care about quality, increase crf (e.g., 28~32).
        audio_exists = has_audio_stream(ffprobe, src)

        base_cmd = [
            ffmpeg,
            "-hide_banner",
            "-y" if args.overwrite else "-n",
            "-i",
            str(src),
            "-filter:v",
            setpts,
            "-c:v",
            "libx264",
            "-preset",
            args.preset,
            "-crf",
            str(args.crf),
            "-movflags",
            "+faststart",
        ]

        if audio_exists:
            cmd = base_cmd + [
                "-filter:a",
                atempo,
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(dst),
            ]
        else:
            cmd = base_cmd + [
                "-an",
                str(dst),
            ]

        if args.dry_run:
            print("[DRY]", " ".join(cmd))
            done += 1
            continue

        try:
            run(cmd)
            done += 1
        except Exception as e:
            # Fallback: sometimes atempo fails due to weird audio; retry without audio.
            try:
                fallback_cmd = base_cmd + ["-an", str(dst)]
                run(fallback_cmd)
                done += 1
            except Exception:
                failed += 1
                print(f"[FAIL] {src} -> {dst}")
                msg = str(e)
                if msg:
                    print("       ", msg.splitlines()[-1])

    print("")
    print(f"[OK] done={done}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()

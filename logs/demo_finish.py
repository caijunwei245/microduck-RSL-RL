"""Post-process the per-skill demonstration videos into one reel plus a contact sheet.

`logs/skill_demo.py --video` writes one mp4 per skill (`logs/demo_videos/<skill>.mp4`) with every
frame labelled `skill (round N)`. This script rebuilds the two human-facing artifacts:

    logs/demo_videos/all_skills_2rounds.mp4   every round of every skill, back to back
    logs/demo_videos/round2_montage.png       one round-2 frame per skill, tiled

Both are *presentation*, not measurement: the verdicts live in the printed table and in
`logs/acceptance_gate.sh`. Concatenation uses the ffmpeg concat demuxer (all clips come from one
cfg, so codec/size/fps match) instead of loading ~7 minutes of frames into RAM.

Usage:  uv run python logs/demo_finish.py [--rounds 2] [--video-dir logs/demo_videos]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

# Skill order = the order of the SKILLS table in skill_demo.py, i.e. the order a viewer wants to
# watch them in (walk first, everything else after). Names are the on-disk stems.
ORDER = [
    "velocity_walk",
    "velocity_turn",
    "walk_turn",
    "ball_kick_right",
    "sitstand",
    "ground_pick",
    "rollers_(fast)",
    "swizzle",
    "roller_slope",
    "roller_standup",
    "roller_crouch",
    "standup_floor_flip",
    "velstand_floor_flip",
    "spin",
    "roulade",
]


def _probe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def _probe_size(path: str) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    w, h = out.split("x")[:2]
    return int(w), int(h)


def concat(video_dir: str, order: list[str], out_path: str) -> int:
    present = [os.path.join(video_dir, f"{n}.mp4") for n in order]
    present = [p for p in present if os.path.exists(p)]
    missing = [n for n in order if not os.path.exists(os.path.join(video_dir, f"{n}.mp4"))]
    if not present:
        print("no clips found - nothing to concatenate")
        return 1
    if missing:
        print(f"note: {len(missing)} expected clip(s) absent: {', '.join(missing)}")
    list_path = os.path.join(video_dir, "_concat.txt")
    with open(list_path, "w") as fh:
        for p in present:
            fh.write(f"file '{os.path.abspath(p)}'\n")
    if os.path.exists(out_path):
        os.remove(out_path)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", list_path,
         "-c", "copy", out_path],
        check=True,
    )
    total = sum(_probe_duration(p) for p in present)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"concat -> {out_path}  ({len(present)} clips, {int(total)//60}m{int(total)%60:02d}s, "
          f"{size_mb:.1f} MB, {_probe_size(out_path)[0]}x{_probe_size(out_path)[1]})")
    return 0


def montage(video_dir: str, order: list[str], out_path: str, rounds: int, cols: int = 4) -> int:
    """One frame per skill, taken from the LAST round (that is what `round2_montage` means)."""
    try:
        import imageio.v2 as iio
        import numpy as np
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - tooling only
        print(f"montage skipped ({exc})")
        return 0

    clips = [(n, os.path.join(video_dir, f"{n}.mp4")) for n in order]
    clips = [(n, p) for n, p in clips if os.path.exists(p)]
    if not clips:
        print("no clips found - nothing to tile")
        return 1

    tiles: list[tuple[str, "np.ndarray"]] = []
    for name, path in clips:
        reader = iio.get_reader(path)
        meta = reader.get_meta_data()
        n_frames = reader.count_frames()
        # round N occupies roughly the last 1/rounds of the clip; sample its middle.
        if rounds > 1:
            target = int(n_frames * (1.0 - 0.5 / rounds))
        else:
            target = n_frames // 2
        target = max(0, min(n_frames - 1, target))
        frame = reader.get_data(target)
        reader.close()
        tiles.append((name, frame))
        print(f"  tile {name:22s} frame {target:5d}/{n_frames} ({meta.get('duration', 0):.1f}s clip)")

    h, w = tiles[0][1].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    pad = 6
    label_h = 22
    sheet = Image.new("RGB", (cols * w + (cols + 1) * pad, rows * (h + label_h) + (rows + 1) * pad),
                      (24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    for i, (name, frame) in enumerate(tiles):
        r, c = divmod(i, cols)
        x = pad + c * (w + pad)
        y = pad + r * (h + label_h + pad)
        sheet.paste(Image.fromarray(frame), (x, y + label_h))
        draw.text((x + 4, y + 5), f"{name}  (round {rounds})", fill=(235, 235, 235))
    sheet.save(out_path)
    print(f"montage -> {out_path}  ({len(tiles)} tiles, {sheet.width}x{sheet.height})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", default="logs/demo_videos")
    ap.add_argument("--rounds", type=int, default=2, help="rounds per clip, for round-N sampling")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--only-concat", action="store_true")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found", file=sys.stderr)
        return 2
    rc = concat(args.video_dir, ORDER, os.path.join(args.video_dir, "all_skills_2rounds.mp4"))
    if rc or args.only_concat:
        return rc
    return montage(args.video_dir, ORDER, os.path.join(args.video_dir, "round2_montage.png"),
                   args.rounds, args.cols)


if __name__ == "__main__":
    raise SystemExit(main())

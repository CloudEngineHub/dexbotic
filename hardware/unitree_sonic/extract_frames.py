"""Pre-extract SONIC mp4 episodes into per-frame JPEGs (the fast DexDataset path).

Dexbotic's ``LoadMultiModal`` decodes a fresh ``VideoReader`` for every sample
(~175 ms/sample) when frames are stored as ``type='video'``. Storing frames as
individual images (``type='image'``) switches it to a plain image read (a few
ms), matching the image-based datasets (LIBERO etc.).

This converts a dexdata dir::

    <in>/jsonl/episode_XXXXXX.jsonl      (images_1 = {type:'video', url, frame_idx})
    <in>/video/episode_XXXXXX_ego_view.mp4

into::

    <out>/jsonl/episode_XXXXXX.jsonl     (images_1 = {type:'image', url})
    <out>/images/episode_XXXXXX/<frame_idx:06d>.jpg

Frames are resized to ``--size`` (default 256x256), which is exactly what the
training image preprocess (Gr00tSonicImagePreprocess) does anyway — so the model
input is identical, just much faster to load.

Usage (from repo root, dexbotic env):
    python -m hardware.unitree_sonic.extract_frames \
        --in-dir ./data/dexbotic_pingzi --out-dir ./data/dexbotic_pingzi_img \
        --size 256 --workers 16
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
import json
import os

import numpy as np
from PIL import Image


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def convert_episode(
    jsonl_path: str,
    video_dir: str,
    out_jsonl_dir: str,
    out_images_dir: str,
    size: int,
    quality: int,
    chunk: int,
) -> tuple[str, int]:
    """Convert one episode; returns (episode, num_frames). Idempotent."""
    from decord import VideoReader, cpu

    episode = os.path.splitext(os.path.basename(jsonl_path))[0]
    out_jsonl = os.path.join(out_jsonl_dir, f"{episode}.jsonl")
    if os.path.exists(out_jsonl):
        return episode, -1  # already done

    records = _load_jsonl(jsonl_path)
    if not records:
        return episode, 0

    # Every per-frame image view: images_1, images_2, ... (each its own video).
    image_keys = sorted(
        k for k, v in records[0].items() if k.startswith("images") and isinstance(v, dict)
    )
    target = (size, size) if size else None

    # Decode each view's video into its own subdir: <episode>/<images_k>/<fi>.jpg
    for key in image_keys:
        video_name = os.path.basename(str(records[0][key]["url"]))
        video_path = os.path.join(video_dir, video_name)
        frame_idxs = [int(r[key]["frame_idx"]) for r in records]
        view_dir = os.path.join(out_images_dir, episode, key)
        os.makedirs(view_dir, exist_ok=True)

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        n_video = len(vr)
        # Decode in chunks to bound memory (a full 480x640 episode is ~1.3 GB).
        for start in range(0, len(frame_idxs), chunk):
            batch = frame_idxs[start : start + chunk]
            safe = [min(i, n_video - 1) for i in batch]
            frames = vr.get_batch(safe).asnumpy()  # (B, H, W, 3) uint8
            for fi, frame in zip(batch, frames):
                img = Image.fromarray(frame).convert("RGB")
                if target is not None:
                    img = img.resize(target)
                img.save(os.path.join(view_dir, f"{fi:06d}.jpg"), quality=quality)
        del vr

    # Rewrite jsonl: each video view -> image view pointing at the saved JPEGs.
    tmp = out_jsonl + ".tmp"
    with open(tmp, "w") as f:
        for r in records:
            r = dict(r)
            for key in image_keys:
                fi = int(r[key]["frame_idx"])
                r[key] = {"type": "image", "url": f"{episode}/{key}/{fi:06d}.jpg"}
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, out_jsonl)
    return episode, len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True, help="dexdata dir with jsonl/ + video/")
    parser.add_argument("--out-dir", required=True, help="output dir for jsonl/ + images/")
    parser.add_argument("--size", type=int, default=256, help="resize HxW (0 keeps original)")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=64, help="frames decoded per batch")
    args = parser.parse_args()

    jsonl_dir = os.path.join(args.in_dir, "jsonl")
    video_dir = os.path.join(args.in_dir, "video")
    out_jsonl_dir = os.path.join(args.out_dir, "jsonl")
    out_images_dir = os.path.join(args.out_dir, "images")
    os.makedirs(out_jsonl_dir, exist_ok=True)
    os.makedirs(out_images_dir, exist_ok=True)

    episodes = sorted(glob.glob(os.path.join(jsonl_dir, "*.jsonl")))
    print(f"Converting {len(episodes)} episodes from {args.in_dir} -> {args.out_dir}")

    done = 0
    total_frames = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                convert_episode,
                jp,
                video_dir,
                out_jsonl_dir,
                out_images_dir,
                args.size,
                args.quality,
                args.chunk,
            )
            for jp in episodes
        ]
        for fut in as_completed(futs):
            ep, n = fut.result()
            done += 1
            if n >= 0:
                total_frames += n
            tag = "skip" if n < 0 else f"{n} frames"
            print(f"[{done}/{len(episodes)}] {ep}: {tag}", flush=True)

    print(f"Done. {done} episodes, {total_frames} frames written to {out_images_dir}")


if __name__ == "__main__":
    main()

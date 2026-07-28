"""Convert a Unitree G1 SONIC LeRobot dataset to dexbotic dex_data format.

The SONIC VLA (GR00T) consumes:
  - 1 ego-view RGB image (monocular)
  - a 46-d proprioceptive state  (joint groups + projected gravity)
  - a 78-d action               (motion_token[64] + left/right hand joints[7+7])

This mirrors the so101/xlerobot converters: it reads the LeRobot parquet +
videos and writes one jsonl per episode plus the matching ego-view mp4, using
the dex_data schema (prompt / state / action / images_1 / extra).

The state/action layouts follow the official ``unitree_g1_sonic`` embodiment
config (gr00t/configs/data/embodiment_configs.py) so that a later weight
conversion from the nvidia GR00T-N1.7 checkpoint stays key/column compatible.

Multi-view: pass ``--cameras`` (comma-separated LeRobot video columns) to export
several views; the i-th becomes ``images_{i}``. The single-camera default
reproduces the original ego-view output exactly.

Output layout (the -o dir is itself the dataset folder; no extra sub-folder):
    <out>/jsonl/episode_XXXXX.jsonl
    <out>/video/episode_XXXXX_<camera>.mp4    (url in jsonl = the bare filename)

Usage:
    # single ego view (default)
    python convert_sonic_to_dexdata.py -i <lerobot_dataset_dir> -o <out_dir>
    # multi-view
    python convert_sonic_to_dexdata.py -i <in> -o <out> \
        -c observation.images.ego_view,observation.images.left_wrist
"""

import json
import os
import shutil
import glob

import click
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

try:
    import pyarrow.parquet as pq
except ImportError:
    logger.error("Missing pyarrow. Please run: pip install pyarrow")
    raise SystemExit(1)


# --- state layout (46-d), in official unitree_g1_sonic modality_keys order -----
# Each tuple slices the raw `observation.state` (43-d) column. Note the raw
# column stores arms/hands as [left_arm, left_hand, right_arm, right_hand], but
# the embodiment concatenation order is arms-then-hands, so we reorder here.
STATE_SLICES_FROM_OBS = [
    ("left_leg", 0, 6),
    ("right_leg", 6, 12),
    ("waist", 12, 15),
    ("left_arm", 15, 22),
    ("right_arm", 29, 36),
    ("left_hand", 22, 29),
    ("right_hand", 36, 43),
]
STATE_DIM = 46  # 43 joint dims (reordered) + 3 projected_gravity
ACTION_DIM = 78  # motion_token[64] + left_hand_joints[7] + right_hand_joints[7]

# Default LeRobot video columns per camera COUNT, in view order. ``--cameras N``
# picks the list for count N (1 = mono ego view, 2 = stereo ego left/right). You
# can still pass explicit comma-separated column names instead of a number.
DEFAULT_CAMERAS_BY_COUNT = {
    1: ["observation.images.ego_view"],
    2: ["observation.images.ego_view_left", "observation.images.ego_view_right"],
}


def resolve_cameras(cameras_arg: str) -> list[str]:
    """Resolve --cameras: a count (int -> default keys) or explicit name list."""
    val = str(cameras_arg).strip()
    if val.isdigit():
        n = int(val)
        if n not in DEFAULT_CAMERAS_BY_COUNT:
            raise SystemExit(
                f"--cameras {n}: no default camera set for {n} views. "
                f"Supported counts: {sorted(DEFAULT_CAMERAS_BY_COUNT)}; "
                f"or pass explicit comma-separated column names."
            )
        return list(DEFAULT_CAMERAS_BY_COUNT[n])
    return [c.strip() for c in val.split(",") if c.strip()]


def camera_short_name(camera_key: str) -> str:
    """e.g. 'observation.images.ego_view' -> 'ego_view'."""
    return camera_key.split(".")[-1]


def get_task_list(meta_dir):
    """Return task strings indexed by task_index (from tasks.jsonl/parquet)."""
    jsonl_path = os.path.join(meta_dir, "tasks.jsonl")
    if os.path.exists(jsonl_path):
        by_index = {}
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                info = json.loads(line)
                by_index[int(info.get("task_index", len(by_index)))] = str(
                    info.get("task", info.get("instruction", ""))
                )
        if by_index:
            return [by_index[i] for i in sorted(by_index)]

    parquet_path = os.path.join(meta_dir, "tasks.parquet")
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        for col in ["task", "instruction", "language_instruction"]:
            if col in df.columns:
                return df[col].astype(str).tolist()
    return []


def assemble_state(row) -> np.ndarray:
    """Build the 46-d state vector for one frame."""
    obs_state = np.asarray(row["observation.state"], dtype=np.float32)
    parts = [obs_state[s:e] for _, s, e in STATE_SLICES_FROM_OBS]
    proj_g = np.asarray(row["observation.projected_gravity"], dtype=np.float32)
    parts.append(proj_g)
    state = np.concatenate(parts, axis=0)
    assert state.shape[0] == STATE_DIM, f"state dim {state.shape[0]} != {STATE_DIM}"
    return state


def assemble_action(row) -> np.ndarray:
    """Build the 78-d action vector for one frame."""
    motion_token = np.asarray(row["action.motion_token"], dtype=np.float32)
    left_hand = np.asarray(row["teleop.left_hand_joints"], dtype=np.float32)
    right_hand = np.asarray(row["teleop.right_hand_joints"], dtype=np.float32)
    action = np.concatenate([motion_token, left_hand, right_hand], axis=0)
    assert action.shape[0] == ACTION_DIM, f"action dim {action.shape[0]} != {ACTION_DIM}"
    return action


def parse_one_episode(df: pd.DataFrame, task_list, camera_rel_urls):
    """Turn one episode dataframe into a list of dex_data frame dicts.

    camera_rel_urls: list of per-camera mp4 filenames (in view order). The i-th
    entry becomes ``images_{i}`` (1-based), so a single-camera list reproduces the
    original ``images_1``-only output.
    """
    data_list = []
    for _, row in df.iterrows():
        state = assemble_state(row)
        action = assemble_action(row)

        task_index = int(row.get("task_index", 0))
        if task_list and task_index < len(task_list):
            prompt = str(task_list[task_index])
        else:
            prompt = "unknown task"

        frame_index = int(row.get("frame_index", 0))
        # dex_data frame schema (see docs/Data.md "Data Format"):
        # images_N + state + prompt + is_robot (+ optional explicit action).
        entry = {
            "state": state.tolist(),
            "action": action.tolist(),
            "prompt": prompt,
            "is_robot": True,
        }
        for i, rel_url in enumerate(camera_rel_urls, start=1):
            entry[f"images_{i}"] = {
                "type": "video",
                "url": rel_url,
                "frame_idx": frame_index,
            }
        data_list.append(entry)
    return data_list


def find_episode_video(video_root, chunk_name, camera_key, stem):
    """Locate the LeRobot mp4 for one (episode, camera). Returns path or None."""
    src = os.path.join(video_root, chunk_name, camera_key, f"{stem}.mp4")
    if os.path.exists(src):
        return src
    # Fallback: some exports omit the chunk dir under videos/.
    cand = glob.glob(
        os.path.join(video_root, "**", camera_key, f"{stem}.mp4"), recursive=True
    )
    return cand[0] if cand else None


def list_available_cameras(video_root):
    """Scan videos/ and return the camera column names that actually have mp4s."""
    keys = set()
    for root, _dirs, files in os.walk(video_root):
        if any(f.endswith(".mp4") for f in files):
            keys.add(os.path.basename(root))
    return sorted(keys)


def save_jsonl(data_list, jsonl_path):
    with open(jsonl_path, "w") as f:
        for data in data_list:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


@click.command()
@click.option("-i", "--lerobot_dir", type=str, required=True, help="LeRobot dataset dir")
@click.option("-o", "--output_dir", type=str, required=True,
              help="dex_data output dir (itself the dataset folder; "
                   "writes <out>/jsonl and <out>/video directly)")
@click.option("-c", "--cameras", type=str, default="1",
              help="Camera COUNT (e.g. 1=ego_view, 2=ego_view_left+right) OR an "
                   "explicit comma-separated list of LeRobot video columns in view "
                   "order (the i-th becomes images_{i}). Example: "
                   "'observation.images.ego_view,observation.images.left_wrist'")
def main(lerobot_dir, output_dir, cameras):
    camera_keys = resolve_cameras(cameras)
    camera_names = [camera_short_name(c) for c in camera_keys]
    logger.info(f"Cameras (view order): {list(zip(camera_keys, camera_names))}")
    meta_dir = os.path.join(lerobot_dir, "meta")
    task_list = get_task_list(meta_dir)
    if not task_list:
        logger.warning("No tasks found in meta/; prompts will be 'unknown task'")
    else:
        logger.info(f"Tasks: {task_list}")

    data_root = os.path.join(lerobot_dir, "data")
    video_root = os.path.join(lerobot_dir, "videos")
    if not os.path.isdir(data_root):
        logger.error(f"data/ not found under {lerobot_dir}")
        raise SystemExit(1)

    # Validate camera keys against what actually exists, fail fast with the list.
    available = list_available_cameras(video_root)
    logger.info(f"Available cameras in {video_root}: {available}")
    missing_keys = [k for k in camera_keys if k not in available]
    if missing_keys:
        logger.error(
            f"Requested camera(s) not found: {missing_keys}. "
            f"Use one of the available cameras above via --cameras "
            f"(comma-separated, full column names)."
        )
        raise SystemExit(1)

    out_jsonl_dir = os.path.join(output_dir, "jsonl")
    out_video_dir = os.path.join(output_dir, "video")
    os.makedirs(out_jsonl_dir, exist_ok=True)
    os.makedirs(out_video_dir, exist_ok=True)

    chunk_dirs = sorted(d for d in os.listdir(data_root)
                        if os.path.isdir(os.path.join(data_root, d)))
    n_done, n_skip = 0, 0
    for chunk_name in chunk_dirs:
        chunk_path = os.path.join(data_root, chunk_name)
        parquet_files = sorted(f for f in os.listdir(chunk_path) if f.endswith(".parquet"))

        for episode_file in tqdm(parquet_files, desc=chunk_name):
            try:
                df = pq.read_table(os.path.join(chunk_path, episode_file)).to_pandas()
            except Exception as e:
                logger.error(f"Bad parquet {episode_file}: {e}")
                n_skip += 1
                continue

            stem = episode_file.replace(".parquet", "")  # episode_000000

            # Resolve every camera's source video first; skip the whole episode if
            # any view is missing so all frames keep a consistent set of views.
            cam_srcs, cam_rel_urls = [], []
            missing = None
            for key, name in zip(camera_keys, camera_names):
                src = find_episode_video(video_root, chunk_name, key, stem)
                if src is None:
                    missing = name
                    break
                cam_srcs.append(src)
                cam_rel_urls.append(f"{stem}_{name}.mp4")
            if missing is not None:
                logger.warning(f"No '{missing}' video for {stem}; skipping episode")
                n_skip += 1
                continue

            data = parse_one_episode(df, task_list, cam_rel_urls)
            if not data:
                n_skip += 1
                continue

            save_jsonl(data, os.path.join(out_jsonl_dir, f"{stem}.jsonl"))
            for src, rel_url in zip(cam_srcs, cam_rel_urls):
                shutil.copy2(src, os.path.join(out_video_dir, rel_url))
            n_done += 1

    logger.info(f"Done. episodes converted={n_done}, skipped={n_skip}")
    logger.info(f"jsonl -> {out_jsonl_dir}")
    logger.info(f"video -> {out_video_dir}")


if __name__ == "__main__":
    main()

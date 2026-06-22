import json
import math
from pathlib import Path
from typing import Any

import click
import numpy as np
from loguru import logger
from tqdm import tqdm


DEFAULT_PROMPT = "Put the blue bowl into the beige bowl, and put the green bowl into the blue bowl."
CAMERA_FILES = {
    "images_1": "cam_high_rgb.mp4",
    "images_2": "cam_left_wrist_rgb.mp4",
    "images_3": "cam_right_wrist_rgb.mp4",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_task_prompt(task_dir: Path, fallback_prompt: str) -> str:
    candidates = [task_dir / "meta" / "task_info.json", task_dir / "task_desc.json"]
    for path in candidates:
        if not path.exists():
            continue
        data = load_json(path)
        task_desc = data.get("task_desc", data)
        prompt = str(task_desc.get("prompt", "")).strip()
        if prompt:
            return prompt
    return fallback_prompt


def get_joint_positions(record: dict[str, Any]) -> list[float]:
    joints = record.get("joint_positions")
    if joints is None:
        raise KeyError("State record must contain `joint_positions`")
    if len(joints) != 6:
        raise ValueError(f"Expected 6 joint positions, got {len(joints)}")
    return [float(v) for v in joints]


def get_gripper(record: dict[str, Any]) -> float:
    for key in ("gripper_width", "gripper"):
        if key in record:
            return float(record[key])
    raise KeyError("State record must contain `gripper_width` or `gripper`")


def build_state(left: dict[str, Any], right: dict[str, Any]) -> list[float]:
    return get_joint_positions(left) + [get_gripper(left)] + get_joint_positions(right) + [get_gripper(right)]


def is_valid_state(state: list[float], gripper_min: float, gripper_max: float) -> bool:
    arr = np.asarray(state, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return False
    joints = np.concatenate([arr[0:6], arr[7:13]])
    if np.any(joints < -math.pi) or np.any(joints > math.pi):
        return False
    grippers = arr[[6, 13]]
    if np.any(grippers < gripper_min) or np.any(grippers > gripper_max):
        return False
    return True


def is_static_frame(prev: list[float], cur: list[float], joint_atol: float, gripper_atol: float) -> bool:
    prev_arr = np.asarray(prev, dtype=np.float64)
    cur_arr = np.asarray(cur, dtype=np.float64)
    return (
        np.allclose(prev_arr[0:6], cur_arr[0:6], atol=joint_atol)
        and np.allclose(prev_arr[7:13], cur_arr[7:13], atol=joint_atol)
        and abs(prev_arr[6] - cur_arr[6]) <= gripper_atol
        and abs(prev_arr[13] - cur_arr[13]) <= gripper_atol
    )


def convert_episode(
    episode_dir: Path,
    prompt: str,
    remove_static: bool,
    joint_atol: float,
    gripper_atol: float,
    gripper_min: float,
    gripper_max: float,
) -> list[dict[str, Any]]:
    meta = load_json(episode_dir / "meta" / "episode_meta.json")
    left_records = load_jsonl(episode_dir / "states" / "left_states.jsonl")
    right_records = load_jsonl(episode_dir / "states" / "right_states.jsonl")
    videos_dir = episode_dir / "videos"

    for video_name in CAMERA_FILES.values():
        video_path = videos_dir / video_name
        if not video_path.exists():
            raise FileNotFoundError(video_path)

    frame_count = min(int(meta.get("frames", len(left_records))), len(left_records), len(right_records))
    if frame_count <= 0:
        return []

    records: list[dict[str, Any]] = []
    last_kept_state: list[float] | None = None

    for frame_idx in range(frame_count):
        left = left_records[frame_idx]
        right = right_records[frame_idx]
        state = build_state(left, right)

        if not is_valid_state(state, gripper_min, gripper_max):
            continue
        if remove_static and last_kept_state is not None and is_static_frame(
            last_kept_state, state, joint_atol, gripper_atol
        ):
            continue

        timestamp = left.get("timestamp", right.get("timestamp"))
        item: dict[str, Any] = {
            "state": state,
            "prompt": prompt,
            "is_robot": True,
            "extra": {
                "timestamp": float(timestamp) if timestamp is not None else None,
                "episode": episode_dir.name,
                "robot_id": meta.get("robot_id"),
            },
        }

        for image_key, video_name in CAMERA_FILES.items():
            item[image_key] = {
                "type": "video",
                "url": str((videos_dir / video_name).resolve()),
                "frame_idx": frame_idx,
            }

        records.append(item)
        last_kept_state = state

    return records


def write_index_cache(output_task_dir: Path) -> None:
    jsonl_files = sorted(output_task_dir.glob("**/*.jsonl"))
    index_cache = {
        "meta_data": {
            "total_samples": 0,
            "total_jsonl_files": len(jsonl_files),
        },
        "data": {},
    }
    for jsonl_file in jsonl_files:
        count = sum(1 for line in jsonl_file.open("r", encoding="utf-8") if line.strip())
        index_cache["data"][str(jsonl_file.resolve())] = count
        index_cache["meta_data"]["total_samples"] += count

    with (output_task_dir / "index_cache.json").open("w", encoding="utf-8") as f:
        json.dump(index_cache, f, indent=2)


@click.command()
@click.option("-i", "--task-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Table30v2 task directory, e.g. /path/to/Table30v2/stack_bowls.")
@click.option("-o", "--output-dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="DexData output root. JSONL files will be written to <output-dir>/<task-name>.")
@click.option("--task-name", type=str, default=None,
              help="Task subdirectory name under output-dir. Defaults to input task directory name.")
@click.option("--prompt", type=str, default=None,
              help="Override prompt. Defaults to task_info.json/task_desc.json prompt.")
@click.option("--keep-static", is_flag=True, help="Keep static frames instead of removing them.")
@click.option("--joint-atol", type=float, default=5e-4, show_default=True,
              help="Joint threshold used for static-frame removal.")
@click.option("--gripper-atol", type=float, default=1e-3, show_default=True,
              help="Gripper threshold used for static-frame removal.")
@click.option("--gripper-min", type=float, default=-1e-3, show_default=True,
              help="Minimum valid gripper value.")
@click.option("--gripper-max", type=float, default=8e-2, show_default=True,
              help="Maximum valid gripper value.")
@click.option("--max-episodes", type=int, default=None,
              help="Optional limit for quick conversion tests.")
def main(
    task_dir: Path,
    output_dir: Path,
    task_name: str | None,
    prompt: str | None,
    keep_static: bool,
    joint_atol: float,
    gripper_atol: float,
    gripper_min: float,
    gripper_max: float,
    max_episodes: int | None,
) -> None:
    task_name = task_name or task_dir.name
    prompt = prompt or get_task_prompt(task_dir, DEFAULT_PROMPT)
    output_task_dir = output_dir / task_name
    output_task_dir.mkdir(parents=True, exist_ok=True)

    data_dir = task_dir / "data"
    episode_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("episode")])
    if max_episodes is not None:
        episode_dirs = episode_dirs[:max_episodes]
    if not episode_dirs:
        raise FileNotFoundError(f"No episode directories found under {data_dir}")

    total_samples = 0
    failed: list[dict[str, str]] = []
    next_episode_idx = len(list(output_task_dir.glob("episode_*.jsonl")))

    logger.info(f"Using prompt: {prompt}")
    for episode_dir in tqdm(episode_dirs, desc="Converting Table30v2 DOS-W1 episodes"):
        try:
            records = convert_episode(
                episode_dir=episode_dir,
                prompt=prompt,
                remove_static=not keep_static,
                joint_atol=joint_atol,
                gripper_atol=gripper_atol,
                gripper_min=gripper_min,
                gripper_max=gripper_max,
            )
            if not records:
                failed.append({"episode": episode_dir.name, "error": "no valid samples"})
                continue
            out_path = output_task_dir / f"episode_{next_episode_idx:05d}.jsonl"
            save_jsonl(records, out_path)
            next_episode_idx += 1
            total_samples += len(records)
        except Exception as exc:
            logger.exception(f"Failed to convert {episode_dir}")
            failed.append({"episode": episode_dir.name, "error": str(exc)})

    write_index_cache(output_task_dir)
    if failed:
        save_jsonl(failed, output_dir / f"{task_name}_table30v2_convert_failed.jsonl")

    logger.info(f"Converted {len(episode_dirs) - len(failed)} episodes into {output_task_dir}")
    logger.info(f"Total valid samples in this run: {total_samples}")
    if failed:
        logger.warning(f"Failed episodes: {len(failed)}; see {output_dir / f'{task_name}_table30v2_convert_failed.jsonl'}")


if __name__ == "__main__":
    main()

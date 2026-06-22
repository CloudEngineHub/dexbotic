# DOS-W1 Stack Bowls DM0 SFT Tutorial

This document describes the DOS-W1 stack-bowls SFT workflow built from `hardware/dosw1/dm0_stack_bowls.py`. It covers the full path from raw DOS-W1 demonstrations, DexData conversion, dataset registration, DM0 SFT training, and inference validation.

## Overview

The task is:

```text
Stack the two smaller bowls on top of the largest bowl one by one.
```

A demonstration video can be added under the DOS-W1 hardware directory as:

```text
hardware/dosw1/demo_stack_bowls.mp4
```

The video is optional for running the code, but it is useful for showing the expected final behavior of the trained policy.

The policy is trained with DM0 using three camera views and a 14-dimensional DOS-W1 joint state:

```text
0:6    left arm joint positions
6      left gripper
7:13   right arm joint positions
13     right gripper
```

During training, the 14-dimensional robot state/action is padded to the model action dimension `32`. The two gripper dimensions `[6, 13]` are marked as `non_delta_mask`, so they are not converted to delta action.

## Data Processing

### Raw Data

This tutorial uses the public DOS-W1 stack-bowls task format from [RoboChallenge/Table30v2](https://huggingface.co/datasets/RoboChallenge/Table30v2).

Place the task directory at:

```text
/path/to/Table30v2/stack_bowls
```

The expected layout is:

```text
stack_bowls/
├── task_desc.json
├── meta/task_info.json
└── data/
    └── episode_000000/
        ├── meta/episode_meta.json
        ├── states/left_states.jsonl
        ├── states/right_states.jsonl
        └── videos/
            ├── cam_high_rgb.mp4
            ├── cam_left_wrist_rgb.mp4
            └── cam_right_wrist_rgb.mp4
```

`states/left_states.jsonl` and `states/right_states.jsonl` store one JSON object per robot frame. Each line should contain at least:

```json
{
  "timestamp": 1750000000.123,
  "joint_positions": [0.0, 0.1, -0.2, 1.0, 0.3, -1.2],
  "ee_positions": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
  "gripper_width": 0.025
}
```

For this DM0 SFT config, only `joint_positions` and `gripper_width` are used to build the 14D state/action. `ee_positions` may exist in the raw data, but is intentionally ignored.

### Run Conversion

Convert the public Table30v2 DOS-W1 task to DexData:

```bash
cd /path/to/dexbotic

python hardware/dosw1/convert_table30v2_dosw1_to_dexdata.py \
  --task-dir /path/to/Table30v2/stack_bowls \
  --output-dir /path/to/dexdata \
  --task-name stack_bowls
```

The converter writes JSONL files and `index_cache.json` under `/path/to/dexdata/stack_bowls`. The video paths in the JSONL records point back to the source episode videos.

### DexData Output

After conversion, place the DexData output at:

```text
/path/to/dexdata
```

The training script uses the task subdirectory:

```text
/path/to/dexdata/stack_bowls
```

A sample JSONL record contains:

```python
{
    "images_1": {"type": "video", "url": ".../cam_high_rgb.mp4", "frame_idx": 0},
    "images_2": {"type": "video", "url": ".../cam_left_wrist_rgb.mp4", "frame_idx": 0},
    "images_3": {"type": "video", "url": ".../cam_right_wrist_rgb.mp4", "frame_idx": 0},
    "state": [/* 14 joint + gripper values */],
    "prompt": "Stack the two smaller bowls on top of the largest bowl one by one.",
    "is_robot": True,
}
```

### Conversion Logic

The converted DexData should follow these rules.

Frame alignment follows the Table30v2 episode order. For each frame index, use the same index from the left state stream, right state stream, and all three camera videos. The effective frame count is the minimum of `episode_meta.frames`, left-state length, and right-state length.

State construction uses only joint positions and gripper values:

```python
left_state = left_joint_positions + [left_gripper_width]
right_state = right_joint_positions + [right_gripper_width]
state = left_state + right_state
```

This produces a 14-dimensional state:

```text
0:6    left joint positions
6      left gripper
7:13   right joint positions
13     right gripper
```

The conversion intentionally excludes end-effector pose fields. If your raw data includes both joint states and EEF states, keep the joint-only 14D representation for this DM0 SFT config.

Prompt handling:

- Read the prompt from `meta/task_info.json` or `task_desc.json`.
- If the prompt is empty, pass `--prompt` explicitly when running the converter.

Invalid state filtering:

- Skip samples with invalid or missing joint/gripper values.
- A typical safe range is `[-pi, pi]` for joint positions and a robot-specific valid range for gripper values.
- Keep this validation consistent with your robot calibration.

Static-frame removal is recommended before training. Keep the first valid frame, then compare each candidate frame against the last kept frame. Drop the candidate if both arms and both grippers are effectively unchanged:

```python
np.allclose(prev[0:6], cur[0:6], atol=5e-4)
np.allclose(prev[7:13], cur[7:13], atol=5e-4)
abs(prev[6] - cur[6]) <= 1e-3
abs(prev[13] - cur[13]) <= 1e-3
```

Recommended thresholds:

```text
joint atol:    5e-4
gripper atol:  1e-3
```

The final DexData directory should contain one JSONL file per valid episode and an `index_cache.json` generated for fast loading.

## Dataset Registration

The SFT config is:

```text
/path/to/dexbotic/hardware/dosw1/dm0_stack_bowls.py
```

It registers the DOS-W1 dataset inline:

```python
DOS_W1_JOINT_META = {
    "non_delta_mask": [6, 13],
    "periodic_mask": None,
    "periodic_range": None,
    "robot_type": "DOS-W1",
    "control_mode": "joint",
    "norm_stats_path": None,
    "fps": 30,
}

DOS_W1_DATASET = {
    "stack_bowls": {
        "annotations": "/path/to/dexdata/stack_bowls",
        "meta_data": DOS_W1_JOINT_META,
        "frequency": 1,
    },
}

register_dataset(DOS_W1_DATASET, prefix="dosw1")
```

The final dataset name used by training is:

```text
dosw1_stack_bowls
```

## SFT Training

### Model and Action Pipeline

Download the DM0 base checkpoint from Hugging Face:

```bash
git clone https://huggingface.co/Dexmal/DM0-base /path/to/DM0-base
```

Model page: [Dexmal/DM0-base](https://huggingface.co/Dexmal/DM0-base)

Then set `DM0ModelConfig.model_name_or_path` to the downloaded checkpoint directory:

```text
/path/to/DM0-base
```

The action pipeline is:

```python
ToDict()
ToNumpy()
AddAction()
PadState(ndim=32, axis=-1)
PadAction(ndim=32, axis=-1)
AddTrajectory(trajectory_length=50, flatten=False, padding_mode="last")
DeltaAction(enable=True)
ActionNorm(..., use_quantiles=True)
LoadMultiModal(return_masks=True)
ToList()
```

The model trains on 50-step action chunks. `DeltaAction` converts joint actions into relative actions except for gripper dimensions `[6, 13]`.

### Normalization Stats

When training starts, DM0 automatically computes normalization statistics if they do not exist. The norm directory is derived from the dataset name:

```python
hashlib.md5("dosw1_stack_bowls".encode()).hexdigest()[:8]
```

The exact hash depends on the final dataset name. The expected automatic norm file is:

```text
dexbotic/norm_assets/<dataset_hash>/norm_stats.json
```

If a checkpoint contains its own `norm_stats.json`, inference will prefer the checkpoint-local file. Otherwise the stack-bowls inference config falls back to the dataset hash path above.

### Start Training

Run from the repository root:

```bash
cd /path/to/dexbotic

torchrun --nproc_per_node=1 \
  hardware/dosw1/dm0_stack_bowls.py \
  --task train \
  --data_dir /path/to/dexdata/stack_bowls \
  --model_name_or_path /path/to/DM0-base
```

For single-GPU training, keep `--nproc_per_node=1`. For multi-GPU training, set `--nproc_per_node` to the number of GPUs used on the current node. If you need to select specific GPUs, set `CUDA_VISIBLE_DEVICES` before launching, for example `CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 ...`.

Use `torchrun` even for single-GPU training. The training backend is DeepSpeed, and launching the training script directly with `python hardware/dosw1/dm0_stack_bowls.py --task train ...` may trigger MPI discovery in some environments and fail with `ModuleNotFoundError: No module named mpi4py`.

Main training defaults in the config:

```text
num_train_steps: 40,000
save_steps: 2,000
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
bf16: True
output_dir: ./user_checkpoints/dexbotic/dosw1_dm0/stack_bowls-<MMDD>
```

Successful training logs should include `fm_loss`, `grad_norm`, `learning_rate`, `step`, and `epoch`. A minimal one-step smoke test should at least print a loss/gradient line, for example:

```text
{"loss": ..., "grad_norm": ..., "learning_rate": ..., "epoch": ...}
```

A saved checkpoint should contain `norm_stats.json`; inference relies on this file to use the same action/state normalization as training.

## Inference Validation

### Start the DM0 Policy Server

Choose a trained checkpoint and set it in `DM0InferenceConfig.model_name_or_path`, or keep the configured default if it already points to the desired checkpoint:

```python
model_name_or_path = "/path/to/trained_checkpoint"
```

Then start inference:

```bash
cd /path/to/dexbotic

python hardware/dosw1/dm0_stack_bowls.py \
  --task inference \
  --model_name_or_path /path/to/trained_checkpoint
```

The policy server listens on:

```text
http://<server-ip>:7891
```

During startup, confirm that the logs show the expected normalization stats and model loading messages. A healthy startup should include lines similar to:

```text
Reading normalization stats from <checkpoint>/norm_stats.json
Model loaded successfully
Running on http://<server-ip>:7891
```

The loading priority is:

```text
1. <checkpoint>/norm_stats.json
2. dexbotic/norm_assets/<dataset_hash>/norm_stats.json
```

### Validate Inputs

The inference side expects the same observation convention as training:

- Three images ordered as front, left, right.
- A 14D DOS-W1 joint state.
- The same task prompt used during training.

The prompt should match:

```text
Stack the two smaller bowls on top of the largest bowl one by one.
```

### Real-Robot Connection

This repository currently provides SO-101 and XLeRobot bridge/client examples. For DOS-W1, connect the robot-side client or bridge to the DM0 policy server above and send observations in the same format used by the training data:

```text
images_1 -> front camera
images_2 -> left camera
images_3 -> right camera
state    -> 14D DOS-W1 joint + gripper state
prompt   -> stack-bowls instruction
```

The returned action is denormalized and converted back to absolute action by the inference config:

```python
ActionDenorm(..., use_quantiles=True)
AbsoluteAction()
```

The final executable action dimension is 14, with grippers at dimensions `[6, 13]`.

## Success Indicators

1. Training automatically creates or finds `dexbotic/norm_assets/<dataset_hash>/norm_stats.json`.
2. Checkpoints are saved under `user_checkpoints/dexbotic/dosw1_dm0/stack_bowls-<MMDD>`.
3. Inference logs show model loading and normalization stats without falling back to `{"min": -1, "max": 1}`.
4. The robot-side validation receives 14D actions with stable gripper dimensions and no obvious scale mismatch.

## Troubleshooting

If training fails with `ModuleNotFoundError: No module named mpi4py`, launch training with `torchrun --nproc_per_node=1` even when using a single GPU.

If action scale looks wrong, first check which norm file inference loaded. A missing or mismatched norm file often causes incorrect action magnitude. Prefer using a checkpoint that contains `<checkpoint>/norm_stats.json`; otherwise inference falls back to `dexbotic/norm_assets/<dataset_hash>/norm_stats.json`.

If dataset loading fails, confirm that `dosw1_stack_bowls` is registered by `dm0_stack_bowls.py` and that the annotations path exists:

```bash
ls /path/to/dexdata/stack_bowls
```

If prompts are empty, inspect the conversion output and ensure the fallback prompt was applied. Newer index-cache logic may filter empty prompts.

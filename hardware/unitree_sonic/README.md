# Unitree G1 SONIC / gr00tsonic

This directory contains the Dexbotic data conversion and serving tools for
fine-tuning **GR00T-N1.7** on the Unitree G1 SONIC whole-body action layout.
The model uses the Qwen3-VL-based `nvidia/Cosmos-Reason2-2B` backbone and the
GR00T-N1.7 flow-matching action head.

The supported workflow is:

```text
GR00T-WholeBodyControl / gear_sonic LeRobot data
        |
        |  hardware/unitree_sonic/convert_sonic_to_dexdata.py
        v
Dexbotic dex_data dataset
        |
        |  playground/example_gr00tsonic_exp.py --task train
        v
Dexbotic gr00tsonic checkpoint
        |
        |  hardware/unitree_sonic/bridge.py
        v
gear_sonic PolicyClient / run_vla_inference.py
```

Run all commands below from the Dexbotic repository root:

```bash
cd dexbotic
conda activate dexbotic
```

The converter additionally needs `click`, `pandas`, and `pyarrow`. The ZMQ
bridge needs `pyzmq`, `msgpack`, and `msgpack-numpy`. Install them in the
Dexbotic environment if they are not already available:

```bash
pip install click pandas pyarrow pyzmq msgpack msgpack-numpy
```

## SONIC data layout

The current implementation uses:

- State: 46 dimensions
  - `left_leg(6) + right_leg(6) + waist(3)`
  - `left_arm(7) + right_arm(7)`
  - `left_hand(7) + right_hand(7)`
  - `projected_gravity(3)`
- Action: 78 dimensions
  - `motion_token(64) + left_hand_joints(7) + right_hand_joints(7)`
- Action horizon: 40
- Padded state/action size: 132
- State history length: 1
- SONIC embodiment ID: 11
- Default flow-matching inference steps: 4

The converter expects a LeRobot-style dataset with parquet files under `data/`
and videos under `videos/`. It optionally reads task metadata from
`meta/tasks.jsonl` or `meta/tasks.parquet`; without it, prompts are set to
`unknown task`. Each parquet frame must contain:

- `observation.state` (43 joint values)
- `observation.projected_gravity` (3 values)
- `action.motion_token` (64 values)
- `teleop.left_hand_joints` (7 values)
- `teleop.right_hand_joints` (7 values)
- one or more video columns

The raw 43-dimensional `observation.state` stores arms and hands in a different
order. The converter reorders it to the SONIC state layout shown above before
appending projected gravity.

## 1. Convert LeRobot data to dex_data

The checked-in example uses the two-camera `beef_pie_xsh` dataset:

```bash
python hardware/unitree_sonic/convert_sonic_to_dexdata.py \
    -i <lerobot_dataset_dir> \
    -o ./data/beef_pie_xsh \
    --cameras 2
```

`--cameras 2` selects these LeRobot columns in order:

```text
observation.images.ego_view_left
observation.images.ego_view_right
```

The result is:

```text
data/beef_pie_xsh/
├── jsonl/
│   └── episode_000000.jsonl
└── video/
    ├── episode_000000_ego_view_left.mp4
    └── episode_000000_ego_view_right.mp4
```

Each JSONL record contains `images_1`, `images_2`, `state`, `action`, `prompt`,
and `is_robot`. Camera order is preserved: the first `--cameras` entry becomes
`images_1`, the second becomes `images_2`, and so on.

For the default mono ego view:

```bash
python hardware/unitree_sonic/convert_sonic_to_dexdata.py \
    -i <lerobot_dataset_dir> \
    -o ./data/<dataset_name> \
    --cameras 1
```

`--cameras 1` selects `observation.images.ego_view`. Custom camera columns can
be passed as a comma-separated list in training view order:

```bash
python hardware/unitree_sonic/convert_sonic_to_dexdata.py \
    -i <lerobot_dataset_dir> \
    -o ./data/<dataset_name> \
    --cameras observation.images.ego_view_left,observation.images.ego_view_right
```

## 2. Register the dataset

Dataset sources are registered in
`dexbotic/data/data_source/sonic_official.py`. The module is imported
automatically, and `prefix="sonic"` is added to every dictionary key.

The current registrations are:

| Training name | Storage | Path |
| --- | --- | --- |
| `sonic_dexbotic_pingzi` | MP4 | `./data/dexbotic_pingzi` |
| `sonic_dexbotic_pingzi_img` | extracted images | `./data/dexbotic_pingzi_img` |
| `sonic_beef_pie_xsh` | MP4 | `./data/beef_pie_xsh` |

To add another MP4 dataset:

```python
SONIC_DATASET = {
    # Existing entries...
    "<dataset_name>": {
        "data_path_prefix": "./data/<dataset_name>/video",
        "annotations": "./data/<dataset_name>/jsonl",
        "frequency": 1,
    },
}
```

Use `sonic_<dataset_name>` in the experiment configuration. These paths are
relative to the current working directory, which is why training should be
launched from the repository root.

## 3. Optional: pre-extract video frames

Raw MP4 datasets are supported, but decoding video for every sample is slower.
Pre-extract frames for the faster image loading path:

```bash
python hardware/unitree_sonic/extract_frames.py \
    --in-dir ./data/beef_pie_xsh \
    --out-dir ./data/beef_pie_xsh_img \
    --size 256 \
    --workers 16
```

For a two-view episode this produces:

```text
data/beef_pie_xsh_img/
├── jsonl/
│   └── episode_000000.jsonl
└── images/
    └── episode_000000/
        ├── images_1/000000.jpg
        └── images_2/000000.jpg
```

The script is idempotent at episode level: it skips an episode when its output
JSONL already exists. Register the extracted dataset with an `images` prefix:

```python
"beef_pie_xsh_img": {
    "data_path_prefix": "./data/beef_pie_xsh_img/images",
    "annotations": "./data/beef_pie_xsh_img/jsonl",
    "frequency": 1,
},
```

Then set `dataset_name` to `sonic_beef_pie_xsh_img`.

## 4. Configure training

The reusable defaults live in `dexbotic/exp/gr00tsonic_exp.py`:

- dataset: `sonic_beef_pie_xsh`
- camera views: 1
- model initialization: `nvidia/GR00T-N1.7-3B`
- frozen language and visual backbone
- trainable projector, diffusion model, and VLLN

The runnable `playground/example_gr00tsonic_exp.py` currently overrides them for
the two-camera `beef_pie_xsh` run:

```python
@dataclass
class DataConfig(Gr00tSonicDataConfig):
    dataset_name: str = field(default="sonic_beef_pie_xsh")
    num_images: int = field(default=2)
```

Its current training defaults are:

- DeepSpeed ZeRO-2 via `./script/deepspeed/zero2.json`
- 30,000 training steps and save interval 30,000
- batch size 32 per GPU, gradient accumulation 1
- bfloat16 training
- 16 data loader workers
- gradient checkpointing disabled
- output directory `./checkpoints`
- Weights & Biases project `dexbotic_gr00tsonic`

For another dataset, update `dataset_name` and make `num_images` equal the
number of `images_N` views emitted by the converter.

### Model files

`pretrained_gr00t_path` accepts either a local GR00T-N1.7 checkpoint directory
or the Hugging Face ID `nvidia/GR00T-N1.7-3B`. HF IDs are resolved with
`local_files_only=True`, so the GR00T-N1.7 files must already exist in the
Hugging Face cache. The Cosmos config and Qwen3-VL processor must also be
available to `transformers`.


### Action normalization

Action normalization statistics are computed automatically on the first
training launch. The merged file is cached at:

```text
dexbotic/norm_assets/<dataset-name-hash>/norm_stats.json
```

Compute it without starting training:

```bash
python playground/example_gr00tsonic_exp.py --task compute_norm_stats
```

The training data pipeline uses the explicit 78-dimensional absolute `action`
field, builds a 40-step trajectory, applies quantile normalization, and pads it
to 132 dimensions. Constant action dimensions are set to zero after
normalization, and normalized targets are clipped to `[-1, 1]`.

At training initialization, `norm_stats.json` is also written to the configured
output directory. Periodic checkpoints receive a copy as well.

## 5. Train

Eight GPUs:

```bash
torchrun --nproc_per_node=8 \
    playground/example_gr00tsonic_exp.py --task train
```

One-GPU smoke test:

```bash
torchrun --nproc_per_node=1 \
    playground/example_gr00tsonic_exp.py --task train
```

With the checked-in example configuration, the final Hugging Face-compatible
model and `norm_stats.json` are saved directly under `./checkpoints`. Change
`TrainerConfig.output_dir` before training if checkpoints from other runs are
also stored there.

## 6. Serve the checkpoint

The current training example writes to `./checkpoints`, while the bridge's own
fallback path is `./checkpoints/Dexbotic-GR00TSonic`. Pass the training output
explicitly:

```bash
python hardware/unitree_sonic/bridge.py \
    --model-path ./checkpoints \
    --host '*' \
    --port 7899 \
    --cameras 2
```

By default, the bridge reads `./checkpoints/norm_stats.json`. Use
`--norm-stats <path>` if the statistics are stored elsewhere.

Camera count and order must match training:

- `--cameras 1` reads `observation["video"]["ego_view"]`
- `--cameras 2` reads `observation["video"]["ego_view_left"]`, then
  `observation["video"]["ego_view_right"]`
- a custom comma-separated list reads those `observation["video"]` keys in the
  given order

For example:

```bash
python hardware/unitree_sonic/bridge.py \
    --model-path <checkpoint_dir> \
    --host '*' \
    --port 7899 \
    --cameras ego_view_left,ego_view_right \
    --num-steps 4
```

`--num-steps` overrides the checkpoint's flow-matching inference-step setting.
The bridge assembles state from:

```text
observation["state"]:
    left_leg, right_leg, waist,
    left_arm, right_arm, left_hand, right_hand,
    projected_gravity
```

It reads the prompt from
`observation["language"]["annotation.human.task_description"][0][0]` and
returns the action structure expected by the official SONIC runner:

```text
motion_token[1, T, 64]
left_hand_joints[1, T, 7]
right_hand_joints[1, T, 7]
```

`T` is 40 with the current model configuration.

## 7. Run on the G1 robot

On the robot or `gear_sonic` side, keep the official runner and point it to the
Dexbotic bridge:

```bash
python gear_sonic/scripts/run_vla_inference.py \
    --host <dexbotic_server_ip> \
    --port 7899
```

The GR00T-WholeBodyControl WBC and the camera/state ZMQ stack remain on the robot
side. Only the policy server is replaced by
`hardware/unitree_sonic/bridge.py`.

## Troubleshooting

- Camera missing during conversion: pass the exact video column names shown by
  the converter's `Available cameras` log.
- Missing or incorrectly ordered views: keep converter `--cameras`,
  `DataConfig.num_images`, and bridge `--cameras` consistent.
- Slow data loading: run `extract_frames.py`, register the resulting `images`
  directory, and switch `dataset_name` to that registration.
- Dataset not found: launch from the repository root and verify the relative
  paths in `sonic_official.py`.
- Model files not found: populate the Hugging Face cache or use local model
  directories; the GR00T-N1.7 initialization path does not download missing
  files.
- Incorrect action scale: verify that the checkpoint directory contains the
  `norm_stats.json` generated for the same training dataset.
- Bridge import error: install `pyzmq`, `msgpack`, and `msgpack-numpy` in the
  same environment used to run the bridge.

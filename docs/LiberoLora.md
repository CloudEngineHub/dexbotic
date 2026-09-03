# LIBERO LoRA SFT Training

This page is the shared developer reference for LIBERO LoRA SFT recipes. It
currently covers PI0, PI05, DM0, CogACT, and DM05.

Full SFT entrypoints are listed only as baseline references. New LIBERO LoRA
work should use the LoRA entrypoints in `playground/benchmarks/libero`.

| Model | Full baseline entrypoint | LoRA entrypoint | LoRA backend |
| --- | --- | --- | --- |
| PI0 | `playground/benchmarks/libero/libero_pi0.py` | `playground/benchmarks/libero/libero_pi0_lora.py` | DDP only |
| PI05 | `playground/benchmarks/libero/libero_pi05.py` | `playground/benchmarks/libero/libero_pi05_lora.py` | DDP only |
| DM0 | `playground/benchmarks/libero/libero_dm0.py` | `playground/benchmarks/libero/libero_dm0_lora.py` | DDP only |
| CogACT | `playground/benchmarks/libero/libero_cogact.py` | `playground/benchmarks/libero/libero_cogact_lora.py` | DDP only |
| DM05 | `playground/benchmarks/libero/libero_dm05.py` | `playground/benchmarks/libero/libero_dm05_lora.py` | DDP only |

## Effectiveness Reference

These results use the four-suite LIBERO protocol with `2000` total episodes.
Treat them as recipe validation references, not guarantees for every machine or
storage setup.

### PI0

The best open all-only Full SFT baseline reached `1923/2000 = 96.15%` at 50k.
The best open all-only LoRA checkpoint reached `1905/2000 = 95.25%` at 42k.

| Recipe | Checkpoint | Overall LIBERO success | Note |
| --- | ---: | ---: | --- |
| Full SFT | 50k | `1923/2000 = 96.15%` | Best open all-only Full baseline |
| LoRA SFT | 42k | `1905/2000 = 95.25%` | Best open all-only LoRA checkpoint |
| Gap | - | `-0.90 pp` | LoRA remains within the 1-2 pp engineering target |

For same-step comparison, PI0 LoRA 50k reached `1897/2000 = 94.85%`, which is
`-1.30 pp` versus Full SFT 50k.

### PI05

The validated PI05 LoRA recipe reached `1936/2000 = 96.80%` at 50k. The best
Full SFT baseline reached `1918/2000 = 95.90%` at 50k.

| Recipe | Checkpoint | Overall LIBERO success | Note |
| --- | ---: | ---: | --- |
| Full SFT F1 | 50k | `1918/2000 = 95.90%` | Best Full baseline |
| LoRA SFT | 50k | `1936/2000 = 96.80%` | Best validated LoRA checkpoint |
| Gap | - | `+0.90 pp` | LoRA is higher than Full at the best checkpoint |

At 40k, the same checkpoint family shows LoRA at `1928/2000 = 96.40%` versus
Full SFT at `1912/2000 = 95.60%`, a `+0.80 pp` same-step gain.

### DM0

The validated DM0 LoRA recipe reached `1855/2000 = 92.75%` at 148k. The best
DM0 Full SFT baseline reached `1900/2000 = 95.00%` at 138k and 142k.

| Recipe | Checkpoint | Overall LIBERO success | Note |
| --- | ---: | ---: | --- |
| Full SFT | 138k / 142k | `1900/2000 = 95.00%` | Best validated Full baseline |
| LoRA SFT | 148k | `1855/2000 = 92.75%` | Best validated LoRA checkpoint |
| Gap | - | `-2.25 pp` | LoRA uses about 2.14% trainable parameters |

### CogACT

The validated CogACT LoRA recipe reached `1931/2000 = 96.55%` at 110k. The
best validated Full SFT checkpoint reached `1925/2000 = 96.25%` at 100k.

| Recipe | Checkpoint | Overall LIBERO success | Note |
| --- | ---: | ---: | --- |
| Full SFT | 100k | `1925/2000 = 96.25%` | Best validated Full checkpoint |
| LoRA SFT | 110k | `1931/2000 = 96.55%` | Best validated LoRA checkpoint |
| Gap | - | `+0.30 pp` | LoRA is higher at the best checkpoint |

The final LoRA checkpoint at 150k reached `1921/2000 = 96.05%`. Both results
use 7D LIBERO actions, one training camera, no augmentation, and the complete
four-suite 2,000-episode protocol.

### DM05

The validated DM05 LoRA recipe reached `1966/2000 = 98.30%` at 49k. The best
Full SFT baseline reached `1980/2000 = 99.0%` at 50k.

| Recipe | Checkpoint | Overall LIBERO success | Note |
| --- | ---: | ---: | --- |
| Full SFT | 50k | `1980/2000 = 99.0%` | Best Full baseline |
| LoRA SFT | 49k | `1966/2000 = 98.30%` | Best validated LoRA checkpoint |
| Gap | - | `-0.70 pp` | LoRA remains within 1 pp of Full |

At 20k / 30k / 40k, the same LoRA run reached `84.55%` / `93.95%` /
`98.00%`. All scores use the four-suite 2,000-episode protocol.

## LIBERO Data

PI0, PI05, DM0, CogACT, and DM05 use the built-in `libero_pi0_all` training target.

| Field | Value |
| --- | --- |
| Dataset name | `libero_pi0_all` |
| JSONL root | `data/libero/libero_pi0_all/jsonl` |
| Image root | `data/libero/libero_pi0_all/image` |
| Image count | `3` views per sample |
| State/action padding | `32D` |

After preparation, the dataset should have this layout:

```text
data/libero/libero_pi0_all/
  jsonl/
    *.jsonl
  image/
    ...
```

Each JSONL row should include image references, robot state, action, and prompt.

PI0 stacks 50 future actions, trains with delta action targets, and converts the
output back to absolute LIBERO actions for inference. PI05 uses a 10-step action
chunk and should run LoRA training and LoRA inference with `chunk_size=10`. DM0
uses a 50-step action trajectory and the same three-view LIBERO input layout.
CogACT uses the first camera, a 16-step action chunk, and raw 7D actions. Its
entrypoint computes a separate raw-action norm-stat cache so older delta-action
or 8D CogACT statistics are not reused. DM05 uses `agentview` + `wrist` and a
10-step absolute chunk; inference needs `norm_stats.json` beside the checkpoint.

## Prepare a Docker Workspace

The public Docker image used by the main project docs is `dexmal/dexbotic`. Use
`dexmal/dexbotic:c130t28` instead on Blackwell GPUs.

On the host:

```bash
git clone https://github.com/dexmal/dexbotic.git
cd dexbotic

mkdir -p data checkpoints user_checkpoints .cache/huggingface

docker run -it --rm --gpus all --network host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD":/dexbotic \
  -v "$PWD/data":/dexbotic/data \
  -v "$PWD/checkpoints":/dexbotic/checkpoints \
  -v "$PWD/user_checkpoints":/dexbotic/user_checkpoints \
  -v "$PWD/.cache/huggingface":/root/.cache/huggingface \
  dexmal/dexbotic \
  bash
```

Inside the container:

```bash
cd /dexbotic
conda activate dexbotic
pip install -e .
pip install -U huggingface_hub
```

## Download Base Models

Download the base model required by the recipe you plan to run:

```bash
huggingface-cli download Dexmal/Dexbotic-PI0 \
  --local-dir checkpoints/Dexbotic-PI0 \
  --local-dir-use-symlinks False

huggingface-cli download Dexmal/Dexbotic-PI05 \
  --local-dir checkpoints/Dexbotic-PI05 \
  --local-dir-use-symlinks False

huggingface-cli download Dexmal/DM0-base \
  --local-dir checkpoints/DM0-base \
  --local-dir-use-symlinks False

huggingface-cli download Dexmal/DM05 \
  --local-dir checkpoints/DM05 \
  --local-dir-use-symlinks False

huggingface-cli download Dexmal/Dexbotic-Base \
  --local-dir checkpoints/Dexbotic-Base \
  --local-dir-use-symlinks False

test -f checkpoints/Dexbotic-PI0/config.json
test -f checkpoints/Dexbotic-PI05/config.json
test -f checkpoints/DM0-base/config.json
test -f checkpoints/DM05/config.json
test -f checkpoints/Dexbotic-Base/config.json
```

Keep each full model directory intact, including processor and tokenizer files.
If a checkpoint is mounted from another storage system, either keep the final
path above or edit the selected entrypoint's `model_name_or_path`.

## Download LIBERO Training Data

Download the LIBERO dataset from Hugging Face:

```bash
huggingface-cli download Dexmal/libero \
  --repo-type dataset \
  --local-dir data/.hf_downloads/libero \
  --local-dir-use-symlinks False
```

Organize the downloaded files under `data/libero`. If the download already
contains a `libero_pi0_all` directory, copy it directly:

```bash
mkdir -p data/libero
cp -a data/.hf_downloads/libero/libero_pi0_all data/libero/
```

If the download contains archive files instead, extract them into `data/libero`
and then verify that `data/libero/libero_pi0_all/jsonl` and
`data/libero/libero_pi0_all/image` exist.

```bash
find data/.hf_downloads/libero -maxdepth 2 -type f \
  \( -name "*.tar" -o -name "*.tar.gz" -o -name "*.tgz" -o -name "*.zip" \)

# Example for tar archives. Run this for the archive files in the download.
tar -xf /path/to/libero_archive.tar -C data/libero

test -d data/libero/libero_pi0_all/jsonl
test -d data/libero/libero_pi0_all/image
```

## Optional: Compute Norm Stats

Training normally computes and saves norm stats through the standard Dexbotic
trainer path. To precompute them explicitly:

```bash
python playground/benchmarks/libero/libero_pi0.py --task compute_norm_stats
python playground/benchmarks/libero/libero_pi05.py --task compute_norm_stats
python playground/benchmarks/libero/libero_dm0_lora.py --task compute_norm_stats
python playground/benchmarks/libero/libero_cogact_lora.py \
  --task compute_norm_stats
python playground/benchmarks/libero/libero_dm05_lora.py --task compute_norm_stats
```

Recompute norm stats if the dataset, action representation, action chunk length,
or trajectory length changes.

## PI0 LoRA Recipe

| Item | Reference value |
| --- | --- |
| Base model | `checkpoints/Dexbotic-PI0` |
| Dataset | `libero_pi0_all` |
| Training backend | DDP only |
| LoRA target modules | `all-linear` |
| LoRA rank / alpha / dropout | `32` / `16` / `0.0` |
| Dense modules to save | `state_proj`, `action_in_proj`, `action_time_mlp_in`, `action_time_mlp_out`, `action_out_proj` |
| Global batch | `64` |
| Per-device batch | `4` |
| Gradient accumulation | `1` |
| Train steps | `50,000` |
| Save interval | `2,000` steps |
| Optimizer | AdamW over LoRA weights and `modules_to_save` dense weights |
| LR / warmup | `5e-4` / `500` |
| Raw backward / warmup | disabled |
| Model max length | `48` |
| Action trajectory length | `50` |

Run from inside the Docker container. PI0 LoRA SFT is supported only with DDP;
the entrypoint rejects DeepSpeed and FSDP backends.

```bash
NPROC_PER_NODE=8

torchrun --nproc_per_node="${NPROC_PER_NODE}" \
  playground/benchmarks/libero/libero_pi0_lora.py \
  --task train \
  --train-backend ddp
```

The default output directory is:

```text
user_checkpoints/dexbotic/libero_all_pi0/pi0_lora_sft_libero_50k
```

The trainable summary is written to:

```text
user_checkpoints/dexbotic/libero_all_pi0/trainable_summaries/pi0_lora_sft_libero_50k.json
```

Check the summary before trusting a run:

- `r` should be `32`.
- `lora_alpha` should be `16`.
- `target_modules` should resolve from `all-linear`.
- `unexpected_trainable_parameters` should be empty.
- The dense saved modules should include state/action projections and action time
  MLPs.

For inference, pass a LoRA checkpoint path to `--model_name_or_path`. The loader
reads `adapter_config.json`, loads the recorded base model, and merges the
adapter before serving.

```bash
python playground/benchmarks/libero/libero_pi0_lora.py \
  --task inference \
  --model_name_or_path \
  user_checkpoints/dexbotic/libero_all_pi0/pi0_lora_sft_libero_50k/checkpoint-50000 \
  --port 7891
```

If the base model path recorded in `adapter_config.json` is not available on the
inference machine, pass `--base_model_name_or_path /path/to/Dexbotic-PI0` or
place the base model at the recorded path before starting inference.

## PI05 LoRA Recipe

| Item | Reference value |
| --- | --- |
| Base model | `checkpoints/Dexbotic-PI05` |
| Dataset | `libero_pi0_all` |
| Training backend | DDP only |
| LoRA target modules | PI05 `llm` and `action_expert` q/k/v/o/gate/up/down projections |
| LoRA rank / alpha / dropout | `32` / `16` / `0.0` |
| Dense modules to save | `action_in_proj`, `action_out_proj`, `time_mlp_in`, `time_mlp_out` |
| Extra full-rank trainables | action expert AdaRMS dense weights and bias |
| Global batch | `64` |
| Per-device batch | `4` |
| Gradient accumulation | `1` |
| Train steps | `50,000` |
| Save interval | `2,000` steps |
| Optimizer | AdamW over LoRA weights, `modules_to_save`, and AdaRMS dense trainables |
| LR / warmup | `5e-4` / `500` |
| Model max length | `200` |
| Action chunk | `10` |
| Runtime chunk | `chunk_size=10` for training and LoRA inference |

Run from inside the Docker container. PI05 LoRA SFT is supported only with DDP;
the entrypoint rejects DeepSpeed and FSDP backends. The reference LoRA recipe
uses 16 GPUs for global batch 64:

```bash
NPROC_PER_NODE=16

torchrun --nproc_per_node="${NPROC_PER_NODE}" \
  playground/benchmarks/libero/libero_pi05_lora.py \
  --task train \
  --train-backend ddp
```

The default output directory is:

```text
user_checkpoints/dexbotic/libero_all_pi05/pi05_lora_sft_libero_50k
```

The trainable summary is written to:

```text
user_checkpoints/dexbotic/libero_all_pi05/trainable_summaries/pi05_lora_sft_libero_50k.json
```

Check the summary before trusting a run:

- `r` should be `32`.
- `lora_alpha` should be `16`.
- `target_modules` should cover PI05 `llm` and `action_expert` projection
  layers.
- `extra_trainable_parameter_names` should include action expert AdaRMS dense
  parameters.
- `unexpected_trainable_parameters` should be empty.

LoRA checkpoints save the PEFT adapter files plus the extra full-rank state that
PEFT does not own:

```text
adapter_config.json
adapter_model.safetensors
extra_trainable_state.safetensors
extra_trainable_state.json
```

The extra trainable state contains the action expert AdaRMS dense parameters.
Keep these files together when copying a checkpoint.

For inference, pass a LoRA checkpoint path to `--model_name_or_path`. The loader
reads `adapter_config.json`, loads the recorded base model, restores
`extra_trainable_state.*`, sets `chunk_size=10`, and merges the adapter before
serving.

```bash
python playground/benchmarks/libero/libero_pi05_lora.py \
  --task inference \
  --model_name_or_path \
  user_checkpoints/dexbotic/libero_all_pi05/pi05_lora_sft_libero_50k/checkpoint-50000 \
  --port 7891
```

If the base model path recorded in `adapter_config.json` is not available on the
inference machine, pass `--base_model_name_or_path /path/to/Dexbotic-PI05` or
place the base model at the recorded path before starting inference.

## DM0 LoRA Recipe

| Item | Reference value |
| --- | --- |
| Base model | `checkpoints/DM0-base` |
| Dataset / augmentation | `libero_pi0_all` / `dm0,color_dm0,color_dm0` |
| Training backend | DDP only |
| LoRA target modules | `all-linear`, excluding `lm_head` |
| LoRA rank / alpha / dropout | `32` / `16` / `0.0` |
| Dense modules to save | `action_in_proj`, `action_out_proj`, `action_time_mlp_in`, `action_time_mlp_out` |
| Global batch | `64` on 8 GPUs |
| Per-device batch | `4` |
| Gradient accumulation | `2` |
| Train steps | `150,000` |
| Save interval / retention | `2,000` / `80` checkpoints |
| Optimizer | AdamW over LoRA weights and `modules_to_save` dense weights |
| LR / warmup | `5e-4` / `500` |
| Gradient checkpointing | disabled |
| Model max length | `200` |
| Action trajectory length | `50` |

Run from inside the Docker container. DM0 LoRA SFT is supported only with DDP;
the entrypoint rejects DeepSpeed and FSDP backends. The validated recipe uses 8
GPUs for global batch 64:

```bash
NPROC_PER_NODE=8

torchrun --nproc_per_node="${NPROC_PER_NODE}" \
  playground/benchmarks/libero/libero_dm0_lora.py \
  --task train \
  --train-backend ddp
```

The default output directory is:

```text
user_checkpoints/dexbotic/libero_all_dm0/dm0_lora_sft_libero_150k
```

The trainable summary is written to:

```text
user_checkpoints/dexbotic/libero_all_dm0/trainable_summaries/dm0_lora_sft_libero_150k.json
```

Check the summary before trusting a run:

- `r` should be `32` and `lora_alpha` should be `16`.
- `target_modules` should cover linear layers in the language, vision, action
  expert, and projector paths without including `lm_head`.
- `modules_to_save` should contain all four action projection/time MLP modules.
- `unexpected_trainable_parameters` should be empty.
- The trainable ratio should be approximately `2.14%` for the reference model.

For inference, pass a LoRA checkpoint path to `--model_name_or_path`. The loader
reads `adapter_config.json`, loads the recorded DM0 base model, and merges the
adapter before serving.

```bash
python playground/benchmarks/libero/libero_dm0_lora.py \
  --task inference \
  --model_name_or_path \
  user_checkpoints/dexbotic/libero_all_dm0/dm0_lora_sft_libero_150k/checkpoint-148000 \
  --port 7891
```

If the base model path recorded in `adapter_config.json` is not available on the
inference machine, pass `--base_model_name_or_path /path/to/DM0-base` or place
the base model at the recorded path before starting inference.

## CogACT LoRA Recipe

| Item | Reference value |
| --- | --- |
| Base model | `checkpoints/Dexbotic-Base` |
| Dataset | `libero_pi0_all` |
| Training backend | DDP only |
| LoRA target modules | Qwen q/k/v/o/gate/up/down and vision q/k/v projections |
| LoRA rank / alpha / dropout | `32` / `16` / `0.0` |
| Dense modules to save | `action_head` |
| Global batch | `64` on 16 GPUs |
| Per-device batch | `4` |
| Gradient accumulation | `1` |
| Train steps | `150,000` |
| Save interval | `2,000` steps |
| Optimizer | AdamW over LoRA weights and `action_head` |
| LR / warmup | `5e-4` / `500` |
| Adam beta2 / weight decay | `0.95` / `1e-10` |
| Model max length | `1024` |
| Action dimension / chunk | `7` / `16` |
| Augmentation | disabled |

Run from inside the Docker container. CogACT LoRA SFT is supported only with
DDP; the entrypoint rejects DeepSpeed and FSDP backends.

```bash
NPROC_PER_NODE=16

torchrun --nproc_per_node="${NPROC_PER_NODE}" \
  playground/benchmarks/libero/libero_cogact_lora.py \
  --task train \
  --train-backend ddp
```

The default output directory is:

```text
user_checkpoints/dexbotic/libero_all_cogact/cogact_lora_sft_libero_150k
```

The trainable summary is written to:

```text
user_checkpoints/dexbotic/libero_all_cogact/trainable_summaries/cogact_lora_sft_libero_150k.json
```

Check the summary before trusting a run:

- `r` should be `32` and `lora_alpha` should be `16`.
- `target_modules` should cover Qwen q/k/v/o/gate/up/down and vision q/k/v
  projections.
- `modules_to_save` should contain `action_head`.
- `unexpected_trainable_parameters` should be empty.

CogACT LoRA checkpoints keep the PEFT adapter and full-rank action head
together. Keep at least these files with the tokenizer and norm stats:

```text
adapter_config.json
adapter_model.safetensors
norm_stats.json
```

For inference, pass the adapter checkpoint and a reachable base model path. The
loader keeps the adapter unmerged, matching the validated evaluation path.

```bash
python playground/benchmarks/libero/libero_cogact_lora.py \
  --task inference \
  --model_name_or_path \
  user_checkpoints/dexbotic/libero_all_cogact/cogact_lora_sft_libero_150k/checkpoint-110000 \
  --base_model_name_or_path checkpoints/Dexbotic-Base \
  --port 7891
```

If `adapter_config.json` records a base model path that is reachable on the
inference machine, `--base_model_name_or_path` can be omitted.

## DM05 LoRA Recipe

`libero_dm05.py` is FSDP2 full SFT. `libero_dm05_lora.py` is DDP LoRA.

| Item | Reference value |
| --- | --- |
| Base model | `checkpoints/DM05` |
| Dataset | `libero_pi0_all` |
| Views / cameras | `2` (`agentview` + `wrist`) |
| Training backend | DDP (`deepspeed=None`) |
| Extra install | `pip install "peft>=0.13.0"` |
| LoRA target modules | `all-linear`, excluding `lm_head` |
| LoRA rank / alpha / dropout | `32` / `16` / `0.0` |
| Dense modules to save | `action_in_proj`, `action_out_proj`, `time_mlp_in`, `time_mlp_out`, `final_time_modulator`, plus `input_time_modulators.*` and `mlp_time_modulators.*` |
| Global batch | `32` on 8 GPUs |
| Per-device batch | `4` |
| Gradient accumulation | `1` |
| Train steps | `50,000` |
| Save interval | `10,000` steps |
| Optimizer | AdamW over LoRA weights and `modules_to_save` dense weights |
| LR / warmup | `5e-4` / `500` |
| LLM / vision / action attention | `eager` / `sdpa` / `sdpa` |
| VLM / AE gradient checkpointing | on |
| Model max length | `1024` |
| Action dimension / chunk | `7` valid (padded `32D`) / `10` absolute |
| HF sidecar | `checkpoint-{step}-hf` (`save_hf_sidecar=True`) |

Run from inside the Docker container. The playground default is DDP. On
multi-GPU 4090 set `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`.

```bash
NPROC_PER_NODE=8

# multi-GPU 4090 only
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

torchrun --nproc_per_node="${NPROC_PER_NODE}" \
  playground/benchmarks/libero/libero_dm05_lora.py \
  --task train \
  --train-backend ddp
```

The default output directory uses the launch date:

```text
user_checkpoints/dexbotic/libero_dm05_lora/libero-lora-MMDD
```

This entrypoint does not write a `trainable_summaries/*.json` file. Before
trusting a run, check `adapter_config.json` on a saved checkpoint:

- `r` should be `32` and `lora_alpha` should be `16`.
- `target_modules` should resolve from `all-linear` and must not include
  `lm_head`.
- `modules_to_save` should include the action/time projection modules listed
  above.

LoRA checkpoints keep the PEFT adapter, tokenizer/processor files, and
`norm_stats.json` together. Infer also accepts a `checkpoint-{step}-hf`
sidecar:

```text
adapter_config.json
adapter_model.safetensors
norm_stats.json
```

For inference, point `DM05InferenceConfig.model_name_or_path` at the run
directory or a `checkpoint-*-hf`, then:

```bash
python playground/benchmarks/libero/libero_dm05_lora.py --task inference
```

The loader reads `adapter_config.json`, loads the recorded base model, and
merges the adapter before serving. If that base path is missing on the
inference machine, place Dexmal/DM05 at `./checkpoints/DM05` or edit
`base_model_name_or_path` in `adapter_config.json`.

For a single-GPU smoke test, reduce `NPROC_PER_NODE` to `1` and edit the
playground file to lower `num_train_steps` and `save_steps`. Do not use a
smoke test to judge the final recipe quality.

# Dexbotic Inference API

Dexbotic inference servers expose the original `/process_frame` route and a
new v1 protocol for policy-based VLA/VLM inference.

The v1 protocol standardizes:

- action inference through `/v1/infer`
- episode reset through `/v1/reset`
- model capability discovery through `/v1/capabilities`
- optional chat-style generation through `/v1/chat/completions`

The legacy route remains supported for existing clients.

## Start an inference server

Each benchmark entry still owns model loading and checkpoint configuration.
For example:

```bash
python playground/benchmarks/libero/libero_dm0.py --task inference
```

Most project-specific launch scripts set the checkpoint path in Python before
calling `exp.inference()`:

```python
from playground.benchmarks.libero.libero_dm0 import DM0Exp

exp = DM0Exp()
exp.inference_config.model_name_or_path = "/path/to/checkpoint"
exp.inference_config.port = 7891
exp.inference()
```

The server listens on `0.0.0.0:<port>` and registers both legacy and v1 routes.

## DM0 realtime inference

DM0 also has an optional realtime inference backend for latency-sensitive
serving. The realtime path keeps the same `/v1/infer` API and policy contract,
but replaces the core DM0 action-generation call with a Triton-backed optimized
runtime. The legacy DM0 Python path remains available and is still the default
unless the realtime entry/config is selected.

Typical realtime launch:

```bash
python playground/benchmarks/libero/libero_dm0_realtime.py --task inference
```

The realtime backend is intended to preserve DM0's v1 inference semantics:
same request schema, same action denormalization path, and the same absolute
action contract exposed by `/v1/capabilities`. It should be evaluated with the
same checkpoint, camera setup, normalization stats, and benchmark configuration
as the non-realtime path.

Measured on libero DM0 checkpoint, v1 API, and `libero_goal` probe benchmark:

| Backend | Core inference mean | Core inference median |
| --- | ---: | ---: |
| DM0 realtime | 100.689 ms | 100.549 ms |
| DM0 non-realtime | 554.053 ms | 550.889 ms |

This corresponds to:

- `5.50x` mean speedup for the core model call.
- `5.48x` median speedup for the core model call.

The core inference timing wraps only the server-side model call
(`realtime_model.forward(...)` vs. `model.inference_action(**inputs)`) with CUDA
synchronization. It excludes HTTP transport, request decoding, image
preprocessing, tokenization, input/output transforms, action denormalization,
and environment stepping.

Reference project: [realtime-vla](https://github.com/dexmal/realtime-vla).

## Routes

### `GET /health`

Lightweight readiness check.

```bash
curl http://localhost:7891/health
```

Response:

```json
{"status": "ok"}
```

### `GET /v1/capabilities`

Returns the model's declared inference contract. Call this before wiring a new
environment or client.

```bash
curl http://localhost:7891/v1/capabilities
```

Example response:

```json
{
  "model_family": "DM0InferenceConfig",
  "vla": true,
  "vlm": false,
  "modalities": {
    "images": {
      "format": "image/{slot_index}",
      "slots": [
        {"slot": 1, "name": "front", "required": true},
        {"slot": 2, "name": "left_wrist", "required": true},
        {"slot": 3, "name": "right_wrist", "required": false}
      ]
    },
    "state": {"used": false, "required": false, "dim": null},
    "prompt": {"required": true}
  },
  "action_spec": {
    "action_dim": 7,
    "chunk_size": null,
    "action_mode": "absolute"
  },
  "max_batch_size": 1,
  "sampling_defaults": {"num_steps": 10, "cfg_scale": 1.0}
}
```

Important fields:

- `modalities.images.slots`: image slots expected by the policy. Slot names
  come from `camera_order`; `null` slots are zero-padded by the model wrapper.
- `modalities.state`: whether proprio/state is used or required.
- `action_spec.action_mode`: whether returned actions are already absolute or
  should be interpreted as relative/delta actions by the caller.
- `vla` and `vlm`: whether the server supports action inference and text
  generation.

### `POST /v1/infer`

Runs one policy inference request.

Request schema:

```json
{
  "observation": {
    "prompt": "pick up a cube and move it to the green point",
    "images": {
      "1": "<front camera base64 encoded image>",
      "2": "<left wrist camera base64 encoded image>",
      "3": "<right wrist camera base64 encoded image>"
    },
    "state": [0.0, 0.0, 0.0]
  },
  "sampling": {
    "num_steps": 10,
    "cfg_scale": 1.5,
    "seed": 42
  }
}
```

Response schema:

```json
{
  "actions": [
    [0.01, 0.02, 0.03, 0.0, 0.0, 0.0, 1.0]
  ],
  "metadata": {
    "latency_ms": 58.2
  }
}
```

Notes:

- `images` keys must be 1-based numeric strings: `"1"`, `"2"`, ...
- Images are base64-encoded PNG/JPEG bytes.
- `state` is optional unless `/v1/capabilities` says it is required. When
  required, the HTTP layer only checks that the `state` key is present; concrete
  policies decide how to consume or validate its value.
- `sampling` is optional. Supported fields are `num_steps`, `cfg_scale`, and
  `seed`; any other fields are ignored before dispatching to the policy.
- DM0 realtime captures a fixed-step CUDA graph at service startup. For that
  backend, request `sampling.num_steps` must match `/v1/capabilities`.
- The response always uses `actions`; legacy `/process_frame` uses `response`.

### `POST /v1/reset`

Marks an episode boundary.

```bash
curl -X POST http://localhost:7891/v1/reset
```

Response:

```json
{"status": "ok"}
```

Most current policies are stateless and treat reset as a no-op. It matters for
stateful policies such as memory-based models that keep cross-step or
cross-episode context. Evaluators can safely call it at the beginning of every
episode.

### `POST /v1/chat/completions`

OpenAI-style chat endpoint for models whose policy declares VLM generation
support. VLA-only models return `501`.

```json
{
  "model": "checkpoint-name",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe the image briefly."},
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,<base64 encoded image>"
          }
        }
      ]
    }
  ],
  "temperature": 0,
  "max_tokens": 16
}
```

For image-conditioned VLM generation, send each image as an OpenAI-compatible
`image_url` content part. The current server parser accepts data URLs such as
`data:image/png;base64,...` and combines them with the text parts before calling
`policy.generate()`. Pure text `content` is also accepted, but questions about
the visual scene should include at least one image.

## Python client

Use `DexClient` for both legacy and v1 inference.

```python
import cv2
from dexbotic.client import DexClient

client = DexClient(
    base_url="http://localhost:7891",
    api_style="v1",
    use_delta=False,
    sampling={"num_steps": 10, "cfg_scale": 1.5},
)

obs = {
    "image": [
        cv2.cvtColor(cv2.imread("front.png"), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.imread("left_wrist.png"), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.imread("right_wrist.png"), cv2.COLOR_BGR2RGB),
    ],
    "state": [0.0] * 7,
}

client.reset()
action = client.act(obs, "put the moka pot on the stove")
```

For legacy clients:

```python
client = DexClient(base_url="http://localhost:7891", api_style="legacy")
```

`use_delta` is a client-side post-processing option. If it is true, the client
accumulates returned actions as deltas against the previous action. Choose this
based on the environment's control mode and the model's `action_mode`.

## Direct HTTP example

```python
import base64
import requests

with open("front.png", "rb") as f:
    image_1 = base64.b64encode(f.read()).decode("utf-8")
with open("left_wrist.png", "rb") as f:
    image_2 = base64.b64encode(f.read()).decode("utf-8")
with open("right_wrist.png", "rb") as f:
    image_3 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "observation": {
        "prompt": "pick up the cube",
        "images": {
            "1": image_1,
            "2": image_2,
            "3": image_3,
        },
        "state": [0.0] * 7,
    }
}

resp = requests.post("http://localhost:7891/v1/infer", json=payload, timeout=30)
resp.raise_for_status()
print(resp.json()["actions"])
```

## Benchmark configuration

Dexbotic benchmark configs select the protocol with `api_style`.

```yaml
base_url: "http://host:7891"
api_style: v1
```

Use `api_style: legacy` to keep the old `/process_frame` route.

Some environments also need action semantics configured. For example,
ManiSkill2 uses delta end-effector control and its VLA agent requires:

```yaml
use_delta: true
```

Always check `/v1/capabilities` for `action_mode` and `state.required` before
reusing a checkpoint in a different benchmark.

## Implementing v1 for a model

To expose `/v1/infer`, an inference config must build a policy:

```python
class MyInferenceConfig(InferenceConfig):
    def _build_policy(self):
        return MyPolicy(
            model=self.model,
            tokenizer=self.tokenizer,
            norm_stats=self.norm_stats,
            camera_order=self.camera_order,
        )
```

The policy should subclass `BasePolicy` and implement `select_action()`:

```python
from dexbotic.policy.base_policy import BasePolicy
from dexbotic.policy.types import ActionOutput

class MyPolicy(BasePolicy):
    action_mode = "relative"
    state_used = False
    state_required = False

    def select_action(self, observation, sampling_config=None):
        # observation contains prompt, internal image/0, image/1, ..., and optional state.
        # HTTP image keys are public 1-based slots: "1", "2", ...
        actions = ...
        return [ActionOutput(actions=actions)]
```

Policy declarations drive `/v1/capabilities`:

- `action_mode`: `absolute`, `relative`, or `unknown`
- `state_used`: whether the model consumes state
- `state_required`: whether requests must include a `state` key
- `state_dim`: expected state dimension if known
- `max_batch_size`: maximum supported batch size
- `supports_vlm()`: whether `/v1/chat/completions` is available

If a model has episode-level memory, override `reset()`.

## Model notes

Current policy wrappers cover the main VLA inference paths used by the
benchmark integration. The table below shows the default declarations made by
the current serving wrappers; these values are not intrinsic properties of a
model architecture.

| Model family | Default v1 action mode | State |
| --- | --- | --- |
| Pi0 / Pi05-style policy | absolute | optional, used when provided |
| DM0 | absolute | not required by current policy |
| OFT / OFT-discrete | relative | not required by current policy |
| CogACT | relative | not required |
| DiscreteVLA / GR00T-N1 | relative | not required |
| MemVLA | relative | reset is meaningful for memory state |

`action_mode` is a serving contract: it declares how the current policy wrapper
interprets and returns actions after model-specific post-processing. It is not
read automatically from the checkpoint. OFT and CogACT default to `relative`
because the existing legacy inference paths and benchmark adapters have used
their outputs with relative/delta semantics. If a checkpoint was trained and
post-processed to produce absolute actions, the policy declaration or benchmark
configuration must be changed accordingly.

A checkpoint still needs to match the environment it is evaluated on: camera
setup, action space, normalization/denormalization statistics, gripper
convention, and control mode are checkpoint- and benchmark-specific.

Navigation-oriented models such as NaviLA, MuVLA, and UniNaVid use different
interaction patterns and were not folded into this VLA policy interface in this
round. They should be integrated with a separate policy contract when their
inputs, memory reset semantics, and output actions are standardized.

## Compatibility

- `/process_frame` is unchanged for existing clients.
- v1 uses JSON and base64 images instead of multipart form data.
- The server runs Flask with `threaded=False`; concurrent benchmark jobs are
  serialized by the service process.
- `seed` in `sampling` sets Python, NumPy, and Torch random seeds for that
  request. Avoid relying on it for concurrent multi-threaded serving.

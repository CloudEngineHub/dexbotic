# DM0 Realtime Inference

Updated: 2026-06-18

DM0 has an optional realtime inference backend for latency-sensitive serving.
It keeps the same `/v1/infer` API and policy contract described in
[Dexbotic Inference API](InferenceAPI.md), but replaces the core DM0
action-generation call with a Triton-backed optimized runtime. On the libero
DM0 probe benchmark, this path provides about `5x` core inference speedup over
the non-realtime backend.

The legacy DM0 Python path remains available and is still the default unless
the realtime entry/config is selected.

## Launch

Typical realtime launch:

```bash
python playground/benchmarks/libero/libero_dm0_realtime.py --task inference
```

The realtime backend should be evaluated with the same checkpoint, camera
setup, normalization stats, and benchmark configuration as the non-realtime
path.

## API contract

The realtime backend is intended to preserve DM0's v1 inference semantics:
same request schema, same action denormalization path, and the same absolute
action contract exposed by `/v1/capabilities`.

DM0 realtime captures a fixed-step CUDA graph at service startup. For that
backend, request `sampling.num_steps` must match `/v1/capabilities`.

For shared routes, request/response schemas, Python client usage, direct HTTP
examples, and benchmark client configuration, see
[Dexbotic Inference API](InferenceAPI.md).

## Benchmark result

Measured with libero DM0 checkpoint, v1 API, and `libero_goal`
probe benchmark:

| Backend | Core inference mean | Core inference median |
| --- | ---: | ---: |
| DM0 realtime | 100.689 ms | 100.549 ms |
| DM0 non-realtime | 554.053 ms | 550.889 ms |

This corresponds to:

- `5.50x` mean speedup for the core model call.
- `5.48x` median speedup for the core model call.

## Timing scope

The core inference timing wraps only the server-side model call
(`realtime_model.forward(...)` vs. `model.inference_action(**inputs)`) with CUDA
synchronization. It excludes HTTP transport, request decoding, image
preprocessing, tokenization, input/output transforms, action denormalization,
and environment stepping.

## Reference

Reference project: [realtime-vla](https://github.com/dexmal/realtime-vla).

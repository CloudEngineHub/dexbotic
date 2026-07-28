"""Bridge gear_sonic's Isaac-GR00T ZMQ client to the dexbotic native gr00tsonic policy.

Run this in the dexbotic environment.  The robot-side
``gear_sonic/scripts/run_vla_inference.py`` can keep using Isaac's
``PolicyClient(host, port)``; this bridge speaks that ZMQ protocol and calls
``Gr00tSonicPolicy`` through dexbotic's native inference config (GR00T N1.7 /
Qwen3-VL backbone + flow-matching DiT action head).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image

from dexbotic.exp.gr00tsonic_exp import DEFAULT_COSMOS, Gr00tSonicInferenceConfig
from dexbotic.policy.types import SamplingConfig

MOTION_TOKEN_DIM = 64
HAND_JOINT_DIM = 7
ACTION_DIM = MOTION_TOKEN_DIM + HAND_JOINT_DIM * 2  # 78, == SONIC_VALID_ACTION_DIM
STATE_KEY_ORDER = (
    "left_leg", "right_leg", "waist",
    "left_arm", "right_arm", "left_hand", "right_hand",
)
LANGUAGE_KEY = "annotation.human.task_description"

DEFAULT_CKPT = "./checkpoints/Dexbotic-GR00TSonic"

# observation["video"] camera keys per view COUNT, in the SAME view order used at
# training (images_1, images_2, ...). Must match how the robot client names the
# views in its observation dict.
VIEW_KEYS_BY_COUNT = {
    1: ["ego_view"],
    2: ["ego_view_left", "ego_view_right"],
}


def resolve_view_keys(cameras_arg: str) -> list[str]:
    """Resolve --cameras: a count (int -> default keys) or explicit name list."""
    val = str(cameras_arg).strip()
    if val.isdigit():
        n = int(val)
        if n not in VIEW_KEYS_BY_COUNT:
            raise SystemExit(
                f"--cameras {n}: no default view keys for {n} views. "
                f"Supported counts: {sorted(VIEW_KEYS_BY_COUNT)}; "
                f"or pass explicit comma-separated observation['video'] keys."
            )
        return list(VIEW_KEYS_BY_COUNT[n])
    return [c.strip() for c in val.split(",") if c.strip()]


def _wire_deps():
    try:
        import zmq
        import msgpack_numpy as mnp
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing bridge wire-protocol dependency. Install inside conda dexbotic: "
            "pip install pyzmq msgpack msgpack-numpy"
        ) from exc
    return zmq, mnp


class MsgSerializer:
    @staticmethod
    def to_bytes(data) -> bytes:
        _, mnp = _wire_deps()
        return mnp.packb(data, default=mnp.encode)

    @staticmethod
    def from_bytes(data: bytes):
        _, mnp = _wire_deps()
        return mnp.unpackb(data, object_hook=mnp.decode, raw=False)


def extract_views(observation: dict, view_keys: list[str]) -> list[np.ndarray]:
    """Read each requested camera from observation['video'] as a uint8 HxWx3 array."""
    views = []
    for key in view_keys:
        image = np.asarray(observation["video"][key])
        while image.ndim > 3:
            image = image[0]
        views.append(image.astype(np.uint8))
    return views


def extract_state(observation: dict) -> np.ndarray:
    state = observation["state"]
    parts = [
        np.asarray(state[key], dtype=np.float32).reshape(-1)
        for key in STATE_KEY_ORDER
    ]
    parts.append(np.asarray(state["projected_gravity"], dtype=np.float32).reshape(-1))
    return np.concatenate(parts, axis=0)


def extract_prompt(observation: dict) -> str:
    try:
        return observation["language"][LANGUAGE_KEY][0][0]
    except (KeyError, IndexError, TypeError):
        return ""


def _sampling_from_options(options: dict | None) -> SamplingConfig:
    # gr00tsonic uses flow-matching with a fixed number of denoising steps from
    # the model config; cfg_scale/num_steps are accepted for protocol compat but
    # the policy ignores the sampling config (kept for a uniform interface).
    options = options or {}
    seed = options.get("seed")
    num_steps = options.get("num_steps", options.get("num_ddim_steps", 4))
    cfg_scale = options.get("cfg_scale", 1.0)
    return SamplingConfig(num_steps=int(num_steps), cfg_scale=float(cfg_scale), seed=seed)


class DexboticSonicBridge:
    """ZMQ bridge that delegates inference to dexbotic's native gr00tsonic policy."""

    def __init__(
        self,
        model_path: str,
        norm_stats_path: str | None,
        cosmos_model_name: str = DEFAULT_COSMOS,
        num_steps: int | None = None,
        view_keys: list[str] | None = None,
    ) -> None:
        self.view_keys = view_keys or list(VIEW_KEYS_BY_COUNT[1])
        inference_config = Gr00tSonicInferenceConfig(
            model_name_or_path=model_path,
            norm_stats=norm_stats_path,
            cosmos_model_name=cosmos_model_name,
            # One policy slot per view, in order — the policy reads image/0..N-1.
            camera_order=list(self.view_keys),
        )
        inference_config._initialize_inference()
        # Optionally override the flow-matching denoising steps.
        if num_steps is not None:
            inference_config.model.model.action_head.num_inference_timesteps = int(num_steps)
        self.inference_config = inference_config
        self.policy = inference_config.policy

    def get_action(self, observation: dict, options: dict | None = None):
        views = extract_views(observation, self.view_keys)
        state = extract_state(observation)
        prompt = extract_prompt(observation)

        policy_obs = {"prompt": prompt, "state": state}
        for i, v in enumerate(views):
            policy_obs[f"image/{i}"] = Image.fromarray(v, mode="RGB")
        sampling = _sampling_from_options(options)
        actions = self.policy.select_action(policy_obs, sampling)[0].actions
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 3:
            actions = actions[0]
        if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
            raise RuntimeError(
                f"Expected SONIC actions with shape [T, {ACTION_DIM}], got {actions.shape}."
            )

        motion_token = actions[:, :MOTION_TOKEN_DIM][None]
        left_hand = actions[
            :, MOTION_TOKEN_DIM:MOTION_TOKEN_DIM + HAND_JOINT_DIM
        ][None]
        right_hand = actions[:, MOTION_TOKEN_DIM + HAND_JOINT_DIM:][None]
        action = {
            "motion_token": motion_token.astype(np.float32),
            "left_hand_joints": left_hand.astype(np.float32),
            "right_hand_joints": right_hand.astype(np.float32),
        }
        return [action, {"backend": "dexbotic_native"}]


def serve(bridge: DexboticSonicBridge, host: str, port: int) -> None:
    zmq, _ = _wire_deps()
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind(f"tcp://{host}:{port}")
    print(f"dexbotic SONIC bridge listening on tcp://{host}:{port}")
    try:
        while True:
            should_stop = False
            msg = socket.recv()
            try:
                req = MsgSerializer.from_bytes(msg)
                endpoint = req.get("endpoint")
                data = req.get("data", {}) or {}
                if endpoint == "ping":
                    resp = {"status": "ok", "message": "dexbotic sonic bridge"}
                elif endpoint == "kill":
                    resp = {"status": "ok"}
                    should_stop = True
                elif endpoint == "reset":
                    bridge.policy.reset()
                    resp = {}
                elif endpoint == "get_action":
                    resp = bridge.get_action(data["observation"], data.get("options"))
                elif endpoint == "get_modality_config":
                    resp = {}
                else:
                    resp = {"error": f"unknown endpoint: {endpoint}"}
            except Exception as exc:
                traceback.print_exc()
                resp = {"error": str(exc)}
            socket.send(MsgSerializer.to_bytes(resp))
            if should_stop:
                break
    except KeyboardInterrupt:
        print("Bridge shutting down.")
    finally:
        socket.close()
        ctx.term()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_CKPT,
                        help="finetuned gr00tsonic checkpoint dir")
    parser.add_argument("--norm-stats", default=None,
                        help="norm_stats.json; defaults to <model-path>/norm_stats.json")
    parser.add_argument("--cosmos-model-name", default=DEFAULT_COSMOS,
                        help="Qwen3-VL processor source (Cosmos-Reason2-2B)")
    parser.add_argument("--host", default="*")
    parser.add_argument("--port", type=int, default=7899)
    parser.add_argument("--num-steps", type=int, default=None,
                        help="override flow-matching denoising steps")
    parser.add_argument("--cameras", default="1",
                        help="View COUNT (1=ego_view, 2=ego_view_left+right) OR an "
                             "explicit comma-separated list of observation['video'] "
                             "keys, in the SAME view order used at training. Must "
                             "match the trained model's num_images.")
    args = parser.parse_args()

    _wire_deps()
    bridge = DexboticSonicBridge(
        model_path=args.model_path,
        norm_stats_path=args.norm_stats,
        cosmos_model_name=args.cosmos_model_name,
        num_steps=args.num_steps,
        view_keys=resolve_view_keys(args.cameras),
    )
    serve(bridge, args.host, args.port)


if __name__ == "__main__":
    main()

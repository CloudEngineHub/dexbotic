import base64
import math
from collections import deque

import cv2
import numpy as np
import requests


class DexClient:
    """
    HTTP client for the Dexbotic inference server.

    api_style="legacy"  → multipart POST to /process_frame  (original behavior)
    api_style="v1"      → JSON POST to /v1/infer, /v1/reset (new v1 protocol)
    """

    def __init__(self, base_url, use_delta=True, api_style="legacy", sampling=None):
        self.base_url = base_url
        self.use_delta = use_delta
        self.api_style = api_style
        self.sampling = dict(sampling or {})

        self.set_init_action()
        self.action_queue = deque()

    def set_init_action(self, action=None):
        self.last_act = action if action is not None else [0, 0, 0, 0, 0, 0, 0]

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset episode state. In v1 mode also notifies the server."""
        self.set_init_action()
        self.action_queue.clear()
        if self.api_style == "v1":
            try:
                requests.post(self.base_url + "/v1/reset", timeout=5)
            except Exception:
                pass

    def act(self, observation, prompt):
        if len(self.action_queue) == 0:
            self.acquire_new_action(observation, prompt)
        action = self.action_queue.popleft()
        self.last_act = action
        return action

    def acquire_new_action(self, observation, prompt):
        if self.api_style == "v1":
            raw_actions = self._infer_v1(observation, prompt)
        else:
            raw_actions = self._infer_legacy(observation, prompt)

        last_act = self.last_act
        for action in raw_actions:
            if self.use_delta:
                action = self._delta_action(last_act, action)
            else:
                action = np.copy(action)
            self.action_queue.append(action)
            last_act = action

    # ── legacy path ───────────────────────────────────────────────────────────

    def _infer_legacy(self, observation, prompt) -> list:
        images = [observation["image"]] if not isinstance(observation["image"], list) else observation["image"]
        encoded = []
        for img in images:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode(".png", img_bgr)
            encoded.append(buf.tobytes())

        ret = requests.post(
            self.base_url + "/process_frame",
            data={"text": prompt, **self._legacy_sampling_payload()},
            files=[("image", b) for b in encoded],
        )
        ret.raise_for_status()
        return ret.json()["response"]

    # ── v1 path ───────────────────────────────────────────────────────────────

    def _infer_v1(self, observation, prompt) -> list:
        obs_payload = {"prompt": prompt, "images": {}}

        # observation["image"] may be a single frame or a list of frames
        images = observation["image"] if isinstance(observation["image"], list) else [observation["image"]]
        for idx, img in enumerate(images):
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode(".png", img_bgr)
            obs_payload["images"][str(idx + 1)] = base64.b64encode(buf.tobytes()).decode()

        state = observation.get("state")
        if state is not None:
            obs_payload["state"] = np.array(state).tolist()

        ret = requests.post(
            self.base_url + "/v1/infer",
            json={"observation": obs_payload, "sampling": self.sampling},
            timeout=30,
        )
        ret.raise_for_status()
        return ret.json()["actions"]

    # ── helpers ───────────────────────────────────────────────────────────────

    def _delta_action(self, last_action, delta_action):
        action = np.copy(last_action)
        action[6:] = 0
        action = action + delta_action
        action[3:6] = np.where(action[3:6] > math.pi, action[3:6] - 2 * math.pi, action[3:6])
        action[3:6] = np.where(action[3:6] < -math.pi, action[3:6] + 2 * math.pi, action[3:6])
        return action

    def _legacy_sampling_payload(self) -> dict:
        payload = {}
        if "seed" in self.sampling and self.sampling["seed"] is not None:
            payload["seed"] = str(self.sampling["seed"])
        return payload

    # keep old name for backward compat
    def delta_action(self, last_action, delta_action):
        return self._delta_action(last_action, delta_action)


if __name__ == "__main__":
    client = DexClient(base_url="http://localhost:7891", api_style="v1")
    observation = {"image": cv2.imread("test_data/libero_test.png")}
    for i in range(5):
        action = client.act(observation, "put the moka pot on the stove")
        print("Action:", action)

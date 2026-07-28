# SPDX-License-Identifier: Apache-2.0
#
# Data-pipeline glue that lets GR00T N1.7 ("gr00tsonic") train through Dexbotic's
# DexDataset. The model keeps the upstream monolithic Qwen3-VL backbone, which
# needs the images and the prompt to be processed *jointly* by Qwen3VLProcessor
# (so the image-placeholder tokens match image_grid_thw). DexDataset, however,
# processes images and text separately per sample.
#
# We bridge the two by:
#   * image_process_func (Gr00tSonicImagePreprocess): resize each view to a uint8
#     CHW tensor that survives DexDataset's ToTensor().
#   * tokenization_func (Gr00tSonicTokenization): stash the raw prompt as a tensor
#     of UTF-8 bytes under "input_ids" (DexDataset requires input_ids/labels).
#   * data collator (Gr00tSonicDataCollator): decode prompts, rebuild PIL images,
#     and run Qwen3VLProcessor on the whole batch to emit the real
#     input_ids / attention_mask / pixel_values / image_grid_thw, then stack the
#     state / action / action_mask / embodiment_id action-head inputs.

import copy
from functools import lru_cache
import random
import re
import warnings

import numpy as np
from PIL import Image
import torch

from dexbotic.data.dataset.dex_dataset import DexDataset, load_jsonl
from dexbotic.data.dataset.transform.common import ToTensor
from dexbotic.data.dataset.transform.language import (
    ToConversation_Old,
    ToConversationWithDiscreteState,
)


def _shallow_copy_record(rec: dict) -> dict:
    """Copy a record + its nested dicts one level (cheap vs deepcopy).

    Only the per-frame image/video dicts (e.g. ``images_1``) are mutated by the
    pipeline (LoadMultiModal sets ``frame['data']``); state/action lists stay
    read-only, so a shallow copy that re-copies nested dicts is enough to keep the
    cached episode pristine.
    """
    out = dict(rec)
    for key, value in rec.items():
        if isinstance(value, dict):
            out[key] = dict(value)
    return out


def _stabilize_normalized_action(action, action_process_func) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    statistic_mapping = getattr(action_process_func, "statistic_mapping", None)
    action_stats = (
        statistic_mapping.get("action")
        if isinstance(statistic_mapping, dict)
        else None
    )

    if action_stats is not None and "min" in action_stats and "max" in action_stats:
        min_vals = np.asarray(action_stats["min"], dtype=np.float32)
        max_vals = np.asarray(action_stats["max"], dtype=np.float32)
        if min_vals.ndim == 1 and max_vals.ndim == 1 and action.ndim >= 1:
            dim = min(action.shape[-1], min_vals.shape[0], max_vals.shape[0])
            constant_dims = np.isclose(max_vals[:dim], min_vals[:dim])
            if np.any(constant_dims):
                action = action.copy()
                action[..., np.nonzero(constant_dims)[0]] = 0.0

    return np.clip(action, -1.0, 1.0).astype(np.float32)


@lru_cache(maxsize=1024)
def _load_episode_cached(jsonl_file: str):
    """Per-worker cache of parsed episodes.

    ``load_jsonl`` re-reads and json-parses the whole file (~34 ms for 1405 lines)
    on every sample. Episodes are immutable, so we cache the parsed list (the
    caller deep-copies the window before mutating). Mirrors the original gr00t
    loader keeping each episode's DataFrame in memory.
    """
    return load_jsonl(jsonl_file, parse=True)

try:
    from transformers import Qwen3VLProcessor
except ImportError:  # pragma: no cover - needs transformers>=4.57
    Qwen3VLProcessor = None


class Gr00tSonicDexDataset(DexDataset):
    """DexDataset variant with step-level (windowed) access.

    The base ``DexDataset.unsafe_getitem`` runs ``action_process_func`` over the
    WHOLE episode (e.g. 1405 frames → build a 40-step trajectory for every frame)
    and then keeps a single frame — O(episode_length) per sample (~139 ms here).

    The upstream Isaac-GR00T loader instead gathers only ``step_index +
    delta_indices`` rows (the action-horizon window) per sample — O(action_horizon).
    This subclass does the same: it slices the episode to a ``window_size`` window
    starting at the sampled frame and processes only that, so cost is independent
    of episode length (~ a few ms). The output is identical to the base class
    (``AddTrajectory(padding_action=True)`` pads the tail near the episode end).
    """

    def __init__(self, *args, window_size: int = 40, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = window_size

    def unsafe_getitem(self, idx) -> dict:
        dataset_index, file_index, frame_index = self.global_index[idx]
        jsonl_file = self.file_name_map[file_index]
        dataset_info = self.dataset_map[dataset_index]
        dataset = dataset_info["data_path"]
        data_path_prefix = dataset_info["data_path_prefix"]
        episode_data_list = _load_episode_cached(jsonl_file)
        valid_state_dim = self.get_valid_state_dim(episode_data_list)

        n = len(episode_data_list)
        if frame_index >= n:
            frame_index = random.randint(0, n - 1)

        # Step-level slice: only the [frame_index, frame_index + window_size) rows
        # are processed. window_size must be >= action_horizon so window[0]'s
        # action chunk is fully present; the tail is padded by AddTrajectory.
        # Cheap copy (NOT deepcopy): only the per-frame image dicts get mutated by
        # LoadMultiModal (it sets frame['data']); state/action lists are read-only
        # in the pipeline (ToNumpy builds new arrays). So shallow-copy each record
        # and re-copy its nested dicts to keep the cached episode pristine.
        window = [
            _shallow_copy_record(rec)
            for rec in episode_data_list[frame_index : frame_index + self.window_size]
        ]
        local_index = 0

        # deepcopy so we never mutate the shared dataset_info["meta_data"].
        meta_data = copy.deepcopy(dataset_info["meta_data"])
        meta_data.update(
            dict(
                fram_indicies=[local_index],
                jsonl_file=jsonl_file,
                dataset=dataset,
                num_images=self.num_images,
                images_keys=self.images_keys,
                depths_keys=self.depths_keys,
                load_depth=self.load_depth,
                data_path_prefix=data_path_prefix,
            )
        )

        # 1. process only the window
        data = self.action_process_func(window, meta_data=meta_data)
        # 2. take the first (== sampled) frame
        if isinstance(data, list):
            data = data[local_index]
        if "action" in data:
            data["action"] = _stabilize_normalized_action(
                data["action"], self.action_process_func
            )
        data.update({"meta_data": meta_data})
        return_dict = {}

        # 3. preprocess rgb
        rgb_data = data.pop("rgb_data", [])
        if len(rgb_data) < self.num_images:
            warnings.warn(
                "The length of rgb_data is less than num_images, padding with None"
            )
            rgb_data = rgb_data + [None] * (self.num_images - len(rgb_data))
        pixel_values = [
            image_process_func(d)
            for image_process_func, d in zip(self.image_process_func, rgb_data, strict=True)
        ]
        return_dict["image"] = (
            pixel_values[0] if len(pixel_values) == 1 else torch.stack(pixel_values, dim=0)
        )

        # 4. tokenize the prompt
        if "conversations" not in data:
            if self.discrete_state_input:
                data = ToConversationWithDiscreteState(valid_state_dim)(data)
            else:
                data = ToConversation_Old()(data)
        tokenized_dict = self.tokenization_func(
            conversations=data["conversations"], has_image=True
        )
        return_dict["input_ids"] = tokenized_dict["input_ids"]
        return_dict["labels"] = tokenized_dict["labels"]

        # 5. extract other data and convert to tensor
        other_keys = [k for k in self.data_keys if k not in return_dict]
        return_dict.update(self.key_extract_func(data, other_keys))
        return_dict = ToTensor()(return_dict)
        return return_dict


class Gr00tSonicImagePreprocess:
    """Resize one view to a uint8 CHW tensor (PIL is rebuilt in the collator)."""

    def __init__(self, target_size=(256, 256)):
        # target_size is (W, H) for PIL.resize.
        self.target_size = tuple(target_size)

    def __call__(self, image, **kwargs):
        if image is None:
            arr = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
            return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = image.convert("RGB").resize(self.target_size)
        # np.array (not asarray) copies, so the tensor backing store is writable.
        arr = np.array(image, dtype=np.uint8)  # HWC
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW uint8


class Gr00tSonicTokenization:
    """Stash the human prompt as UTF-8 bytes under input_ids (decoded in collator)."""

    def __call__(self, conversations, has_image: bool = True, **kwargs):
        prompt = ""
        for turn in conversations:
            if turn.get("from") == "human":
                prompt = turn.get("value", "") or ""
                break
        data = prompt.encode("utf-8")
        if len(data) == 0:
            input_ids = torch.zeros(1, dtype=torch.long)
        else:
            input_ids = torch.tensor(list(data), dtype=torch.long)
        return {"input_ids": input_ids, "labels": torch.zeros(1, dtype=torch.long)}


class Gr00tSonicDataCollator:
    """Collate DexDataset samples into Qwen3-VL + action-head model inputs."""

    def __init__(
        self,
        processor=None,
        model_name: str = "nvidia/Cosmos-Reason2-2B",
        max_state_dim: int = 132,
        max_action_dim: int = 132,
        action_horizon: int = 40,
        state_history_length: int = 1,
        valid_action_dim: int = 78,
        embodiment_id: int = 11,
        formalize_language: bool = True,
    ):
        if processor is not None:
            self.processor = processor
        else:
            if Qwen3VLProcessor is None:
                raise ImportError("Qwen3VLProcessor requires transformers>=4.57.")
            self.processor = Qwen3VLProcessor.from_pretrained(model_name)
        self.processor.tokenizer.padding_side = "left"
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.action_horizon = action_horizon
        self.state_history_length = state_history_length
        self.valid_action_dim = valid_action_dim
        self.embodiment_id = embodiment_id
        self.formalize_language = formalize_language

    @staticmethod
    def _decode_prompt(input_ids: torch.Tensor) -> str:
        vals = [int(x) for x in input_ids.tolist()]
        if not any(vals):
            return ""
        return bytes(vals).decode("utf-8", errors="ignore")

    def _to_pil_views(self, image: torch.Tensor) -> list[Image.Image]:
        if image.ndim == 3:
            image = image[None]
        views = []
        for v in image:
            arr = v.permute(1, 2, 0).to(torch.uint8).cpu().numpy()
            views.append(Image.fromarray(arr))
        return views

    def __call__(self, instances: list[dict]) -> dict:
        texts, all_images = [], []
        for inst in instances:
            pil_views = self._to_pil_views(inst["image"])
            prompt = self._decode_prompt(inst["input_ids"])
            if self.formalize_language:
                prompt = re.sub(r"[^\w\s]", "", prompt.lower())
            conversation = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": im} for im in pil_views],
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            all_images.extend(pil_views)

        vlm = self.processor(
            text=texts, images=all_images, return_tensors="pt", padding=True
        )

        B = len(instances)
        state = torch.stack([inst["state"].float() for inst in instances])  # [B, Ds]
        state = state[:, None, :].repeat(1, self.state_history_length, 1)
        action = torch.stack([inst["action"].float() for inst in instances])  # [B, T, Da]

        action_mask = torch.ones(B, self.action_horizon, self.max_action_dim)
        action_mask[:, :, self.valid_action_dim :] = 0.0
        embodiment_id = torch.full((B,), self.embodiment_id, dtype=torch.long)

        return {
            "input_ids": vlm["input_ids"],
            "attention_mask": vlm["attention_mask"],
            "pixel_values": vlm["pixel_values"],
            "image_grid_thw": vlm["image_grid_thw"],
            "state": state,
            "action": action,
            "action_mask": action_mask,
            "embodiment_id": embodiment_id,
        }

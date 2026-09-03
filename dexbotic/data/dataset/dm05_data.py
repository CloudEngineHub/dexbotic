"""DexDataset glue for DM05 (Gemma3 processor + flow-matching action head)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from dexbotic.data.dataset.transform.action import ActionNorm
from dexbotic.data.dataset.transform.common import ToTensor


class DM05ActionNorm(ActionNorm):
    """Quantile norm with clip + zero constant dims (OpenDM parity)."""

    def _normalize(self, data, stats):
        lo = np.asarray(stats["min"], dtype=np.float32)
        hi = np.asarray(stats["max"], dtype=np.float32)
        data = np.clip(np.asarray(data, dtype=np.float32), lo, hi)
        out = ((data - lo) / (hi - lo + 1e-6) * 2.0 - 1.0).astype(np.float32)
        return np.where((lo == 0) & (hi == 0), 0.0, out).astype(np.float32)


class DM05ToTensor(ToTensor):
    def __call__(self, data):
        if isinstance(data, dict):
            return {key: self.__call__(value) for key, value in data.items()}
        if isinstance(data, list):
            return [self.__call__(item) for item in data]
        if isinstance(data, str) or data is None:
            return data
        return torch.as_tensor(data)


class DM05ImagePreprocess:
    target_size = (448, 448)

    def process_pil(self, image: Image.Image | None) -> Image.Image:
        if image is None:
            return Image.new("RGB", self.target_size, color=(0, 0, 0))
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = image.convert("RGB")
        width, height = image.size
        size = max(width, height)
        if width != height:
            canvas = Image.new("RGB", (size, size), color=(0, 0, 0))
            canvas.paste(image, ((size - width) // 2, (size - height) // 2))
            image = canvas
        resample = getattr(Image, "Resampling", Image).BILINEAR
        return image.resize(self.target_size, resample)

    def __call__(self, image, **kwargs):
        arr = np.array(self.process_pil(image), dtype=np.uint8)
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class DM05Tokenization:
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


class DM05DataCollator:
    image_prompts = ("Head", "Left wrist")

    def __init__(
        self,
        processor: AutoProcessor,
        max_length: int = 768,
        valid_action_dim: int = 7,
        model_action_dim: int = 32,
        chunk_size: int = 10,
    ):
        self.processor = processor
        self.tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        self.max_length = max_length
        self.valid_action_dim = valid_action_dim
        self.model_action_dim = model_action_dim
        self.chunk_size = chunk_size
        self.pad_token_id = self.tokenizer.pad_token_id

    def _tokenize_instance(
        self, prompt: str, pil_views: list[Image.Image]
    ) -> dict[str, torch.Tensor]:
        text = f"Robot: Franka\nOverall speed: 0.5\nTask: {prompt}.\n"
        user_content = [{"type": "text", "text": text}]
        for label, image in zip(self.image_prompts, pil_views, strict=True):
            if user_content[-1]["type"] == "text":
                user_content[-1]["text"] += f"{label} image: "
            else:
                user_content.append({"type": "text", "text": f"{label} image: "})
            user_content.append({"type": "image", "image": image})
        messages = [{"role": "user", "content": user_content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if inputs["input_ids"].shape[1] <= self.max_length:
            return inputs
        prompt_token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        overflow = inputs["input_ids"].shape[1] - self.max_length
        keep_tokens = max(0, len(prompt_token_ids) - overflow - 16)
        if keep_tokens < len(prompt_token_ids):
            shortened = self.tokenizer.decode(
                prompt_token_ids[:keep_tokens], skip_special_tokens=False
            ).strip()
            user_content[0]["text"] = user_content[0]["text"].replace(
                prompt, shortened, 1
            )
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        if inputs["input_ids"].shape[1] > self.max_length:
            raise ValueError(
                f"DM05 sequence length {inputs['input_ids'].shape[1]} exceeds "
                f"max_length={self.max_length}; truncating would split image "
                "tokens from pixel_values."
            )
        return inputs

    def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor]:
        tokenized = []
        actions = []
        for inst in instances:
            vals = [int(x) for x in inst["input_ids"].tolist()]
            prompt = bytes(vals).decode("utf-8", errors="ignore") if any(vals) else ""
            image = inst["image"]
            if image.ndim == 3:
                image = image[None]
            pil_views = [
                Image.fromarray(view.permute(1, 2, 0).to(torch.uint8).cpu().numpy())
                for view in image
            ]
            tokenized.append(self._tokenize_instance(prompt, pil_views))
            actions.append(inst["action"].float())

        max_len = max(item["input_ids"].shape[1] for item in tokenized)
        input_ids, attention_mask, token_type_ids, pixel_values = [], [], [], []
        for item in tokenized:
            pad_len = max_len - item["input_ids"].shape[1]
            input_ids.append(
                torch.cat(
                    [
                        item["input_ids"],
                        torch.full(
                            (1, pad_len),
                            self.pad_token_id,
                            dtype=item["input_ids"].dtype,
                        ),
                    ],
                    dim=1,
                )
            )
            attention_mask.append(
                torch.cat(
                    [
                        item["attention_mask"],
                        torch.zeros((1, pad_len), dtype=item["attention_mask"].dtype),
                    ],
                    dim=1,
                )
            )
            token_type_ids.append(
                torch.cat(
                    [
                        item["token_type_ids"],
                        torch.zeros((1, pad_len), dtype=item["token_type_ids"].dtype),
                    ],
                    dim=1,
                )
            )
            pixel_values.append(item["pixel_values"])

        action = torch.stack(actions, dim=0)
        action_mask = torch.zeros(
            len(instances), self.chunk_size, self.model_action_dim, dtype=action.dtype
        )
        action_mask[..., : self.valid_action_dim] = 1.0
        return {
            "input_ids": torch.cat(input_ids, dim=0),
            "attention_mask": torch.cat(attention_mask, dim=0),
            "token_type_ids": torch.cat(token_type_ids, dim=0),
            "pixel_values": torch.cat(pixel_values, dim=0),
            "action": action,
            "action_mask": action_mask,
        }

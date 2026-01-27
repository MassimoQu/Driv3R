from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Iterable

import numpy as np
import torch
from PIL import Image


def _require_transformers():
    try:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Semantic masking requires 'transformers'. Install with: `python3 -m pip install transformers`."
        ) from e
    return SegformerForSemanticSegmentation, SegformerImageProcessor


def _stable_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()


@dataclass
class SegformerRunner:
    model_name: str
    device: str
    fp16: bool
    _processor: object
    _model: torch.nn.Module
    _label2id: dict[str, int]

    _GLOBAL: ClassVar[dict[tuple[str, str, bool], "SegformerRunner"]] = {}

    @classmethod
    def from_pretrained(cls, *, model_name: str, device: str = "cuda:0", fp16: bool = False) -> "SegformerRunner":
        key = (str(model_name), str(device), bool(fp16))
        cached = cls._GLOBAL.get(key)
        if cached is not None:
            return cached

        SegformerForSemanticSegmentation, SegformerImageProcessor = _require_transformers()

        processor = SegformerImageProcessor.from_pretrained(model_name)
        model = SegformerForSemanticSegmentation.from_pretrained(model_name)

        device = str(device)
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        model = model.to(device)
        if fp16 and device.startswith("cuda"):
            model = model.half()
        model.eval()

        # Normalize labels to lowercase for convenience.
        id2label = getattr(model.config, "id2label", {}) or {}
        label2id = {str(v).strip().lower(): int(k) for k, v in id2label.items()}

        runner = cls(
            model_name=str(model_name),
            device=device,
            fp16=bool(fp16 and device.startswith("cuda")),
            _processor=processor,
            _model=model,
            _label2id=label2id,
        )
        cls._GLOBAL[key] = runner
        return runner

    def label_ids(self, labels: Iterable[str]) -> list[int]:
        ids: list[int] = []
        for label in labels:
            label_norm = str(label).strip().lower()
            if not label_norm:
                continue
            idx = self._label2id.get(label_norm)
            if idx is None:
                continue
            ids.append(int(idx))
        # stable order for reproducibility
        return sorted(set(ids))

    @staticmethod
    def cache_path(cache_dir: Path | None, key: str) -> Path | None:
        if cache_dir is None:
            return None
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{_stable_hash(key)}.png"

    def label_map(self, image: Image.Image, *, cache_path: Path | None) -> np.ndarray:
        if cache_path is not None and cache_path.is_file():
            # Stored as uint8 PNG with label ids.
            return np.asarray(Image.open(cache_path), dtype=np.uint8)

        processor = self._processor
        model = self._model

        # transformers processor returns pixel_values float32 by default
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
        if self.fp16 and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].half()

        with torch.no_grad():
            outputs = model(**inputs)
            target_sizes = [(image.size[1], image.size[0])]  # (H, W)
            seg = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)[0]

        seg_np = seg.detach().cpu().to(torch.uint8).numpy()

        if cache_path is not None:
            tmp_path = cache_path.with_name(
                f"{cache_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp.png"
            )
            Image.fromarray(seg_np, mode="L").save(tmp_path, format="PNG")
            try:
                tmp_path.replace(cache_path)
            except FileNotFoundError:
                # Another process may have won the race; caching is best-effort.
                pass

        return seg_np

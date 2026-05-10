"""PyTorch Dataset loading multimodal samples from ZIP archives.

Each ZIP (produced by ``DataRecorder.export_zip``) contains:

* ``samples.jsonl``  — one JSON object per sample
* ``screenshots/<timestamp>.png`` — one PNG per sample

This module pairs each screenshot with its tabular features and label.
"""

import io
import json
import logging
import random
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms

from src.feature_engineering import TABULAR_DIM, extract_tabular_features

logger = logging.getLogger(__name__)

_LABEL_MAP = {"Not Gaming": 0, "Gaming": 1}

# Legacy labels from earlier collection sessions. Older ZIP archives in the
# data lake still contain these strings; remap them at load time so the
# 996+ historical zips remain usable without re-collection.
_LEGACY_LABEL_REMAP = {"Idle": "Not Gaming"}


def _default_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _screenshot_name(timestamp: float) -> str:
    """Reconstruct the PNG filename from a sample timestamp."""
    return datetime.fromtimestamp(
        timestamp, tz=timezone.utc,
    ).strftime("%Y%m%dT%H%M%S%f") + ".png"


class MultimodalGameDataset(Dataset):
    """Dataset loading (image, tabular, label) triples from ZIP archives.

    Parameters
    ----------
    zip_paths:
        Paths to ZIP files exported by ``DataRecorder.export_zip``.
    transform:
        Image transform (default: resize 224, ImageNet normalise).
    """

    def __init__(
        self,
        zip_paths: list[str | Path],
        transform: transforms.Compose | None = None,
    ) -> None:
        self._transform = transform or _default_transform()

        # Index: (zip_path_str, screenshot_zip_path, sample_dict, label_int)
        self._index: list[tuple[str, str, dict, int]] = []
        self._zip_handles: dict[str, zipfile.ZipFile] = {}

        for zp in zip_paths:
            self._index_zip(str(zp))

        logger.info(
            "MultimodalGameDataset: %d samples from %d ZIPs",
            len(self._index), len(zip_paths),
        )

    def _index_zip(self, zip_path: str) -> None:
        with zipfile.ZipFile(zip_path, "r") as zf:
            nameset = set(zf.namelist())
            if "samples.jsonl" not in nameset:
                logger.warning("No samples.jsonl in %s — skipping", zip_path)
                return
            raw = zf.read("samples.jsonl").decode()

        for line in raw.strip().split("\n"):
            if not line.strip():
                continue
            sample = json.loads(line)
            label_str = sample.get("label", "")
            label_str = _LEGACY_LABEL_REMAP.get(label_str, label_str)
            if label_str not in _LABEL_MAP:
                logger.warning("Unknown label %r in %s — skipping", label_str, zip_path)
                continue
            shot_name = "screenshots/" + _screenshot_name(sample["timestamp"])
            if shot_name not in nameset:
                logger.debug("Missing screenshot %s in %s — skipping", shot_name, zip_path)
                continue
            self._index.append((zip_path, shot_name, sample, _LABEL_MAP[label_str]))

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        zip_path, shot_name, sample, label = self._index[idx]

        zf = self._zip_handles.get(zip_path)
        if zf is None:
            zf = zipfile.ZipFile(zip_path, "r")
            self._zip_handles[zip_path] = zf

        img_bytes = zf.read(shot_name)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image_tensor = self._transform(image)

        tabular = torch.tensor(
            extract_tabular_features(sample), dtype=torch.float32,
        )

        return image_tensor, tabular, label

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        for zf in self._zip_handles.values():
            zf.close()
        self._zip_handles.clear()

    def __del__(self) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Split utility
# ---------------------------------------------------------------------------

def split_multimodal_dataset(
    zip_paths: list[str | Path],
    test_ratio: float = 0.2,
    seed: int = 42,
    transform: transforms.Compose | None = None,
) -> tuple[Subset, Subset]:
    """Create train/test splits from ZIP archives.

    Returns two ``Subset`` objects wrapping a single ``MultimodalGameDataset``
    so the ZipFile cache is shared.
    """
    ds = MultimodalGameDataset(zip_paths, transform=transform)
    n = len(ds)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    split = int(n * (1 - test_ratio))
    return Subset(ds, indices[:split]), Subset(ds, indices[split:])

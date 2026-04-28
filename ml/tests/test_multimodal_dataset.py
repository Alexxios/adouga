"""Tests for src.multimodal_dataset — MultimodalGameDataset."""

import io
import json
import zipfile

import torch
from PIL import Image

from src.feature_engineering import TABULAR_DIM
from src.multimodal_dataset import MultimodalGameDataset, split_multimodal_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample_dict(timestamp: float = 1000.0, label: str = "Gaming") -> dict:
    return {
        "timestamp": timestamp,
        "label": label,
        "app_name": "valorant.exe",
        "window_title": "VALORANT",
        "hw_recent": [{
            "timestamp": timestamp,
            "cpu_percent": 30.0,
            "cpu_freq_ghz": 3.0,
            "ram_percent": 50.0,
            "ram_used_gb": 8.0,
            "gpu_present": 0,
            "gpu_load_percent": None,
            "gpu_memory_used_gb": None,
            "gpu_temperature_c": None,
            "disk_read_bps": 1024.0,
            "disk_write_bps": 512.0,
        }],
        "input_since_last": {
            "key_press_count": 3,
            "mouse_click_count": 1,
            "mouse_scroll_count": 0,
            "mouse_move_count": 4,
            "total_count": 8,
            "flick_count": 1,
            "flick_mag_mean": 22.36,
            "flick_mag_max": 22.36,
            "flick_dx_mean": 1.0,
            "flick_dy_mean": 2.0,
            "gaming_keys": {
                "w": 1, "a": 0, "s": 0, "d": 0,
                "space": 0, "shift": 0, "left": 0, "right": 0,
            },
        },
    }


def _screenshot_name(timestamp: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(
        timestamp, tz=timezone.utc,
    ).strftime("%Y%m%dT%H%M%S%f") + ".png"


def _tiny_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), color=(100, 150, 200)).save(buf, format="PNG")
    return buf.getvalue()


def _create_zip(tmp_path, samples: list[dict], name: str = "test.zip") -> str:
    """Write a synthetic ZIP archive and return its path."""
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as zf:
        meta = {"sample_count": len(samples), "labels_present": list({s["label"] for s in samples})}
        zf.writestr("metadata.json", json.dumps(meta))
        zf.writestr("samples.jsonl", "\n".join(json.dumps(s) for s in samples))
        for s in samples:
            shot = "screenshots/" + _screenshot_name(s["timestamp"])
            zf.writestr(shot, _tiny_png())
    return str(path)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def test_dataset_loads_from_zip(tmp_path):
    samples = [_make_sample_dict(1000.0, "Gaming"), _make_sample_dict(1005.0, "Not Gaming")]
    zp = _create_zip(tmp_path, samples)
    ds = MultimodalGameDataset([zp])
    assert len(ds) == 2
    ds.close()


def test_getitem_returns_correct_types(tmp_path):
    zp = _create_zip(tmp_path, [_make_sample_dict()])
    ds = MultimodalGameDataset([zp])
    img, tab, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert isinstance(tab, torch.Tensor)
    assert tab.shape == (TABULAR_DIM,)
    assert isinstance(label, int)
    ds.close()


def test_label_mapping(tmp_path):
    samples = [_make_sample_dict(1000.0, "Gaming"), _make_sample_dict(1005.0, "Not Gaming")]
    zp = _create_zip(tmp_path, samples)
    ds = MultimodalGameDataset([zp])
    _, _, label0 = ds[0]
    _, _, label1 = ds[1]
    assert label0 == 2  # Gaming
    assert label1 == 1  # Not Gaming
    ds.close()


def test_missing_screenshot_skipped(tmp_path):
    """Sample present in JSONL but screenshot missing -> skipped."""
    samples = [_make_sample_dict(1000.0), _make_sample_dict(1005.0)]
    path = tmp_path / "partial.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("metadata.json", "{}")
        zf.writestr("samples.jsonl", "\n".join(json.dumps(s) for s in samples))
        # Only include screenshot for the first sample
        shot = "screenshots/" + _screenshot_name(1000.0)
        zf.writestr(shot, _tiny_png())
    ds = MultimodalGameDataset([str(path)])
    assert len(ds) == 1
    ds.close()


def test_unknown_label_skipped(tmp_path):
    samples = [_make_sample_dict(1000.0, "Unknown")]
    zp = _create_zip(tmp_path, samples)
    ds = MultimodalGameDataset([zp])
    assert len(ds) == 0
    ds.close()


def test_multiple_zips(tmp_path):
    zp1 = _create_zip(tmp_path, [_make_sample_dict(1000.0)], "a.zip")
    zp2 = _create_zip(tmp_path, [_make_sample_dict(2000.0)], "b.zip")
    ds = MultimodalGameDataset([zp1, zp2])
    assert len(ds) == 2
    ds.close()


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def test_split_sizes(tmp_path):
    samples = [_make_sample_dict(float(i)) for i in range(10)]
    zp = _create_zip(tmp_path, samples)
    train_ds, test_ds = split_multimodal_dataset([zp], test_ratio=0.2)
    assert len(train_ds) + len(test_ds) == 10
    assert len(train_ds) == 8
    assert len(test_ds) == 2

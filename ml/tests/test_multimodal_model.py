"""Tests for src.multimodal_model — MultimodalGameClassifier."""

import torch
import pytest

from src.feature_engineering import TABULAR_DIM
from src.multimodal_model import MultimodalGameClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    """Module-scoped model to avoid re-downloading YOLO weights per test."""
    return MultimodalGameClassifier()


def _random_batch(batch_size: int = 2):
    return torch.randn(batch_size, 3, 224, 224), torch.randn(batch_size, TABULAR_DIM)


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def test_forward_output_shape(model):
    img, tab = _random_batch(1)
    model.eval()
    with torch.no_grad():
        out = model(img, tab)
    assert out.shape == (1, 2)


def test_forward_batch(model):
    img, tab = _random_batch(4)
    model.eval()
    with torch.no_grad():
        out = model(img, tab)
    assert out.shape == (4, 2)


def test_forward_produces_finite_values(model):
    img, tab = _random_batch(2)
    model.eval()
    with torch.no_grad():
        out = model(img, tab)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Freeze / unfreeze
# ---------------------------------------------------------------------------

def test_freeze_backbone(model):
    model.freeze_backbone()
    for p in model.backbone.parameters():
        assert not p.requires_grad
    for p in model.backbone_conv.parameters():
        assert not p.requires_grad
    # Head params should still require grad
    for p in model.get_head_params():
        assert p.requires_grad
    model.unfreeze_backbone()  # restore


def test_unfreeze_backbone(model):
    model.freeze_backbone()
    model.unfreeze_backbone()
    for p in model.backbone.parameters():
        assert p.requires_grad
    for p in model.backbone_conv.parameters():
        assert p.requires_grad


def test_get_head_params_excludes_backbone(model):
    backbone_ids = {id(p) for p in model.get_backbone_params()}
    head_ids = {id(p) for p in model.get_head_params()}
    assert backbone_ids.isdisjoint(head_ids)


def test_all_params_covered(model):
    all_ids = {id(p) for p in model.parameters()}
    covered = {id(p) for p in model.get_backbone_params()} | {
        id(p) for p in model.get_head_params()
    }
    assert all_ids == covered


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def test_onnx_export(model, tmp_path):
    import onnx

    model.eval()
    path = tmp_path / "model.onnx"
    img, tab = _random_batch(1)
    torch.onnx.export(
        model,
        (img, tab),
        str(path),
        input_names=["image", "tabular"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
        dynamic_axes={
            "image": {0: "batch"},
            "tabular": {0: "batch"},
            "logits": {0: "batch"},
        },
    )
    onnx_model = onnx.load(str(path))
    onnx.checker.check_model(onnx_model)

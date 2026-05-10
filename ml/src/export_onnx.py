"""Export PyTorch model to ONNX format."""

import sys
from pathlib import Path
import torch
import torch.onnx
import onnx

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import GameClassifier


def export_to_onnx(
    model_path: str = "models/model_best.pth",
    onnx_path: str = "models/model.onnx",
    opset_version: int = 17
):
    """Export PyTorch model to ONNX format.

    Args:
        model_path: Path to PyTorch model checkpoint
        onnx_path: Output path for ONNX model
        opset_version: ONNX opset version (17 is stable and widely supported)
    """
    device = torch.device('cpu')

    # Load model
    print(f"Loading model from {model_path}...")
    model = GameClassifier(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    # Export to ONNX
    print(f"Exporting to ONNX format (opset {opset_version})...")

    # Use dynamo=False to use the legacy exporter which is more stable
    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        dynamo=False  # Use legacy exporter for better compatibility
    )
    print(f"✓ Model successfully exported to: {onnx_path}")

    # Verify the exported model
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verification passed")

    # Print model info
    print(f"\nModel info:")
    print(f"  Input shape: [batch_size, 3, 224, 224]")
    print(f"  Output shape: [batch_size, 2]")
    print(f"  Opset version: {opset_version}")


def export_multimodal_to_onnx(
    model_path: str = "models/multimodal_best.pth",
    onnx_path: str = "models/multimodal.onnx",
    opset_version: int = 17,
) -> None:
    """Export the multimodal classifier to ONNX format.

    The exported model accepts two inputs: ``image`` [B,3,224,224] and
    ``tabular`` [B, TABULAR_DIM].
    """
    from src.feature_engineering import TABULAR_DIM
    from src.multimodal_model import MultimodalGameClassifier

    device = torch.device("cpu")

    print(f"Loading multimodal model from {model_path}...")
    model = MultimodalGameClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    dummy_image = torch.randn(1, 3, 224, 224, device=device)
    dummy_tabular = torch.randn(1, TABULAR_DIM, device=device)

    print(f"Exporting to ONNX format (opset {opset_version})...")
    torch.onnx.export(
        model,
        (dummy_image, dummy_tabular),
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["image", "tabular"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "tabular": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        dynamo=False,
    )
    print(f"Model exported to: {onnx_path}")

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed")

    print(f"\nModel info:")
    print(f"  Image input:   [batch_size, 3, 224, 224]")
    print(f"  Tabular input: [batch_size, {TABULAR_DIM}]")
    print(f"  Output:        [batch_size, 2]")
    print(f"  Opset version: {opset_version}")


def main():
    """Main export function."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal", action="store_true",
                        help="Export the multimodal model instead of the image-only model")
    args = parser.parse_args()
    if args.multimodal:
        export_multimodal_to_onnx()
    else:
        export_to_onnx()


if __name__ == "__main__":
    main()

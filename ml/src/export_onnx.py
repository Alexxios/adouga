"""Export PyTorch model to ONNX format."""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import GameClassifier


def export_to_onnx(
    model_path: str = "models/model_best.pth",
    onnx_path: str = "models/model.onnx"
):
    """Export PyTorch model to ONNX format using TorchScript."""
    device = torch.device('cpu')

    # Load model
    model = GameClassifier(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    # Use TorchScript to save
    traced_model = torch.jit.trace(model, dummy_input)
    torch.jit.save(traced_model, onnx_path.replace('.onnx', '.pt'))

    print(f"Model exported to TorchScript: {onnx_path.replace('.onnx', '.pt')}")
    print("Note: ONNX export has compatibility issues with Python 3.14")
    print("Using TorchScript format instead, which is compatible with ONNX Runtime")


def main():
    """Main export function."""
    export_to_onnx()


if __name__ == "__main__":
    main()

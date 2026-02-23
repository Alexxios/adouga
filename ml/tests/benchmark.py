"""Benchmark script for PyTorch and TorchScript models."""

import sys
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import GameDataset
from src.model import GameClassifier


def benchmark_pytorch(model_path: str, test_dir: str, num_runs: int = 100):
    """Benchmark PyTorch model."""
    device = torch.device('cuda' if torch.cuda.is_available() else
                         'mps' if torch.backends.mps.is_available() else 'cpu')

    # Load model
    model = GameClassifier(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Load test dataset
    test_dataset = GameDataset(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Warmup
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= 10:
                break
            images = images.to(device)
            _ = model(images)

    # Benchmark
    times = []
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= num_runs:
                break
            images = images.to(device)

            start = time.time()
            _ = model(images)
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()

            times.append(end - start)

    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time

    return avg_time, fps


def benchmark_torchscript(model_path: str, test_dir: str, num_runs: int = 100):
    """Benchmark TorchScript model."""
    device = torch.device('cuda' if torch.cuda.is_available() else
                         'mps' if torch.backends.mps.is_available() else 'cpu')

    # Load TorchScript model
    model = torch.jit.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()

    # Load test dataset
    test_dataset = GameDataset(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Warmup
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= 10:
                break
            images = images.to(device)
            _ = model(images)

    # Benchmark
    times = []
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= num_runs:
                break
            images = images.to(device)

            start = time.time()
            _ = model(images)
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.time()

            times.append(end - start)

    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time

    return avg_time, fps


def main():
    """Main benchmark function."""
    print("="*50)
    print("Model Benchmark")
    print("="*50)

    # Benchmark PyTorch model
    print("\nBenchmarking PyTorch model...")
    pytorch_time, pytorch_fps = benchmark_pytorch(
        "models/model_best.pth",
        "data/split/test",
        num_runs=100
    )
    print(f"PyTorch - Avg time: {pytorch_time*1000:.2f}ms, FPS: {pytorch_fps:.2f}")

    # Benchmark TorchScript model
    print("\nBenchmarking TorchScript model...")
    torchscript_time, torchscript_fps = benchmark_torchscript(
        "models/model.pt",
        "data/split/test",
        num_runs=100
    )
    print(f"TorchScript - Avg time: {torchscript_time*1000:.2f}ms, FPS: {torchscript_fps:.2f}")

    # Comparison
    speedup = pytorch_time / torchscript_time
    print(f"\nSpeedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()

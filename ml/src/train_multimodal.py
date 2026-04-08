"""Training script for the multimodal game classifier (YOLO + tabular).

Usage::

    python -m src.train_multimodal data/zips/*.zip
    python -m src.train_multimodal --epochs 20 --unfreeze-epoch 5 path/to/*.zip
"""

import argparse
import logging
import sys
from glob import glob
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.multimodal_dataset import split_multimodal_dataset
from src.multimodal_model import MultimodalGameClassifier

logger = logging.getLogger(__name__)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_multimodal(
    zip_paths: list[str],
    num_epochs: int = 20,
    batch_size: int = 32,
    lr: float = 0.001,
    unfreeze_epoch: int = 5,
    test_ratio: float = 0.2,
) -> None:
    """Train the multimodal classifier end-to-end.

    Parameters
    ----------
    zip_paths:
        Paths to ZIP archives exported by ``DataRecorder.export_zip``.
    num_epochs:
        Total training epochs.
    batch_size:
        Mini-batch size.
    lr:
        Base learning rate.
    unfreeze_epoch:
        Epoch at which the YOLO backbone is unfrozen for fine-tuning.
    test_ratio:
        Fraction of samples held out for evaluation.
    """
    device = _get_device()
    print(f"Using device: {device}")

    # ---- Data ----
    train_subset, test_subset = split_multimodal_dataset(
        zip_paths, test_ratio=test_ratio,
    )
    print(f"Train: {len(train_subset)}, Test: {len(test_subset)}")
    if len(train_subset) == 0:
        print("No training samples — aborting.")
        return

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    # ---- Model ----
    model = MultimodalGameClassifier().to(device)
    model.freeze_backbone()
    print("Backbone frozen for initial training phase")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.get_head_params(), lr=lr)

    # ---- Training loop ----
    best_acc = 0.0
    for epoch in range(num_epochs):
        # Unfreeze backbone at the designated epoch
        if epoch == unfreeze_epoch:
            model.unfreeze_backbone()
            optimizer = optim.Adam([
                {"params": model.get_backbone_params(), "lr": lr * 0.1},
                {"params": model.get_head_params(), "lr": lr},
            ])
            print(f"Epoch {epoch + 1}: backbone unfrozen, LR backbone={lr * 0.1}")

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, tabular, labels in train_loader:
            images = images.to(device)
            tabular = tabular.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, tabular)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --- Eval ---
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, tabular, labels in test_loader:
                images = images.to(device)
                tabular = tabular.to(device)
                labels = labels.to(device)

                outputs = model(images, tabular)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_loss /= max(test_total, 1)
        test_acc = test_correct / max(test_total, 1)

        print(
            f"Epoch {epoch + 1}/{num_epochs} — "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} — "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            Path("models").mkdir(exist_ok=True)
            torch.save(model.state_dict(), "models/multimodal_best.pth")
            print(f"  Saved best model (acc: {best_acc:.4f})")

    print(f"\nTraining complete! Best accuracy: {best_acc:.4f}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(
        description="Train the multimodal game classifier",
    )
    parser.add_argument(
        "zips", nargs="*", default=None,
        help="ZIP archive paths (default: data/zips/*.zip)",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--unfreeze-epoch", type=int, default=5)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    args = parser.parse_args()

    zip_paths = args.zips or sorted(glob("data/zips/*.zip"))
    if not zip_paths:
        print("No ZIP files found. Pass paths as arguments or place them in data/zips/.")
        sys.exit(1)

    print(f"Training on {len(zip_paths)} ZIP archive(s)")
    train_multimodal(
        zip_paths=zip_paths,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        unfreeze_epoch=args.unfreeze_epoch,
        test_ratio=args.test_ratio,
    )


if __name__ == "__main__":
    main()

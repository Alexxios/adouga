"""Evaluation script for the trained model."""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import GameDataset
from src.model import GameClassifier


def evaluate_model(model_path: str = "models/model_best.pth", test_dir: str = "data/split/test"):
    """Evaluate the trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = GameClassifier(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Load test dataset
    test_dataset = GameDataset(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Test dataset size: {len(test_dataset)}")

    # Evaluate
    correct = 0
    total = 0
    class_correct = [0, 0]
    class_total = [0, 0]

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for label, pred in zip(labels, predicted):
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1

    accuracy = correct / total
    game_acc = class_correct[0] / class_total[0] if class_total[0] > 0 else 0
    not_game_acc = class_correct[1] / class_total[1] if class_total[1] > 0 else 0

    print(f"\nEvaluation Results:")
    print(f"Overall Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Game Accuracy: {game_acc:.4f} ({class_correct[0]}/{class_total[0]})")
    print(f"Not-Game Accuracy: {not_game_acc:.4f} ({class_correct[1]}/{class_total[1]})")


def main():
    """Main evaluation function."""
    evaluate_model()


if __name__ == "__main__":
    main()

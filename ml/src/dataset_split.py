"""Split dataset into train and test sets."""

import shutil
import random
from pathlib import Path


def split_dataset(combined_dir: Path, output_dir: Path, test_ratio: float = 0.2):
    """Split dataset into train and test."""
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"

    for label in ["game", "not_game"]:
        (train_dir / label).mkdir(parents=True, exist_ok=True)
        (test_dir / label).mkdir(parents=True, exist_ok=True)

        # Get all images for this label
        images = list((combined_dir / label).glob("*.jpg"))
        random.shuffle(images)

        # Split
        split_idx = int(len(images) * (1 - test_ratio))
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        # Copy files
        for img in train_images:
            shutil.copy2(img, train_dir / label / img.name)

        for img in test_images:
            shutil.copy2(img, test_dir / label / img.name)

        print(f"{label}: {len(train_images)} train, {len(test_images)} test")


def main():
    """Split the combined dataset."""
    random.seed(42)

    combined_dir = Path("data/combined")
    output_dir = Path("data/split")

    print("Splitting dataset...")
    split_dataset(combined_dir, output_dir, test_ratio=0.2)
    print("Split completed!")


if __name__ == "__main__":
    main()

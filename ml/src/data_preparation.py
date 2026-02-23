"""Data preparation: download and combine datasets."""

from pathlib import Path
from datasets import load_dataset
from PIL import Image


def download_dataset(dataset_name: str, label: str, output_dir: Path, max_images: int = 1000):
    """Download images from a dataset."""
    print(f"Downloading {dataset_name}...")
    label_dir = output_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_name, split="train", streaming=True)
    count = 0

    for idx, sample in enumerate(dataset):
        if count >= max_images:
            break

        # Find image field
        image = None
        for field in ['image', 'img', 'picture', 'screenshot', 'frame']:
            if field in sample and isinstance(sample[field], Image.Image):
                image = sample[field]
                break

        if image is None:
            continue

        # Save image
        filename = f"{dataset_name.replace('/', '_')}_{count:06d}.jpg"
        filepath = label_dir / filename

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image.save(filepath, 'JPEG', quality=95)
        count += 1

        if count % 100 == 0:
            print(f"  {count} images saved")

    print(f"Completed: {count} images from {dataset_name}")
    return count


def main():
    """Download and combine datasets."""
    output_dir = Path("data/combined")

    # Game datasets
    game_datasets = [
        "taesiri/GameplayCaptions-GPT-4V",
        "taesiri/GameplayCaptions-GPT-4V-V2",
        "Bingsu/Gameplay_Images",
    ]

    # Not-game datasets
    not_game_datasets = [
        "mlfoundations-cua-dev/easyr1-showui-desktop-only-4k9-omniparser-qwen-tool-call-4MP",
        "showlab/ShowUI-desktop",
    ]

    # Download game images
    print("\n=== Downloading GAME images ===")
    for dataset in game_datasets:
        download_dataset(dataset, "game", output_dir, max_images=500)

    # Download not-game images
    print("\n=== Downloading NOT-GAME images ===")
    for dataset in not_game_datasets:
        download_dataset(dataset, "not_game", output_dir, max_images=500)

    print("\nData preparation completed!")


if __name__ == "__main__":
    main()

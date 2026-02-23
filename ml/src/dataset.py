"""PyTorch Dataset for game classification."""

import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class GameDataset(Dataset):
    """Simple dataset for game vs not-game classification."""

    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform or self.get_default_transform()

        # Collect all images
        self.samples = []
        for label_idx, label in enumerate(["game", "not_game"]):
            label_dir = self.data_dir / label
            if label_dir.exists():
                for img_path in label_dir.glob("*.jpg"):
                    self.samples.append((str(img_path), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

    @staticmethod
    def get_default_transform():
        """Get default image transform."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

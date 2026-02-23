# Game Classification Model

A ResNet18-based binary classifier to distinguish between game and non-game images.

## Project Structure

```
.
├── src/
│   ├── data_preparation.py  # Download and combine datasets
│   ├── dataset_split.py     # Split into train/test sets
│   ├── dataset.py           # PyTorch Dataset class
│   ├── model.py             # ResNet18 classifier
│   ├── train.py             # Training script
│   └── export_onnx.py       # Model export to TorchScript
├── tests/
│   ├── evaluate.py          # Model evaluation
│   └── benchmark.py         # Performance benchmarking
├── data/
│   ├── combined/            # Combined dataset
│   └── split/               # Train/test split
└── models/                  # Saved models
```

## Setup

Install dependencies using Poetry:

```bash
poetry install
```

## Usage

### 1. Download and Prepare Data

Download images from HuggingFace datasets and combine them:

```bash
poetry run python src/data_preparation.py
```

### 2. Split Dataset

Split the combined dataset into train (80%) and test (20%) sets:

```bash
poetry run python src/dataset_split.py
```

### 3. Train Model

Train the ResNet18-based classifier:

```bash
poetry run python src/train.py
```

Training results:
- Train dataset: 1600 images
- Test dataset: 400 images
- Best test accuracy: 100%
- Model saved to: `models/model_best.pth`

### 4. Evaluate Model

Evaluate the trained model on the test set:

```bash
poetry run python tests/evaluate.py
```

### 5. Export Model

Export the model to TorchScript format:

```bash
poetry run python src/export_onnx.py
```

Note: ONNX export has compatibility issues with Python 3.14, so TorchScript format is used instead.

### 6. Benchmark Models

Compare performance between PyTorch and TorchScript models:

```bash
poetry run python tests/benchmark.py
```

Benchmark results (on Apple M-series):
- PyTorch: ~4.43ms per image (~226 FPS)
- TorchScript: ~4.52ms per image (~221 FPS)

## Model Architecture

- Base: ResNet18 (pretrained on ImageNet)
- Input: 224x224 RGB images
- Output: 2 classes (game, not_game)
- Training: 10 epochs, Adam optimizer, learning rate 0.001

## Dataset

### Game Images (Label: 0)
- taesiri/GameplayCaptions-GPT-4V
- taesiri/GameplayCaptions-GPT-4V-V2
- Bingsu/Gameplay_Images

### Not-Game Images (Label: 1)
- mlfoundations-cua-dev/easyr1-showui-desktop-only-4k9-omniparser-qwen-tool-call-4MP
- showlab/ShowUI-desktop

Total: ~2000 images (1500 game, 500 not-game)

## Results

- Overall Accuracy: 100%
- Game Accuracy: 100% (300/300)
- Not-Game Accuracy: 100% (100/100)

## Requirements

- Python 3.14
- PyTorch 2.9+
- torchvision 0.24+
- datasets 4.5+
- Pillow
- Other dependencies in pyproject.toml

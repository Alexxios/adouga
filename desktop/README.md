# Adouga - Activity Detection and Observation Using Gaming Analytics

A Python-based system monitoring and AI prediction application that captures screenshots of the foreground application and uses machine learning to classify activity.

## Project Structure

```
adouga/
├── backend/          # Backend services
├── common/           # Shared utilities
├── desktop/          # Desktop application
│   ├── src/
│   │   ├── main.py           # Main application entry point
│   │   ├── inference.py      # ONNX inference module
│   │   ├── utils.py          # Screenshot utilities
│   │   ├── system_monitor/   # System monitoring modules
│   │   └── ui/               # UI components
│   └── pyproject.toml        # Desktop dependencies
└── ml/               # Machine learning models
    ├── models/
    │   └── model.onnx        # ONNX classification model
    └── src/                  # ML training scripts
```

## Desktop Application

The desktop application provides real-time system monitoring and AI-powered activity classification.

### Features

- **Live System Monitoring**: Track CPU, RAM usage, and user input activity
- **Screenshot Capture**: Automatically captures screenshots of the foreground application
- **AI Classification**: Uses ONNX model to classify activity (Gaming vs Not Gaming)
- **Multi-Platform Support**: Works on macOS and Windows
- **GPU Acceleration**: Supports CUDA, CoreML, and other execution providers

### Installation

1. Navigate to the desktop directory:
```bash
cd desktop
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```

### Running the Application

```bash
python -m src.main
```

Or from the project root:
```bash
cd desktop && python -m src.main
```

### Dependencies

The desktop application uses minimal dependencies for optimal performance:

- **System Monitoring**: `psutil`, `pynput`, `mss`
- **UI**: `tkinter`, `matplotlib`, `pillow`
- **AI Inference**: `numpy`, `onnxruntime` (CPU/GPU)
- **macOS Support**: `pyobjc` frameworks for window detection

**Note**: PyTorch is NOT required for inference - only ONNX Runtime is used.

### UI Pages

1. **Live Stats**: Real-time graphs showing CPU, RAM, and input activity
2. **AI Analysis**: Screenshot display with AI prediction results showing:
   - Predicted class (Gaming/Not Gaming)
   - Confidence percentage
   - Probability distribution
   - Execution provider (CPU/CUDA/CoreML)

### ONNX Model

The application uses a ResNet18-based binary classifier exported to ONNX format:
- **Input**: 224x224 RGB images
- **Output**: 2 classes (Not Gaming, Gaming)
- **Location**: `ml/models/model.onnx`

### Execution Providers

The inference module automatically selects the best available execution provider:
1. **CUDA** (NVIDIA GPUs)
2. **TensorRT** (NVIDIA optimized)
3. **CoreML** (Apple Silicon/macOS)
4. **DirectML** (Windows)
5. **ROCm** (AMD GPUs)
6. **CPU** (fallback)

### Development

#### Running Tests

```bash
cd desktop
poetry run pytest
```

#### Code Structure

- `main.py`: Application controller and main loop
- `inference.py`: ONNX model inference with multi-provider support
- `ui/__init__.py`: UI pages (MonitorPage, AIPage)
- `system_monitor/`: Platform-specific window detection and input monitoring
- `utils.py`: Screenshot capture utilities

### Platform-Specific Notes

#### macOS
- Requires "Screen Recording" permission for screenshots
- Requires "Input Monitoring" permission for input tracking
- Uses PyObjC frameworks for native window detection

#### Windows
- Uses Win32 API for window detection
- Supports DirectML for GPU acceleration

## ML Module

The ML module contains training scripts and model export utilities:

```bash
cd ml
poetry install
poetry shell

# Train model
python -m src.train

# Export to ONNX
python -m src.export_onnx
```

## License

This project is for educational and research purposes.

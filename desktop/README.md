# Adouga - Automatic Detection of User Gaming Activity

A Python-based system monitoring and AI prediction application that captures screenshots of the foreground application and uses machine learning to classify activity.

## Project Structure

```
adouga/
├── backend/          # Backend services
├── common/           # Shared utilities
├── desktop/          # Desktop application
│   ├── src/
│   │   ├── main.py           # User app entry point
│   │   ├── dev_main.py       # Dev app entry point (ML data collection)
│   │   ├── core/             # Shared modules (used by both apps)
│   │   │   ├── models.py         # DataSample dataclass
│   │   │   ├── input_monitor.py  # Keyboard & mouse tracking
│   │   │   ├── hardware_monitor.py  # CPU, RAM, GPU, disk sampling
│   │   │   ├── window.py         # Active window detection
│   │   │   ├── screenshot.py     # Screen capture
│   │   │   └── theme.py          # UI theme constants
│   │   ├── app/              # User app only
│   │   │   ├── inference.py      # ONNX model inference
│   │   │   └── ui/               # User app pages & widgets
│   │   └── dev/              # Dev app only
│   │       ├── recorder.py       # DataRecorder (rolling buffer + export)
│   │       ├── uploader.py       # YaDisk upload
│   │       ├── batch_uploader.py # Queue-based batch uploader
│   │       ├── hotkeys.py        # Global hotkey manager
│   │       └── ui/               # Dev app pages
│   ├── tests/
│   │   ├── core/             # Tests for shared modules
│   │   ├── app/              # Tests for user app modules
│   │   └── dev/              # Tests for dev app modules
│   └── pyproject.toml
└── ml/               # Machine learning models & training
```

### Package layout

- **`core/`** — shared foundation used by both apps: hardware/input monitors, window detection, screenshot capture, data models, and UI theme.
- **`app/`** — user-facing application: ONNX inference and UI pages (Live Stats, AI Analysis, Mouse Flicks).
- **`dev/`** — developer tool for collecting ML training data: data recorder, YaDisk uploader, batch upload pipeline, hotkeys, and dev UI.

## User Application

Real-time system monitoring and AI-powered activity classification.

### Features

- **Live System Monitoring**: Track CPU, RAM usage, and user input activity
- **Screenshot Capture**: Automatically captures screenshots of the foreground application
- **AI Classification**: Uses ONNX model to classify activity (Gaming vs Not Gaming)
- **Multi-Platform Support**: Works on macOS and Windows
- **GPU Acceleration**: Supports CUDA, CoreML, and other execution providers

### Running

```bash
cd desktop
poetry install
poetry run python -m src.main
```

### UI Pages

1. **AI Analysis**: Screenshot display with AI prediction results (class, confidence, provider)
2. **Live Stats**: Real-time graphs for CPU, RAM, and input activity
3. **Mouse Flicks**: Polar chart of recent mouse movement vectors

### ONNX Model

ResNet18-based binary classifier exported to ONNX format:
- **Input**: 224x224 RGB images
- **Output**: 2 classes (Not Gaming, Gaming)
- **Location**: `ml/models/model.onnx`

### Execution Providers

The inference module automatically selects the best available provider:
1. CUDA (NVIDIA) / TensorRT
2. CoreML (Apple Silicon)
3. DirectML (Windows)
4. ROCm (AMD)
5. CPU (fallback)

## Dev Application

ML training data collection tool with automated YaDisk upload.

### Running

```bash
cd desktop

# Basic usage (saves locally, no upload)
poetry run python src/dev_main.py --no-upload

# With Yandex Disk upload (requires .env with YADISK_TOKEN)
cp .env.example .env          # fill in your token
poetry run python src/dev_main.py

# Custom states and window
poetry run python src/dev_main.py --states "Not Gaming" "Gaming" --window 300

# See all options
poetry run python src/dev_main.py --help
```

### Features

- 0.5s capture interval (configurable via `--interval`)
- Configurable first-capture delay
- Tester name field for multi-tester concurrent use
- Automatic batch upload to YaDisk (every 10 samples)
- Manual flush via "Flush Now" button
- Global hotkeys (work even when window not focused)

### Global Hotkeys

| Hotkey | Action |
|--------|--------|
| `Ctrl+Shift+R` | Toggle recording on/off |
| `Ctrl+Shift+N` | Advance to next classification state |

## Development

### Dependencies

- **System Monitoring**: `psutil`, `pynput`, `mss`
- **UI**: `tkinter`, `matplotlib`, `pillow`
- **AI Inference**: `numpy`, `onnxruntime` (CPU/GPU)
- **macOS Support**: `pyobjc` frameworks for window detection

### Running Tests

```bash
cd desktop

# Run all tests
poetry run pytest

# Verbose output
poetry run pytest -v --tb=short

# Run a single module
poetry run pytest tests/dev/test_recorder.py

# Filter by name
poetry run pytest -k "upload"
```

Tests are written in **pure pytest style** (flat functions, no unittest classes).
The suite is split into:
- **Unit tests** — all external I/O mocked, run instantly offline
- **Integration tests** — guarded with `skipif`; require real OS resources or environment variables

### Platform Notes

#### macOS
- Requires "Screen Recording" and "Input Monitoring" permissions
- Uses PyObjC frameworks for native window detection

#### Windows
- Uses Win32 API for window detection
- Supports DirectML for GPU acceleration

## License

This project is for educational and research purposes.

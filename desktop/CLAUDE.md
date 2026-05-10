# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Adouga (Automatic Detection of User Gaming Activity) — a Python desktop application that captures screenshots and system metrics, then uses an ONNX ML model to classify activity (Gaming vs Not Gaming). This repo is the `desktop/` component of a larger monorepo (`adouga/`).

There are **two separate apps** sharing a common `core/` layer:
- **User app** (`src/main.py`) — real-time monitoring + ONNX inference UI (tkinter)
- **Dev app** (`src/dev_main.py`) — ML training data collection tool with YaDisk upload

## Commands

All commands run from the `desktop/` directory.

```bash
# Install dependencies
poetry install

# Run user app
poetry run python -m src.main

# Run dev app
poetry run python -m src.dev_main --no-upload
poetry run python -m src.dev_main --states "Not Gaming" "Gaming" --window 300

# Run all tests
poetry run pytest

# Run tests verbose
poetry run pytest -v --tb=short

# Run a single test file
poetry run pytest tests/dev/test_recorder.py

# Run tests by keyword
poetry run pytest -k "upload"
```

## Architecture

```
src/
├── main.py              # User app entry (SystemMonitorApp — tk.Tk subclass)
├── dev_main.py          # Dev app entry (DevApp — tk.Tk subclass)
├── core/                # Shared foundation (both apps use this)
│   ├── models.py        # DataSample dataclass — central data structure
│   ├── hardware_monitor.py  # Threaded CPU/RAM/GPU/disk sampling
│   ├── input_monitor.py     # Keyboard & mouse tracking (pynput)
│   ├── window.py            # Active window rect detection (platform-specific)
│   ├── screenshot.py        # Screen capture via mss
│   └── theme.py             # ModernTheme UI constants + recolor helpers
├── app/                 # User app only
│   ├── inference.py     # ONNXClassifier — auto-selects execution provider
│   └── ui/              # Pages: AIPage, MonitorPage, FlicksPage, DistributionsPage
└── dev/                 # Dev app only
    ├── recorder.py      # DataRecorder — rolling buffer + batch export
    ├── uploader.py      # YaDiskUploader — single file upload
    ├── batch_uploader.py # Queue-based threaded batch uploader
    ├── hotkeys.py       # Global hotkey manager (Ctrl+Shift+R/N)
    └── ui/              # DevPage UI
```

Key patterns:
- Both apps are `tk.Tk` subclasses with a main loop polling at intervals
- Monitors (HardwareMonitor, InputMonitor) run in background threads; data is retrieved via getter methods
- Dev app uses a callback chain: `DataRecorder.on_batch_ready` → `BatchUploader.enqueue` → `YaDiskUploader`
- ONNX inference auto-selects provider: CUDA > CoreML > DirectML > ROCm > CPU

## Test Conventions

- **Pure pytest style**: flat `def test_*()` functions, no unittest classes
- All external I/O is mocked; tests run offline
- Integration tests use `pytest.mark.skipif` guards for OS resources / env vars
- Test layout mirrors `src/`: `tests/core/`, `tests/app/`, `tests/dev/`
- `pytest.ini` sets `pythonpath = .` and `--import-mode=importlib`

## Platform Notes

- **macOS**: requires Screen Recording + Input Monitoring permissions; uses PyObjC for window detection
- **Windows**: uses Win32 API for window detection; DirectML for GPU; PyInstaller build via `dev_main.spec`
- ONNX inference group is optional (`poetry install --with inference`)
- macOS PyObjC group is optional (`poetry install --with macos`)

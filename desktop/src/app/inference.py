"""ONNX inference module — supports both image-only and multimodal models.

The classifier inspects the ONNX session at load time. If the model has two
inputs named ``image`` and ``tabular`` (multimodal), a 149-dim tabular vector
is required for inference. Otherwise the legacy single-input path is used.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from PIL import Image

from src.core.feature_engineering import TABULAR_DIM, extract_tabular_features


class ONNXClassifier:
    """ONNX-based classifier for binary Gaming / Not Gaming classification.

    Supports two model variants:

    * **Image-only** (legacy): single input named ``input``, shape
      ``[1, 3, 224, 224]``.
    * **Multimodal**: two inputs ``image`` ``[1, 3, 224, 224]`` and
      ``tabular`` ``[1, 149]``.

    The variant is detected at session-load time from the model's input
    metadata; callers always invoke :meth:`predict_with_details` with an
    optional ``sample`` dict.
    """

    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = True):
        if model_path is None:
            if getattr(sys, "frozen", False):
                model_path = Path(sys._MEIPASS) / "ml" / "models" / "model.onnx"
            else:
                model_path = (
                    Path(__file__).parent.parent.parent.parent
                    / "ml" / "models" / "model.onnx"
                )

        self.model_path = str(model_path)

        providers = self._get_execution_providers(use_gpu)
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        print(f"ONNX Runtime using provider: {self.session.get_providers()[0]}")

        inputs = self.session.get_inputs()
        input_names = {inp.name for inp in inputs}
        self.is_multimodal = "tabular" in input_names and "image" in input_names

        if self.is_multimodal:
            self.image_input_name = "image"
            self.tabular_input_name = "tabular"
            print(f"Multimodal ONNX detected (image + tabular[{TABULAR_DIM}])")
        else:
            self.image_input_name = inputs[0].name
            self.tabular_input_name = None
            print(f"Image-only ONNX detected (input: {self.image_input_name})")

        self.output_name = self.session.get_outputs()[0].name
        self.input_size = (224, 224)
        self.class_labels = ["Not Gaming", "Gaming"]

    def _get_execution_providers(self, use_gpu: bool) -> list[str]:
        available = ort.get_available_providers()
        providers: list[str] = []
        if use_gpu:
            for p in (
                "CUDAExecutionProvider",
                "TensorrtExecutionProvider",
                "CoreMLExecutionProvider",
                "DmlExecutionProvider",
                "ROCMExecutionProvider",
            ):
                if p in available:
                    providers.append(p)
        providers.append("CPUExecutionProvider")
        return providers

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        image = image.resize(self.input_size, Image.Resampling.LANCZOS)
        if image.mode != "RGB":
            image = image.convert("RGB")
        arr = np.array(image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, axis=0)

    def _build_tabular(self, sample: Optional[dict]) -> np.ndarray:
        if sample is None:
            return np.zeros((1, TABULAR_DIM), dtype=np.float32)
        return np.asarray(extract_tabular_features(sample), dtype=np.float32).reshape(1, -1)

    def _run(self, image: Image.Image, sample: Optional[dict]) -> np.ndarray:
        feeds = {self.image_input_name: self.preprocess_image(image)}
        if self.is_multimodal:
            feeds[self.tabular_input_name] = self._build_tabular(sample)
        outputs = self.session.run([self.output_name], feeds)
        return outputs[0][0]

    def predict(
        self,
        image: Image.Image,
        sample: Optional[dict] = None,
    ) -> tuple[str, float]:
        logits = self._run(image, sample)
        exp = np.exp(logits - np.max(logits))
        probs = exp / np.sum(exp)
        idx = int(np.argmax(probs))
        return self.class_labels[idx], float(probs[idx])

    def predict_with_details(
        self,
        image: Image.Image,
        sample: Optional[dict] = None,
    ) -> dict:
        logits = self._run(image, sample)
        exp = np.exp(logits - np.max(logits))
        probs = exp / np.sum(exp)
        idx = int(np.argmax(probs))
        return {
            "predicted_class": self.class_labels[idx],
            "confidence": float(probs[idx]),
            "probabilities": {
                label: float(p) for label, p in zip(self.class_labels, probs)
            },
            "provider": self.session.get_providers()[0],
        }

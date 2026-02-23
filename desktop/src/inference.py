"""ONNX inference module for image classification."""

import numpy as np
from pathlib import Path
from PIL import Image
import onnxruntime as ort
from typing import Optional

class ONNXClassifier:
    """ONNX-based image classifier for binary classification."""

    def __init__(self, model_path: Optional[str] = None, use_gpu: bool = True):
        """Initialize ONNX classifier.

        Args:
            model_path: Path to ONNX model file. If None, uses default path.
            use_gpu: Whether to try using GPU acceleration if available.
        """
        if model_path is None:
            # Default path relative to desktop module
            model_path = Path(__file__).parent.parent.parent / "ml" / "models" / "model.onnx"

        self.model_path = str(model_path)

        # Determine available execution providers
        providers = self._get_execution_providers(use_gpu)

        # Create ONNX Runtime session
        self.session = ort.InferenceSession(
            self.model_path,
            providers=providers
        )

        # Log which provider is being used
        print(f"ONNX Runtime using provider: {self.session.get_providers()[0]}")

        # Get model input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Model expects 224x224 RGB images
        self.input_size = (224, 224)

        # Class labels (adjust based on your model)
        self.class_labels = ["Not Gaming", "Gaming"]

    def _get_execution_providers(self, use_gpu: bool) -> list[str]:
        """Get list of execution providers in priority order.

        Args:
            use_gpu: Whether to try using GPU acceleration.

        Returns:
            List of provider names in priority order.
        """
        available_providers = ort.get_available_providers()
        providers = []

        if use_gpu:
            # Priority order for GPU providers
            gpu_providers = [
                'CUDAExecutionProvider',      # NVIDIA CUDA
                'TensorrtExecutionProvider',  # NVIDIA TensorRT
                'CoreMLExecutionProvider',    # Apple CoreML (macOS)
                'DmlExecutionProvider',       # DirectML (Windows)
                'ROCMExecutionProvider',      # AMD ROCm
            ]

            for provider in gpu_providers:
                if provider in available_providers:
                    providers.append(provider)

        # Always add CPU as fallback
        providers.append('CPUExecutionProvider')

        return providers

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess PIL Image for model input.

        Args:
            image: PIL Image object

        Returns:
            Preprocessed numpy array ready for inference
        """
        # Resize to model input size
        image = image.resize(self.input_size, Image.Resampling.LANCZOS)

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert to numpy array and normalize
        img_array = np.array(image).astype(np.float32)

        # Normalize to [0, 1] range
        img_array = img_array / 255.0

        # ImageNet normalization (standard for ResNet models)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std

        # Transpose to CHW format (channels first)
        img_array = np.transpose(img_array, (2, 0, 1))

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict(self, image: Image.Image) -> tuple[str, float]:
        """Run inference on an image.

        Args:
            image: PIL Image object

        Returns:
            Tuple of (predicted_class, confidence)
        """
        # Preprocess image
        input_data = self.preprocess_image(image)

        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )

        # Get predictions (logits)
        logits = outputs[0][0]  # Remove batch dimension

        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probabilities = exp_logits / np.sum(exp_logits)

        # Get predicted class and confidence
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx]
        predicted_class = self.class_labels[predicted_idx]

        return predicted_class, float(confidence)

    def predict_with_details(self, image: Image.Image) -> dict:
        """Run inference and return detailed results.

        Args:
            image: PIL Image object

        Returns:
            Dictionary with prediction details
        """
        # Preprocess image
        input_data = self.preprocess_image(image)

        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )

        # Get predictions (logits)
        logits = outputs[0][0]

        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        # Get predicted class
        predicted_idx = np.argmax(probabilities)
        predicted_class = self.class_labels[predicted_idx]
        confidence = probabilities[predicted_idx]

        return {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'probabilities': {
                label: float(prob)
                for label, prob in zip(self.class_labels, probabilities)
            },
            'provider': self.session.get_providers()[0]
        }

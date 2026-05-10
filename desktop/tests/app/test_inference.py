"""Unit tests for ONNX inference module."""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.app.inference import ONNXClassifier


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    # Create a 100x100 RGB image with random data
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return Image.fromarray(img_array, mode='RGB')


@pytest.fixture
def mock_onnx_session():
    """Mock ONNX Runtime session."""
    with patch('src.app.inference.ort.InferenceSession') as mock_session:
        # Mock session instance
        session_instance = MagicMock()
        mock_session.return_value = session_instance

        # Mock get_providers
        session_instance.get_providers.return_value = ['CPUExecutionProvider']

        # Mock get_inputs and get_outputs
        mock_input = MagicMock()
        mock_input.name = 'input'
        session_instance.get_inputs.return_value = [mock_input]

        mock_output = MagicMock()
        mock_output.name = 'output'
        session_instance.get_outputs.return_value = [mock_output]

        # Mock run method to return logits
        # Logits for binary classification: [not_gaming, gaming]
        session_instance.run.return_value = [np.array([[0.2, 0.8]])]

        yield mock_session, session_instance


class TestONNXClassifier:
    """Test suite for ONNXClassifier class."""

    def test_initialization_default_path(self, mock_onnx_session):
        """Test classifier initialization with default model path."""
        mock_session, _ = mock_onnx_session

        classifier = ONNXClassifier()

        # Check that session was created
        assert mock_session.called
        assert classifier.input_name == 'input'
        assert classifier.output_name == 'output'
        assert classifier.input_size == (224, 224)
        assert classifier.class_labels == ["Not Gaming", "Gaming"]

    def test_initialization_custom_path(self, mock_onnx_session):
        """Test classifier initialization with custom model path."""
        mock_session, _ = mock_onnx_session
        custom_path = "/custom/path/model.onnx"

        classifier = ONNXClassifier(model_path=custom_path)

        assert classifier.model_path == custom_path
        assert mock_session.called

    def test_initialization_gpu_disabled(self, mock_onnx_session):
        """Test classifier initialization with GPU disabled."""
        mock_session, _ = mock_onnx_session

        classifier = ONNXClassifier(use_gpu=False)

        # Check that only CPU provider was requested
        call_args = mock_session.call_args
        providers = call_args.kwargs['providers']
        assert providers == ['CPUExecutionProvider']

    @patch('src.app.inference.ort.get_available_providers')
    def test_get_execution_providers_with_cuda(self, mock_get_providers, mock_onnx_session):
        """Test execution provider selection with CUDA available."""
        mock_get_providers.return_value = [
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]

        classifier = ONNXClassifier(use_gpu=True)

        # Should prioritize CUDA
        call_args = mock_onnx_session[0].call_args
        providers = call_args.kwargs['providers']
        assert 'CUDAExecutionProvider' in providers
        assert 'CPUExecutionProvider' in providers

    @patch('src.app.inference.ort.get_available_providers')
    def test_get_execution_providers_with_coreml(self, mock_get_providers, mock_onnx_session):
        """Test execution provider selection with CoreML available."""
        mock_get_providers.return_value = [
            'CoreMLExecutionProvider',
            'CPUExecutionProvider'
        ]

        classifier = ONNXClassifier(use_gpu=True)

        # Should prioritize CoreML
        call_args = mock_onnx_session[0].call_args
        providers = call_args.kwargs['providers']
        assert 'CoreMLExecutionProvider' in providers
        assert 'CPUExecutionProvider' in providers

    def test_preprocess_image_shape(self, mock_onnx_session, sample_image):
        """Test that preprocessing produces correct output shape."""
        classifier = ONNXClassifier()

        preprocessed = classifier.preprocess_image(sample_image)

        # Should be (1, 3, 224, 224) - batch, channels, height, width
        assert preprocessed.shape == (1, 3, 224, 224)
        assert preprocessed.dtype == np.float32

    def test_preprocess_image_normalization(self, mock_onnx_session, sample_image):
        """Test that preprocessing applies correct normalization."""
        classifier = ONNXClassifier()

        preprocessed = classifier.preprocess_image(sample_image)

        # After ImageNet normalization, values should be roughly in [-3, 3] range
        assert preprocessed.min() >= -5.0
        assert preprocessed.max() <= 5.0

    def test_preprocess_image_rgb_conversion(self, mock_onnx_session):
        """Test that preprocessing converts non-RGB images to RGB."""
        classifier = ONNXClassifier()

        # Create a grayscale image
        gray_img = Image.new('L', (100, 100), color=128)

        preprocessed = classifier.preprocess_image(gray_img)

        # Should still produce 3 channels
        assert preprocessed.shape[1] == 3

    def test_predict_returns_correct_format(self, mock_onnx_session, sample_image):
        """Test that predict returns class and confidence."""
        classifier = ONNXClassifier()

        predicted_class, confidence = classifier.predict(sample_image)

        assert isinstance(predicted_class, str)
        assert predicted_class in ["Not Gaming", "Gaming"]
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_predict_gaming_class(self, mock_onnx_session, sample_image):
        """Test prediction when gaming class has higher probability."""
        mock_session, session_instance = mock_onnx_session
        # Set logits to favor gaming class
        session_instance.run.return_value = [np.array([[0.1, 0.9]])]

        classifier = ONNXClassifier()
        predicted_class, confidence = classifier.predict(sample_image)

        assert predicted_class == "Gaming"
        assert confidence > 0.5

    def test_predict_not_gaming_class(self, mock_onnx_session, sample_image):
        """Test prediction when not gaming class has higher probability."""
        mock_session, session_instance = mock_onnx_session
        # Set logits to favor not gaming class
        session_instance.run.return_value = [np.array([[0.9, 0.1]])]

        classifier = ONNXClassifier()
        predicted_class, confidence = classifier.predict(sample_image)

        assert predicted_class == "Not Gaming"
        assert confidence > 0.5

    def test_predict_with_details_structure(self, mock_onnx_session, sample_image):
        """Test that predict_with_details returns correct structure."""
        classifier = ONNXClassifier()

        result = classifier.predict_with_details(sample_image)

        assert isinstance(result, dict)
        assert 'predicted_class' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        assert 'provider' in result

    def test_predict_with_details_probabilities(self, mock_onnx_session, sample_image):
        """Test that probabilities sum to 1.0."""
        classifier = ONNXClassifier()

        result = classifier.predict_with_details(sample_image)

        probabilities = result['probabilities']
        assert len(probabilities) == 2
        assert 'Not Gaming' in probabilities
        assert 'Gaming' in probabilities

        # Probabilities should sum to approximately 1.0
        prob_sum = sum(probabilities.values())
        assert abs(prob_sum - 1.0) < 0.001

    def test_predict_with_details_provider_info(self, mock_onnx_session, sample_image):
        """Test that provider information is included."""
        classifier = ONNXClassifier()

        result = classifier.predict_with_details(sample_image)

        assert result['provider'] == 'CPUExecutionProvider'

    def test_softmax_numerical_stability(self, mock_onnx_session, sample_image):
        """Test that softmax handles large logit values without overflow."""
        mock_session, session_instance = mock_onnx_session
        # Set very large logits that could cause overflow
        session_instance.run.return_value = [np.array([[1000.0, 1001.0]])]

        classifier = ONNXClassifier()

        # Should not raise overflow error
        result = classifier.predict_with_details(sample_image)

        # Probabilities should still be valid
        assert 0.0 <= result['confidence'] <= 1.0
        prob_sum = sum(result['probabilities'].values())
        assert abs(prob_sum - 1.0) < 0.001

    def test_predict_calls_session_run(self, mock_onnx_session, sample_image):
        """Test that predict calls ONNX session run method."""
        mock_session, session_instance = mock_onnx_session

        classifier = ONNXClassifier()
        classifier.predict(sample_image)

        # Check that session.run was called
        assert session_instance.run.called

        # Check that it was called with correct input name
        call_args = session_instance.run.call_args
        input_dict = call_args[0][1]
        assert 'input' in input_dict

    def test_predict_with_different_image_sizes(self, mock_onnx_session):
        """Test that prediction works with various input image sizes."""
        classifier = ONNXClassifier()

        # Test with different sizes
        sizes = [(50, 50), (224, 224), (500, 300), (1920, 1080)]

        for width, height in sizes:
            img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            img = Image.fromarray(img_array, mode='RGB')

            # Should not raise error
            predicted_class, confidence = classifier.predict(img)
            assert isinstance(predicted_class, str)
            assert 0.0 <= confidence <= 1.0


class TestIntegration:
    """Integration tests requiring actual ONNX model."""

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.parent.parent.joinpath('ml/models/model.onnx').exists(),
        reason="ONNX model not found"
    )
    def test_real_model_inference(self, sample_image):
        """Test inference with actual ONNX model."""
        classifier = ONNXClassifier()

        # Should successfully run inference
        predicted_class, confidence = classifier.predict(sample_image)

        assert predicted_class in ["Not Gaming", "Gaming"]
        assert 0.0 <= confidence <= 1.0

    @pytest.mark.skipif(
        not Path(__file__).parent.parent.parent.parent.joinpath('ml/models/model.onnx').exists(),
        reason="ONNX model not found"
    )
    def test_real_model_detailed_prediction(self, sample_image):
        """Test detailed prediction with actual ONNX model."""
        classifier = ONNXClassifier()

        result = classifier.predict_with_details(sample_image)

        assert 'predicted_class' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        assert 'provider' in result

        # Check probabilities sum to 1
        prob_sum = sum(result['probabilities'].values())
        assert abs(prob_sum - 1.0) < 0.001

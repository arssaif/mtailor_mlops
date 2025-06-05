"""
model.py

Contains:
1. Preprocessor: loads an image from disk, converts to a NumPy array, and calls
   the same preprocessing steps that pytorch_model.py used (via its preprocess_numpy).
2. ONNXModel: loads the exported ONNX file (model.onnx) and runs inference via ONNX Runtime.
"""

import numpy as np
from PIL import Image
import onnxruntime as ort
from pytorch_model import Classifier, BasicBlock

class Preprocessor:
    """
    Wraps exactly the same preprocessing logic from pytorch_model.py.
    We load the raw image, convert to NumPy, then call that model's .preprocess_numpy(...) to get a
    (1,3,224,224) float32 array ready for ONNX.
    """
    def __init__(self):
        # Instantiate the PyTorch model class only for its preprocess method (no weights needed here)
        self._dummy_model = Classifier(BasicBlock, [2,2,2,2], num_classes=1000)
        # We do NOT load weights here; we only need the preprocess_numpy(...) function.

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Given an image file path, load it via PIL, convert to RGB, then call the original
        preprocess_numpy(...) method. That returns a Torch tensor of shape (3,224,224).
        We unsqueeze to (1,3,224,224) and return as a float32 NumPy array.
        """
        img = Image.open(image_path).convert("RGB")
        # Call pytorch_model's preprocess_numpy, which expects a PIL Image
        tensor = self._dummy_model.preprocess_numpy(img)  # shape (3,224,224), torch.float32
        tensor = tensor.unsqueeze(0)                      # shape (1,3,224,224)
        return tensor.cpu().numpy().astype(np.float32)


class ONNXModel:
    """
    Loads an ONNX model (model.onnx) via ONNX Runtime and performs inference.
    The ONNX expects input named "input" of shape (batch_size,3,224,224) float32, values 0-255.
    """
    def __init__(self, model_path: str, use_gpu: bool = True):
        providers = []
        if use_gpu:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(model_path, providers=providers)

    def predict(self, input_array: np.ndarray) -> np.ndarray:
        """
        input_array must be of shape (1,3,224,224) dtype float32 (0-255).
        Returns a (1,1000) float32 array of probabilities.
        """
        if input_array.dtype != np.float32:
            input_array = input_array.astype(np.float32)
        outputs = self.session.run(None, {"input": input_array})
        return outputs[0]

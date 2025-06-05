# convert_to_onnx.py

import argparse
import torch
from pytorch_model import Classifier, BasicBlock

class OnnxWrapper(torch.nn.Module):
    """
    Wraps the Classifier so that normalization and softmax are part of the ONNX graph.
    Assumes input x comes in as float32 with values in [0,255] and shape (N,3,224,224).
    """

    def __init__(self, base_model: torch.nn.Module):
        super(OnnxWrapper, self).__init__()
        self.model = base_model

        # Register mean and std as buffers so they are baked into the graph
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,3,224,224) float32 in [0,255]
        x = x / 255.0
        x = (x - self.mean) / self.std
        logits = self.model(x)
        probs = torch.nn.functional.softmax(logits, dim=1)
        return probs


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch Classifier to ONNX (with normalization + softmax)."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="pytorch_model_weights.pth",
        help="Path to the PyTorch .pth file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model.onnx",
        help="Where to save the ONNX model"
    )
    args = parser.parse_args()

    # 1. Instantiate your exact PyTorch model and load weights
    model = Classifier(BasicBlock, [2, 2, 2, 2], num_classes=1000)
    checkpoint = torch.load(args.weights, map_location="cpu")

    # If your .pth is a state_dict directly:
    if isinstance(checkpoint, dict) and not ("epoch" in checkpoint or "state_dict" in checkpoint):
        model.load_state_dict(checkpoint)
    # If checkpoint has a 'state_dict' key:
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Fallback: assume checkpoint is exactly the state_dict
        model.load_state_dict(checkpoint)

    model.eval()

    # 2. Wrap the model so normalization & softmax live inside ONNX
    wrapped = OnnxWrapper(model)
    wrapped.eval()

    # 3. Create a dummy input tensor (batch=1, 3×224×224) of zeros in [0,255]
    dummy_input = torch.zeros((1, 3, 224, 224), dtype=torch.float32)

    # 4. Export to ONNX
    torch.onnx.export(
        wrapped,
        dummy_input,
        args.output,
        input_names=["input"],
        output_names=["probabilities"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "probabilities": {0: "batch_size"}
        },
        opset_version=12,
        do_constant_folding=True
    )

    print(f"ONNX model saved to {args.output}")

if __name__ == "__main__":
    main()

"""
test.py

Locally validate that model.onnx produces the same top-1 predictions on your sample images
that the original PyTorch + preprocess_numpy pipeline does.
"""

import argparse
import numpy as np
from pytorch_model import Classifier, BasicBlock
from model import Preprocessor, ONNXModel
import torch

# Two sample images provided in the original repo:
#  - n01440764_tench.jpeg (class ID 0)
#  - n01667114_mud_turtle.JPEG (class ID 35)
DEFAULT_SAMPLES = [
    "samples/n01440764_tench.jpeg",
    "samples/n01667114_mud_turtle.JPEG"
]

def main():
    parser = argparse.ArgumentParser(description="Run local tests of ONNX vs PyTorch on sample images.")
    parser.add_argument("--onnx", type=str, default="model.onnx", help="Path to the ONNX model file")
    parser.add_argument("--images", nargs="+", default=DEFAULT_SAMPLES,
                        help="List of image paths to test (defaults to the two sample images)")
    args = parser.parse_args()

    # 1. Load the original PyTorch model for reference
    pytorch_model = Classifier(BasicBlock, [2, 2, 2, 2], num_classes=1000)
    checkpoint = torch.load("pytorch_model_weights.pth", map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        pytorch_model.load_state_dict(checkpoint["state_dict"])
    else:
        pytorch_model.load_state_dict(checkpoint)
    pytorch_model.eval()

    pre = Preprocessor()
    onnx_m = ONNXModel(args.onnx, use_gpu=False)

    for img_path in args.images:
        # a) PyTorch pipeline
        img_np = pre.preprocess_image(img_path)   # shape (1,3,224,224)
        img_tensor = torch.from_numpy(img_np)      # convert to torch tensor
        with torch.no_grad():
            logits = pytorch_model(img_tensor)
            probs_pt = torch.nn.functional.softmax(logits, dim=1).cpu().numpy().squeeze()
        top1_pt = int(np.argmax(probs_pt))
        # b) ONNX inference
        # Note: Our ONNX graph expects raw (0-255) input, but Preprocessor already normalized from 0-255 -> 0-1 -> normalized.
        # In this test, we want to show that ONNX’s internal preprocessing matches. So we have to pass the raw RGB → np.uint8
        # back into ONNXModel. But our ONNXModel code expects the array to be “0-255 float32” so we must bypass Preprocessor’s normalize.
        # Instead, we re-load the raw image as float32(0-255).
        from PIL import Image

        raw = Image.open(img_path).convert("RGB").resize((224,224), Image.BILINEAR)
        raw_np = np.array(raw, dtype=np.float32)  # shape (224,224,3)
        # reorder to (1,3,224,224)
        raw_np = np.transpose(raw_np, (2,0,1))[None, ...]
        probs_onnx = onnx_m.predict(raw_np).squeeze()
        top1_onnx = int(np.argmax(probs_onnx))

        print(f"Image: {img_path}")
        print(f"  → PyTorch top-1  : {top1_pt}")
        print(f"  → ONNX (export) top-1: {top1_onnx}")
        print("  " + ("✅ match" if top1_pt == top1_onnx else "❌ mismatch"))

if __name__ == "__main__":
    main()

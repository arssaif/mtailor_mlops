"""
test_server.py

Sends HTTP requests to your Cerebrium‐deployed endpoint and verifies that it matches local ONNX.

Usage examples:

  # Single image prediction
  python test_server.py \
    --api-url https://.../predict \
    --api-key <YOUR_JWT_TOKEN> \
    --image samples/n01440764_tench.jpeg

  # Built‐in test against the two sample images
  python test_server.py \
    --api-url https://.../predict \
    --api-key <YOUR_JWT_TOKEN> \
    --test
"""

import argparse
import requests
import numpy as np
from model import Preprocessor, ONNXModel

# Two sample images
DEFAULT_SAMPLES = [
    "samples/n01440764_tench.jpeg",
    "samples/n01667114_mud_turtle.JPEG"
]

# Load ImageNet class names (optional). If unavailable, we show "Class <idx>".
CLASS_NAMES = None
try:
    with open("imagenet_classes.txt", "r") as f:
        CLASS_NAMES = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    CLASS_NAMES = None

def get_top(probs: np.ndarray):
    idx = int(np.argmax(probs))
    prob = float(probs[idx])
    name = CLASS_NAMES[idx] if CLASS_NAMES else f"Class {idx}"
    return idx, name, prob

def main():
    parser = argparse.ArgumentParser(description="Test the deployed ONNX model on Cerebrium.")
    parser.add_argument("--api-url", type=str, required=True,
                        help="Full URL to your /predict endpoint (e.g. https://…/predict)")
    parser.add_argument("--api-key", type=str, required=True,
                        help="Your Bearer token (JWT) to authenticate with Cerebrium")
    parser.add_argument("--image", type=str, help="Path to a single image (for one‐off test)")
    parser.add_argument("--test", action="store_true",
                        help="If set, run built‐in tests against the two sample images")

    args = parser.parse_args()
    headers = {"Authorization": f"Bearer {args.api_key}"}

    if args.test:
        print("Running built‐in test mode against sample images…")
        # Compare local ONNX vs remote for each sample image
        onnx_m = ONNXModel("model.onnx", use_gpu=False)
        Preprocessor()
        all_match = True

        for img_path in DEFAULT_SAMPLES:
            # 1) Local ONNX prediction
            raw = __import__("PIL").Image.open(img_path).convert("RGB").resize((224,224), __import__("PIL").Image.BILINEAR)
            raw_np = np.array(raw, dtype=np.float32)
            raw_np = np.transpose(raw_np, (2,0,1))[None, ...]
            local_probs = onnx_m.predict(raw_np).squeeze()
            local_idx, local_name, _ = get_top(local_probs)

            # 2) Remote prediction
            files = {"file": open(img_path, "rb")}
            resp = requests.post(args.api_url, headers=headers, files=files)
            if resp.status_code != 200:
                print(f"ERROR: HTTP {resp.status_code} for {img_path}: {resp.text}")
                all_match = False
                continue
            result = resp.json()
            remote_probs = np.array(result.get("probabilities", []), dtype=np.float32)
            remote_idx, remote_name, _ = get_top(remote_probs)

            print(f"Image: {img_path}")
            print(f"  • LOCAL ONNX top‐1  = {local_idx} ({local_name})")
            print(f"  • REMOTE top‐1      = {remote_idx} ({remote_name})")
            if local_idx != remote_idx:
                print("  ❌ MISMATCH")
                all_match = False
            else:
                print("  ✅ match")

        if all_match:
            print("\n✅ All sample‐image predictions matched!")
        else:
            print("\n❌ Some predictions did NOT match.")
    else:
        # Single‐image mode
        if not args.image:
            parser.error("--image <path> is required in single‐image mode.")
        files = {"file": open(args.image, "rb")}
        print('Uploading image ......')
        resp = requests.post(args.api_url, headers=headers, files=files)
        if resp.status_code != 200:
            print(f"ERROR: HTTP {resp.status_code}\n{resp.text}")
            return
        print('Predicting....')
        result = resp.json()
        probs = np.array(result.get("probabilities", []), dtype=np.float32)
        idx, name, prob = get_top(probs)
        print(f"Prediction for {args.image}: {idx} ({name}) with probability {prob:.4f}")

if __name__ == "__main__":
    main()

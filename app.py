"""
app.py

A FastAPI app that loads model.onnx at startup and serves /predict, /health, /ready.
Cerebrium’s Docker‐based deployment will launch this via Uvicorn.
"""

from fastapi import FastAPI, File, UploadFile
import numpy as np
import io
from PIL import Image
from model import ONNXModel

app = FastAPI()

# Load the ONNX model once at startup.  model.onnx should be in the same folder.
onnx_model = ONNXModel("model.onnx", use_gpu=True)

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/ready")
async def ready():
    return {"status": "ready"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Expects a file upload (image). Returns JSON with a 1000‐length 'probabilities' list.
    """
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Resize to 224×224
    image = image.resize((224, 224), Image.BILINEAR)
    arr = np.array(image, dtype=np.float32)        # shape (224,224,3)
    arr = np.transpose(arr, (2, 0, 1))[None, ...]  # shape (1,3,224,224), dtype float32, [0,255]
    probs = onnx_model.predict(arr).squeeze()      # (1000,)

    return {"probabilities": probs.tolist()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
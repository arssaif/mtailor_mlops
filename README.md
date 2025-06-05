# mtailor_mlops

## Image Classification → ONNX → Cerebrium Deployment
This repository shows how to convert a PyTorch image classification model (trained on ImageNet) to ONNX format and deploy it on Cerebrium’s serverless GPU platform using a custom Docker image. The model accepts 224×224 RGB images and returns a probability vector over 1000 ImageNet classes.

## Prerequisites
- Python 3.10
- Docker (to build and run the container locally)
- Cerebrium CLI (install via pip install cerebrium) and a valid Cerebrium account with an API key

## 1. Install Python Dependencies
Install all required packages:
```bash
pip install -r requirements.txt
```

## 2. (Optional - To produce/utilize model.onnx) Convert the PyTorch Model to ONNX
-   Convert the PyTorch Model to ONNX
    Run:
    ```bash
    python convert_to_onnx.py \
      --weights pytorch_model_weights.pth \
      --output model.onnx
    ```
    Successful output:
    ```css
    ONNX model saved to model.onnx
    The resulting model.onnx is self-contained and can be used for inference without requiring PyTorch or the original weights.
    ```
-  Local ONNX Testing
    Before deployment, verify that model.onnx produces the same top-1 predictions as the original PyTorch model on the two sample images.
    
    Ensure samples/n01440764_tench.jpeg and samples/n01667114_mud_turtle.JPEG exist.
    
    Run:
    ```bash
    python test.py
    ```
    Expected output:
    
    ```yaml
    Image: samples/n01440764_tench.jpeg
      → PyTorch top-1  : 0
      → ONNX (export) top-1: 0
      ✅ match
    
    Image: samples/n01667114_mud_turtle.JPEG
      → PyTorch top-1  : 35
      → ONNX (export) top-1: 35
      ✅ match
    If both lines show “✅ match”, the conversion is correct. To test different images, pass them via --images:
    ```
    ```bash
    python test.py --images path/to/image1.jpg path/to/image2.jpg
    ```
## 3. Running the API Locally
    Start the FastAPI server:
    ```bash
    python app.py
    ```
    The API will be available at http://localhost:8080. You can access the interactive API documentation at http://localhost:8080/docs.
## 4. Build and Run the Docker Image Locally
-  Build the image:

    ```bash
    docker build -t cerebrium-image-classifier .
    ```
    The Dockerfile installs dependencies and copies all code into the container.
    Run the container in detached mode on port 8192:
    ```bash
    docker run -d --name local_test -p 8192:8192 cerebrium-image-classifier
    ```
-  Check the health endpoint:
  
    Using curl:
    ```bash
    curl -f http://localhost:8192/health
    ```
    Using endpoint
    ```bash
    http://<your_host>:8192/docs#/default/health_health_get
    ```
    Response:
    ```json
    {"status":"healthy"}
    ```
-  Test inference with a sample image:
    
    Using curl:
    ```bash
    curl -X POST -F file=@samples/n01440764_tench.jpeg http://localhost:8192/predict \
      | jq '.probabilities[0:5]'
    ```
    Using endpoint
    ```bash
    http://<your_host>:8192/docs#/default/predict_predict_post)
    ```
    You should receive a JSON array of length 1000. The top-1 for n01440764_tench.jpeg is index 0.

-  Stop and remove the container:
    ```bash
    docker stop local_test; docker rm local_test
    ```
## 5. Deploy to Cerebrium
Login to Cerebrium:

```bash
cerebrium login
```
Follow browser instructions to authenticate.

Deploy:

```bash
cerebrium deploy
```
Cerebrium will:

- Build the Docker image (using Dockerfile)

- Push it to Cerebrium’s registry

- Launch a serverless GPU instance per cerebrium.toml

- After deployment, you will see the endpoint URL for your inference API. For example:

```bash
https://api.cortex.cerebrium.ai/v4/p-<project-id>/mtailor-mlops-classifier/predict
```
API Key: Obtain your API key from the Cerebrium dashboard under “API Keys”. You need this to call the endpoint.

## 6. Test the Deployed Endpoint
Use test_server.py to verify that remote inference matches local ONNX:

- Single‐image Mode:

  ```bash
  python test_server.py \
    --api-url https://api.cortex.cerebrium.ai/v4/p-e9164d52/mtailor-mlops-classifier/predict \
    --api-key <your_api_key> \
    --image <path_to_you_image.jpeg>
  ```
  Output:
  
  ```bash
  Uploading image ......
  Predicting....
  Prediction for samples/n01440764_tench.jpeg: 0 (Class 0) with probability 0.8267
  ```
- Built‐in Test Mode (compares both samples):

```bash
python test_server.py \
    --api-url --api-url https://api.cortex.cerebrium.ai/v4/p-e9164d52/mtailor-mlops-classifier/predict \
    --api-key <your_api_key> \
    --test
```
You should see something like:

```sql
Running built‐in test mode against sample images…
Image: samples/n01440764_tench.jpeg
  • LOCAL ONNX top‐1  = 0 (n01440764 tench)
  • REMOTE top‐1      = 0 (n01440764 tench)
  ✅ match
Image: samples/n01667114_mud_turtle.JPEG
  • LOCAL ONNX top‐1  = 35 (n01667114 mud turtle)
  • REMOTE top‐1      = 35 (n01667114 mud turtle)
  ✅ match

✅ All sample‐image predictions matched!
```
If any prediction mismatches, double‐check that:

The request is hitting /predict on the correct URL with the correct api-key.

## 7. Cerebrium Configuration
This project is deployed on Cerebrium, a MLOps platform for easy model deployment.

Deployment Details
- Project ID: p-e9164d52
- API Endpoint: https://api.cortex.cerebrium.ai/v4/p-e9164d52/mtailor-mlops-classifier/predict
- Authentication: Bearer token required in Authorization header
Testing the Deployed Model
You can test the deployed model using the e2e_test.py script, which sends a test image to the Cerebrium endpoint and verifies the response.

Making API Calls to the Deployed Model
Using curl:
```bash
curl -X POST   https://api.cortex.cerebrium.ai/v4/p-e9164d52/mtailor-mlops-classifier/predict  -H 'Authorization: Bearer <your_api_key>' -F  'file=@<path_to_you_image.jpeg>'
```
Replace YOUR_TOKEN_HERE with your actual Cerebrium API token and path/to/your/image.jpeg with the path to the image you want to classify.

To check health of deployed model:
```bash
curl -X GET   https://api.cortex.cerebrium.ai/v4/p-e9164d52/mtailor-mlops-classifier/health  -H 'Authorization: Bearer <your_api_key>'
```

## 8. Continuous Integration (Optional)
A GitHub Actions workflow (.github/workflows/ci.yml) automates building, testing, and packaging:

- Checkout code

- Set up Python 3.10

- Install dependencies

- Download PyTorch weights

- Convert to ONNX (convert_to_onnx.py)

- Run local tests (test.py)

- Build Docker image (docker build)

- Run container locally and test /health and /predict

- This pipeline ensures that any new commit is validated end to end.
##
Follow these steps in order to convert, test, and deploy your model. Once deployed, you will have a scalable, serverless GPU inference endpoint on Cerebrium that classifies any image in about 2–3 seconds.

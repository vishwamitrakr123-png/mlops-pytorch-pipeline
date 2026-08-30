# MLOps PyTorch Pipeline on Kubernetes

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)
![Kubernetes](https://img.shields.io/badge/Kubernetes-1.25%2B-326ce5)
![Docker](https://img.shields.io/badge/Docker-20.10%2B-2496ed)

An end-to-end, production-style MLOps pipeline that automates model training on the CIFAR-10 dataset using PyTorch, state persistence via Kubernetes PersistentVolumeClaims (PVC), and containerized inference serving using FastAPI on a local Kubernetes cluster.

---

## 🏗️ System Architecture

The pipeline operates entirely within an isolated `ml-training` Kubernetes namespace. It uses a decoupled decoupled architecture where training and inference are separate microservices linked by a shared storage layer:

+-----------------------------------------------------------------------------------+
| Kubernetes Cluster (ml-training Namespace)                                      |
|                                                                                   |
|  +---------------------------+                   +-----------------------------+  |
|  | PyTorch Training Job      |                   | FastAPI Serving Deployment  |  |
|  | (10 Epochs on CIFAR-10)   |                   | (mlops-serve:v1)          |  |
|  +-------------+-------------+                   +--------------+--------------+  |
|                |                                                |                 |
|                | Writes Checkpoint                              | Reads Model     |
|                v                                                v                 |
|  +-----------------------------------------------------------------------------+  |
|  | PersistentVolumeClaim (mlops-pvc -> /app/checkpoints/classifier_v1.pt)   |  |
|  +-----------------------------------------------------------------------------+  |
|                                                                 ^                 |
|                                                                 | TargetPort: 8080|
|                                                  +--------------+--------------+  |
|                                                  | Service (model-serving)   |  |
|                                                  | Port: 80                    |  |
|                                                  +--------------+--------------+  |
+-----------------------------------------------------------------|-----------------+
|
kubectl port-forward| (8080:80)
v
+--------------------------------+
| Local Client / cURL Validation |
+--------------------------------+


---

## 💡 Pipeline Components & Workflow

1. **Storage Layer (`mlops-pvc`):** A persistent volume claim allocating shared storage inside the cluster to persist trained weights across pod restarts.
2. **Training Execution (`pytorch-training-job`):** A Kubernetes `Job` that executes `src/train.py`. It downloads CIFAR-10, trains a PyTorch convolutional network over 10 epochs, and outputs the model binary to `/app/checkpoints/classifier_v1.pt`.
3. **Inference Service (`model-serving`):** A FastAPI server deployed with 1 replica running Uvicorn on port `8080`. On boot, it loads the model weights from the PVC and handles HTTP POST requests.
4. **Networking (`model-serving` Service):** A `ClusterIP` service exposing internal port `80` routed to container `targetPort: 8080`.

---

## 📁 Repository Structure

```text
.
├── k8s/
│   ├── namespace.yaml           # Isolated ml-training namespace
│   ├── pvc.yaml                 # PersistentVolumeClaim storage definition
│   ├── configmap.yaml           # Hyperparameters and runtime variables
│   ├── training-job.yaml        # PyTorch CIFAR-10 training job manifest
│   ├── serving-deployment.yaml  # FastAPI serving deployment manifest
│   └── serving-service.yaml     # ClusterIP service mapping (80 -> 8080)
├── src/
│   ├── train.py                 # Training script with PyTorch pipeline
│   └── app.py                   # FastAPI REST API implementation
├── tests/
│   └── test_unit.py             # Pipeline unit tests for CI validation
├── Dockerfile                   # Container definition for serving app
└── README.md                    # Project documentation
🚀 Step-by-Step Setup & Deployment
Prerequisites
Docker Desktop or Minikube with Kubernetes enabled

kubectl CLI installed and configured

Python 3.10+ (for local development)

Step 1: Initialize Namespace & Storage
Bash
# Create the ml-training namespace
kubectl apply -f k8s/namespace.yaml

# Create storage allocations and hyperparameter configmaps
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/configmap.yaml
Step 2: Build and Load Docker Image
Bash
# Build the FastAPI inference container
docker build -t mlops-serve:v1 .

# If using Minikube or Kind:
minikube image load mlops-serve:v1
Step 3: Run Training Job
Bash
# Submit the PyTorch training job
kubectl apply -f k8s/training-job.yaml

# Watch pod status until job completion
kubectl get pods -n ml-training -w
Step 4: Deploy Model Serving API
Bash
# Deploy the FastAPI serving instance and ClusterIP service
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml

# Verify pod readiness
kubectl get pods -n ml-training
🧪 API Validation & Testing
Expose the internal ClusterIP service to your local machine:

Bash
kubectl port-forward svc/model-serving 8080:80 -n ml-training --address 127.0.0.1
In a second terminal window, perform API sanity checks:

Health Check Endpoint
Bash
curl -X GET [http://127.0.0.1:8080/health](http://127.0.0.1:8080/health)
Response:

JSON
{"status": "healthy"}
Inference Endpoint
Bash
curl -X POST [http://127.0.0.1:8080/predict](http://127.0.0.1:8080/predict) \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@test_image.png"
Response:

JSON
{
  "probabilities": [0.2978, 0.0135, 0.0086, 0.3808, 0.0051, 0.2783, 0.0057, 0.0030, 0.0017, 0.0050],
  "predicted_class": 3
}
🔧 Key Engineering Learnings & Troubleshooting
Container Port Misalignment: Initial deployment resulted in connection refused on /predict. Inspection via kubectl logs revealed Uvicorn binding to container port 8080. Updating the serving-service.yaml manifest's targetPort from 8000 to 8080 resolved internal traffic routing.

Volume Mounting: Using Kubernetes PersistentVolumeClaims allowed decoupling training jobs from serving pods, ensuring model artifact persistence without re-training on pod restarts.
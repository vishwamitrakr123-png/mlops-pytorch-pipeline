\# MLOps PyTorch Pipeline on Kubernetes



!\[Python Version](https://img.shields.io/badge/python-3.10%2B-blue)

!\[PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)

!\[FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)

!\[Kubernetes](https://img.shields.io/badge/Kubernetes-1.25%2B-326ce5)



An end-to-end MLOps pipeline automating model training on the CIFAR-10 dataset using PyTorch, state persistence via Kubernetes PersistentVolumeClaims (PVC), and containerized inference serving using FastAPI deployed on a local Kubernetes cluster.



\---



\## 🏗️ System Architecture



The pipeline consists of two primary operational phases inside the `ml-training` Kubernetes namespace:



1\. \*\*Training Pipeline (`pytorch-training-job`):\*\* Executes PyTorch training over 10 epochs on CIFAR-10, saving the resulting weights (`classifier\_v1.pt`) directly to a shared Persistent Volume (`mlops-pvc`).

2\. \*\*Inference Pipeline (`model-serving`):\*\* A FastAPI server (`mlops-serve:v1`) running on Uvicorn (port 8080), mounted to `mlops-pvc` to load trained model artifacts and serve prediction endpoints exposed via a Kubernetes ClusterIP Service.


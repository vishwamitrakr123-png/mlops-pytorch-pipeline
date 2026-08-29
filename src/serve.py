import io
import os
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
import torch
import torch.nn.functional as F
from src.dataset import get_transforms
from src.model import get_model

app = FastAPI()

MODEL_PATH = os.getenv("MODEL_PATH", "/app/checkpoints/classifier_v1.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None

def load_checkpoint():
    global model
    path = Path(MODEL_PATH)
    if path.exists():
        model = get_model("resnet18", 10)
        checkpoint = torch.load(path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(DEVICE)
        model.eval()

@app.on_event("startup")
def startup():
    load_checkpoint()

@app.get("/health")
def health():
    if model is None:
        load_checkpoint()
    if model is None:
        raise HTTPException(status_code=503, detail="Model standard runtime unavailable")
    return {"status": "healthy"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    transform = get_transforms(train=False)
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1).squeeze().tolist()

    return {"probabilities": probs, "predicted_class": int(torch.argmax(outputs, dim=1).item())}
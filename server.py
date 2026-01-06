"""
Simple server example
"""

import os
import requests
import torch
from json import load, dump
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from torch import inference_mode
from torch import softmax, squeeze
from torchvision.transforms import Compose, Resize, ToTensor
from model.vit import ViT 
 
MODELS = [
    "ViT-B-16p-Imagenet1k",
    "ViT-B-32p-Imagenet1k",
    "ViT-L-16p-Imagenet1k",
    "ViT-L-32p-Imagenet1k"
]

application = FastAPI(
    title="ViT-Imagenet1k",
    description="Server example of pretrained ViT"
)

model = ViT.build(MODELS[0])
model.eval()

transform = Compose([
    Resize(model.image_size),
    ToTensor()
])
  
filename = "labels.txt" 

if os.path.exists(filename):
    with open(filename, "r") as file:
        labels = load(file)
else:
    try:
        url = "https://huggingface.co/eric-hermosis/ViT-Imagenet1k/resolve/main/labels.txt"
        response = requests.get(url)
        response.raise_for_status()
        labels = response.json()
        with open(filename, "w") as file:
            dump(labels, file)
    except requests.RequestException:
        raise RuntimeError("Unable to load labels from local file or Hugging Face URL")

labels = [labels[str(index)] for index in range(len(labels))] 

@application.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
 
    bytes = await file.read()
    image = Image.open(BytesIO(bytes)).convert('RGB')
    input = transform(image)
 
    with inference_mode():
        output = squeeze(model(input), 0)
        topk   = torch.topk(softmax(output, dim=-1) , k=3)

    results = []
    for index, probability in zip(topk.indices.tolist(), topk.values.tolist()):
        results.append({"index": index, "label": labels[index], "probability": round(probability*100, 2)})
    return JSONResponse({"predictions": results})

if __name__ == "__main__":
    from uvicorn import run
    run(application, host="0.0.0.0", port=8000)
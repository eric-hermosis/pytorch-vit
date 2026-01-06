# Pretrained ViT-Imagenet1k

This repository provides a PyTorch implementation of a **Vision Transformer (ViT)** pretrained on **ImageNet-1k**, along with examples for both **notebook inference** and a simple **FastAPI server**.

---

## Table of Contents

* [Features](#features)
* [Installation](#installation)
* [Available Models](#available-models)
* [Manual Weights and Labels Download](#manual-weights-and-labels-download)
* [Notebook Example](#notebook-example)
* [FastAPI Server Example](#fastapi-server-example)
* [License](#license)

---

## Features

* Pretrained ViT models: small (B) and large (L) variants with patch sizes 16×16 and 32×32.
* Easy-to-use `ViT.build(name)` API for model initialization.
* Support for both single-image inference and batch processing.
* Automatic or manual download of model weights and ImageNet labels.
* FastAPI server for image classification via HTTP requests.

---

## Installation

```bash
git clone https://github.com/eric-hermosis/pytorch-vit.git
cd pytorch-vit
pip install -r requirements.txt
```

`requirements.txt` should include necessary packages for running the examples, but they are not necessary for the model.

```
torch
torchvision
Pillow
fastapi
uvicorn
requests
```

---

## Available Models

| Model Name             | Patch Size | Image Size | Model Dimension | Hidden Dimension | # Layers | # Heads | # Classes |
| ---------------------- | ---------- | ---------- | --------- | ---------- | -------- | ------- | --------- |
| `ViT-B-16p-Imagenet1k` | [16,16]    | [384,384]  | 768       | 3072       | 12       | 12      | 1000      |
| `ViT-B-32p-Imagenet1k` | [32,32]    | [384,384]  | 768       | 3072       | 12       | 12      | 1000      |
| `ViT-L-16p-Imagenet1k` | [16,16]    | [384,384]  | 1024      | 4096       | 24       | 16      | 1000      |
| `ViT-L-32p-Imagenet1k` | [32,32]    | [384,384]  | 1024      | 4096       | 24       | 16      | 1000      |

---

## Manual Weights and Labels Download

If automatic downloading fails, you can manually download weights and labels:

1. **Weights**

Download `.pth` files from the Hugging Face repository:

```
https://huggingface.co/eric-hermosis/ViT-Imagenet1k/tree/main
```

Place the files in your project directory. Then load them manually, for example:

```python
vit = ViT(**settings.dump())
vit.initialize("ViT-B-16p-Imagenet1k", path="./ViT-B-16p-Imagenet1k.pth")
```

2. **Labels**

Download `labels.txt` from:

```
https://huggingface.co/eric-hermosis/ViT-Imagenet1k/resolve/main/labels.txt
```

---

## Notebook Example

For an image like the one in the repo:

![Panda](image.png)

You can create a script like this:

```python
import os
import requests
import torch
from json import load, dump 
from PIL import Image
from torch import inference_mode, softmax, squeeze
from torchvision.transforms import Compose, Resize, ToTensor
from model.vit import ViT

MODELS = [
    "ViT-B-16p-Imagenet1k",
    "ViT-B-32p-Imagenet1k",
    "ViT-L-16p-Imagenet1k",
    "ViT-L-32p-Imagenet1k"
]
 
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
    url = "https://huggingface.co/eric-hermosis/ViT-Imagenet1k/resolve/main/labels.txt"
    response = requests.get(url)
    response.raise_for_status()
    labels = response.json()
    with open(filename, "w") as file:
        dump(labels, file) 

labels = [labels[str(index)] for index in range(len(labels))]


def predict(image_path, topk=3):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image)

    with inference_mode():
        output = squeeze(model(input_tensor), 0)
        topk = torch.topk(softmax(output, dim=-1), k=topk)

    results = []
    for index, probability in zip(topk.indices.tolist(), topk.values.tolist()):
        results.append({"index": index, "label": labels[index], "probability": round(probability*100, 2)})
    return results

predictions = predict( "image.png")
for prediction in predictions:
    print(prediction['label'], ", ", prediction['probability'], "%")
```

Should give something like:

```
giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca ,  99.74 %
lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens ,  0.13 %
slh bear, Melursus ursinus, Ursus ursinus ,  0.02 %
```

---

## FastAPI Server Example

```bash
uvicorn server:application --host 0.0.0.0 --port 8000
```

**Server endpoint:**

* `POST /predict` – upload an image file and get top-3 predictions.

**Example using `curl`:**

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@example.jpg"
```

**Response:**

```json
{
  "predictions": [
    {
      "index": 388,
      "label": "giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca",
      "probability": 99.79
    },
    {
      "index": 387,
      "label": "lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens",
      "probability": 0.07
    },
    {
      "index": 297,
      "label": "sloth bear, Melursus ursinus, Ursus ursinus",
      "probability": 0.02
    }
  ]
}
```

---

## License

This project is licensed under the APACHE-2.0 License.
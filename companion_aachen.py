import os
import torch
from torchvision import models, transforms
from PIL import Image
import wikipedia
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "checkpoints/aachen_resnet50_epoch10.pth"
DATA_DIR = "data/aachen"

# Load class names dynamically
class_names = sorted(os.listdir(DATA_DIR))

# Preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ----------------------------
# Load model once at startup
# ----------------------------
num_classes = len(class_names)
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ----------------------------
# Utils
# ----------------------------
def recognize_landmark(image_path: str):
    img = Image.open(image_path).convert("RGB")
    inp = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(inp)
        pred_idx = logits.argmax(dim=1).item()
    return class_names[pred_idx]

def fetch_wikipedia_summary(landmark: str, sentences: int = 3):
    try:
        return wikipedia.summary(landmark, sentences=sentences, auto_suggest=True, redirect=True)
    except:
        return f"Sorry, I couldn’t find detailed info on {landmark}."

def generate_story(landmark: str, context: str):
    """
    Placeholder story generator.
    Later, replace with Qwen or GPT model call.
    """
    return (
        f"Welcome to {landmark}! "
        f"Here's something interesting: {context} "
        "Imagine walking around here — every stone has a story!"
    )

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="Travel Companion - Aachen")

@app.post("/identify_landmark/")
async def identify_landmark(file: UploadFile = File(...)):
    temp_path = "temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Step 1: classify landmark
    landmark = recognize_landmark(temp_path)

    # Step 2: fetch info
    summary = fetch_wikipedia_summary(landmark)

    # Step 3: generate story
    story = generate_story(landmark, summary)

    return JSONResponse(content={
        "landmark": landmark,
        "summary": summary,
        "story": story
    })

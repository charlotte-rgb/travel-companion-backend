import os
import re
import torch
from torchvision import models, transforms
from PIL import Image
from geopy.distance import geodesic
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from transformers import pipeline
import wikipedia

# ----------------------------
# Config
# ----------------------------
DATA_DIR = "data/aachen"
MODEL_PATH = "checkpoints/aachen_resnet50_epoch10.pth"

landmarks = [
    {"name": "Aachen Cathedral", "lat": 50.774444, "lon": 6.083611},
    {"name": "Aachen Town Hall", "lat": 50.7753, "lon": 6.0839},
    {"name": "Elisenbrunnen", "lat": 50.7761, "lon": 6.0882},
]

# ----------------------------
# Image recognition setup
# ----------------------------
class_names = sorted(os.listdir(DATA_DIR))
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

num_classes = len(class_names)
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

def recognize_landmark(image_path: str):
    img = Image.open(image_path).convert("RGB")
    inp = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(inp)
        pred_idx = logits.argmax(dim=1).item()
    return class_names[pred_idx]

# ----------------------------
# Story generation setup
# ----------------------------
story_gen = pipeline("text-generation", model="Qwen/Qwen2.5-3B-Instruct", device=0)

def fetch_summary(landmark: str) -> str:
    try:
        return wikipedia.summary(landmark, sentences=3, auto_suggest=True)
    except Exception:
        return f"No Wikipedia summary available for {landmark}."

def clean_story(text: str) -> str:
    """Remove prompt echoes, tokens, and instructions."""
    text = re.sub(r"<\|im.*?\|>", "", text)  # remove tokens like <|im_end|>
    text = re.sub(r"(?i)(only output.*|do not repeat.*)", "", text)
    text = re.sub(r"(?i)story\s*:", "", text)
    text = text.replace("system", "").replace("user", "").replace("assistant", "")
    return text.strip(" .'\"\n")

def generate_story(landmark: str, summary: str) -> str:
    messages = [
        {"role": "system", "content": "You are a friendly travel companion."},
        {"role": "user", "content": f"""
The user is visiting {landmark}.
Here is factual context: {summary}

Write a short, lively travel story (100-150 words), conversational, spoken style.
Do not repeat these instructions, do not output 'Story:', just give the story text.
"""}
    ]
    prompt = story_gen.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    out = story_gen(
        prompt,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.8,
        pad_token_id=story_gen.tokenizer.eos_token_id
    )
    raw = out[0]["generated_text"]

    # ✅ Keep only after "assistant" if present
    if "assistant" in raw:
        raw = raw.split("assistant", 1)[-1]

    return clean_story(raw)

# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="Travel Companion with Clean Stories")

# 🌍 HTML demo page
@app.get("/", response_class=HTMLResponse)
def home_page():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Travel Companion Demo</title></head>
    <body>
      <h2>Travel Companion Demo</h2>
      <p>This will grab your GPS (with permission) and optionally let you upload a photo.</p>
      <form id="uploadForm" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*"><br><br>
        <button type="button" onclick="sendData()">Send My Location + Photo</button>
      </form>
      <p><strong>Detected Location:</strong></p>
      <p id="coords">Waiting for GPS...</p>
      <pre id="result"></pre>
      <script>
        let currentLat = null, currentLon = null;
        if (navigator.geolocation) {
          navigator.geolocation.getCurrentPosition(pos => {
            currentLat = pos.coords.latitude;
            currentLon = pos.coords.longitude;
            document.getElementById("coords").textContent =
              "Lat: " + currentLat.toFixed(6) + ", Lon: " + currentLon.toFixed(6);
          }, err => {
            document.getElementById("coords").textContent = "Error: " + err.message;
          });
        }
        function sendData() {
          if (!currentLat || !currentLon) {
            alert("GPS not available yet.");
            return;
          }
          const form = document.getElementById("uploadForm");
          const formData = new FormData(form);
          formData.append("lat", currentLat);
          formData.append("lon", currentLon);
          fetch("/detect_landmark/", { method: "POST", body: formData })
            .then(r => r.json())
            .then(data => {
              document.getElementById("result").textContent = JSON.stringify(data, null, 2);
            })
            .catch(err => alert("Error: " + err));
        }
      </script>
    </body>
    </html>
    """

@app.post("/detect_landmark/")
async def detect_landmark(
    lat: float = Form(...),
    lon: float = Form(...),
    file: UploadFile = File(None)
):
    user_loc = (lat, lon)

    # Step 1: GPS nearest landmark
    nearest = min(
        landmarks,
        key=lambda l: geodesic(user_loc, (l["lat"], l["lon"])).meters
    )
    distance = geodesic(user_loc, (nearest["lat"], nearest["lon"])).meters

    if distance < 200:  # GPS reliable
        summary = fetch_summary(nearest["name"])
        story = generate_story(nearest["name"], summary)
        return {
            "method": "gps",
            "landmark": nearest["name"],
            "distance_m": round(distance, 2),
            "summary": summary,
            "story": story
        }

    # Step 2: fallback image
    if file:
        temp_path = "temp.jpg"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        landmark = recognize_landmark(temp_path)
        summary = fetch_summary(landmark)
        story = generate_story(landmark, summary)
        return {
            "method": "image",
            "landmark": landmark,
            "note": "GPS unreliable, used image",
            "summary": summary,
            "story": story
        }

    return {"method": "none", "message": "GPS unreliable. Please upload a photo."}

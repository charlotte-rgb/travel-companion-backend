import os
import torch
from torchvision import models, transforms
from PIL import Image

# Path to your model checkpoint and dataset
MODEL_PATH = "checkpoints/aachen_resnet50_epoch10.pth"
DATA_DIR = "data/aachen"

# Load class names from folder structure
class_names = sorted(os.listdir(DATA_DIR))

# Same preprocessing as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_model():
    num_classes = len(class_names)
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

def recognize_landmark(image_path):
    model = load_model()
    img = Image.open(image_path).convert("RGB")
    inp = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(inp)
        pred_idx = logits.argmax(dim=1).item()

    return class_names[pred_idx]

if __name__ == "__main__":
    # Example usage
    result = recognize_landmark("sample_aachen.jpg")
    print("Predicted Landmark:", result)

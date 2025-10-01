import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "data/landmarks"   # folder: data/landmarks/{class_name}/*.jpg
OUTPUT_DIR = "checkpoints"
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 1e-4
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Dataset & Augmentation
# -----------------------------
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(DATA_DIR, transform=train_tfms)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

num_classes = len(train_ds.classes)
print(f"Found {num_classes} landmark classes")

# -----------------------------
# Model (ResNet50 backbone)
# -----------------------------
model = models.resnet50(pretrained=True)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model = model.to(DEVICE)

# -----------------------------
# Loss + Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    acc = 100. * correct / total
    print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Acc={acc:.2f}%")

    # Save checkpoint
    ckpt_path = os.path.join(OUTPUT_DIR, f"resnet50_epoch{epoch+1}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved: {ckpt_path}")

print("✅ Training complete")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from pathlib import Path
import time
import os

# --- CONFIGURATION ---
# Get path relative to this script
CURRENT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CURRENT_DIR.parent.parent

# Input Data (The split folders we just made)
TRAIN_DIR = PROJECT_ROOT / "data" / "split_dataset" / "train"
VAL_DIR = PROJECT_ROOT / "data" / "split_dataset" / "val"

# Output Files (Saved inside src/model)
MODEL_SAVE_PATH = CURRENT_DIR / "waste_model.pth"
LABEL_SAVE_PATH = CURRENT_DIR / "labels.txt"

BATCH_SIZE = 32
NUM_EPOCHS = 10  # A bit higher for better accuracy
LEARNING_RATE = 0.001

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on device: {device}")

    if not TRAIN_DIR.exists():
        print(f"❌ Error: Data not found at {TRAIN_DIR}")
        print("   Run 'prepare_data.py' first!")
        return

    # 1. PREPARE IMAGE TRANSFORMS
    # Training: Add randomness to help it learn better (Augmentation)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Validation: Just resize and normalize
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. LOAD DATA
    print("📥 Loading images...")
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_names = train_dataset.classes
    print(f"✅ Classes detected: {class_names}")

    # Save labels for your grading script later
    with open(LABEL_SAVE_PATH, 'w') as f:
        for name in class_names:
            f.write(name + '\n')

    # 3. BUILD MODEL (MobileNetV2)
    print("🧠 Building MobileNetV2 model...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # Freeze the base (so we don't ruin the pre-trained brain)
    for param in model.features.parameters():
        param.requires_grad = False

    # Change the last layer to match your 12 classes
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    # 4. TRAINING LOOP
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

    print("🔥 Starting training...")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        # Train on batches
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validate (Check accuracy)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(f"   Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Validation Acc: {val_acc:.2f}%")

    total_time = time.time() - start_time
    print(f"⏱️ Training finished in {total_time//60:.0f}m {total_time%60:.0f}s")

    # 5. SAVE MODEL
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"💾 Model saved successfully to: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train()
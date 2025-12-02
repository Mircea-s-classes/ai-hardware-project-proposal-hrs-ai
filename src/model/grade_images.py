import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import os
from pathlib import Path

CURRENT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CURRENT_DIR.parent.parent

INPUT_FOLDER = PROJECT_ROOT / "data" / "test_images"

OUTPUT_FOLDER = PROJECT_ROOT / "data" / "graded_images"

MODEL_PATH = CURRENT_DIR / "waste_model.pth"
LABEL_PATH = CURRENT_DIR / "labels.txt"

def get_smart_city_decision(label):
    recycling = ['cardboard', 'metal', 'paper', 'plastic', 'green-glass', 'brown-glass', 'white-glass']
    trash = ['biological', 'trash']
    hazardous = ['battery']
    donate = ['clothes', 'shoes']

    if label in recycling:
        return "RECYCLING", (0, 255, 0)      # Green
    elif label in hazardous:
        return "HAZARDOUS", (0, 165, 255)    # Orange
    elif label in donate:
        return "DONATE", (255, 255, 0)       # Cyan/Yellow
    else:
        return "TRASH", (0, 0, 255)          # Red

def grade():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not INPUT_FOLDER.exists():
        print(f"❌ Error: Create a folder at '{INPUT_FOLDER}' and put some random photos in it!")
        return
    
    if not MODEL_PATH.exists():
        print("❌ Error: Model not found. Run 'train_model.py' first!")
        return

    with open(LABEL_PATH, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    print(f"🧠 Loading model...")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not OUTPUT_FOLDER.exists():
        OUTPUT_FOLDER.mkdir(parents=True)

    image_files = [f for f in INPUT_FOLDER.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    print(f"📸 Found {len(image_files)} images to grade...")

    for img_path in image_files:
        try:
            pil_img = Image.open(img_path).convert('RGB')
        except:
            continue

        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            conf, index = torch.max(probs, 0)

        label = class_names[index.item()]
        category, color = get_smart_city_decision(label)

        cv_img = cv2.imread(str(img_path))
        if cv_img is None: continue
        
        cv_img = cv2.resize(cv_img, (600, 600))
        
        text_main = f"{category}: {label}"
        text_conf = f"Confidence: {conf.item()*100:.1f}%"
        
        cv2.putText(cv_img, text_main, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.putText(cv_img, text_conf, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        save_path = OUTPUT_FOLDER / f"graded_{img_path.name}"
        cv2.imwrite(str(save_path), cv_img)
        print(f"   ✅ Saved: graded_{img_path.name}")

    print(f"\n🎉 Done! Check the folder: {OUTPUT_FOLDER}")

if __name__ == '__main__':
    grade()
import numpy as np 
import onnxruntime as rt 
from PIL import Image
import cv2
from pathlib import Path
import os 

# --- PATH CONFIGURATION (FIXED) ---
CURRENT_DIR = Path(__file__).parent.resolve()
# Assuming your structure is project/src/hardware/script.py, PROJECT_ROOT is 'project/'
PROJECT_ROOT = CURRENT_DIR.parent.parent 

INPUT_FOLDER = PROJECT_ROOT / "data" / "test_images"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "graded_images"

# --- FIXED: Assuming model files are located in PROJECT_ROOT / "model" / ---
# Adjusting the path to find the model files in the project's 'model' directory.
MODEL_PATH = PROJECT_ROOT / "src" / "hardware" / "waste_model.onnx" 
LABEL_PATH = PROJECT_ROOT / "src" / "model" / "labels.txt"

# --- HELPER FUNCTION (UNCHANGED) ---
def get_smart_city_decision(label):
    recycling = ['cardboard', 'metal', 'paper', 'plastic', 'green-glass', 'brown-glass', 'white-glass']
    trash = ['biological', 'trash']
    hazardous = ['battery']
    donate = ['clothes', 'shoes']

    if label in recycling:
        return "RECYCLING", (0, 255, 0)       # Green
    elif label in hazardous:
        return "HAZARDOUS", (0, 165, 255)     # Orange
    elif label in donate:
        return "DONATE", (255, 255, 0)        # Cyan/Yellow
    else:
        return "TRASH", (0, 0, 255)           # Red

def grade():
    # REMOVED: device = torch.device(...) - ONNX Runtime manages this

    if not INPUT_FOLDER.exists():
        print(f"❌ Error: Create a folder at '{INPUT_FOLDER}' and put some random photos in it!")
        return
    
    # 1. PATH CHECK
    if not MODEL_PATH.exists():
        print(f"❌ Error: ONNX model not found at {MODEL_PATH}")
        print("   Make sure the model file exists at the specified path, or adjust MODEL_PATH.")
        return

    with open(LABEL_PATH, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    print(f"🧠 Loading ONNX model...")
    
    # 2. LOAD ONNX SESSION
    # This initializes the inference engine
    sess = rt.InferenceSession(str(MODEL_PATH))
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    # 3. DEFINE PREPROCESSING (Matches the PyTorch training/export)
    def preprocess_image(pil_img):
        img_resized = pil_img.resize((224, 224))
        img_np = np.array(img_resized, dtype=np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        img_norm = (img_np - mean) / std
        img_transposed = img_norm.transpose((2, 0, 1))
        
        input_tensor = np.expand_dims(img_transposed, axis=0)
        return input_tensor

    if not OUTPUT_FOLDER.exists():
        OUTPUT_FOLDER.mkdir(parents=True)

    image_files = [f for f in INPUT_FOLDER.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    print(f"📸 Found {len(image_files)} images to grade...")

    for img_path in image_files:
        try:
            pil_img = Image.open(img_path).convert('RGB')
        except:
            continue

        # --- INFERENCE USING ONNX RUNTIME ---
        input_tensor = preprocess_image(pil_img)

        # Run the model
        output = sess.run([output_name], {input_name: input_tensor})
        
        # Output is a list of NumPy arrays; we need the first one (the prediction)
        raw_prediction = output[0] 
        
        # Apply Softmax and find the predicted class index (NumPy operations)
        # np.exp is used to perform the softmax calculation (logits to probabilities)
        probs = np.exp(raw_prediction) / np.sum(np.exp(raw_prediction))
        conf = np.max(probs)
        index = np.argmax(probs)
        
        # --- CV2 Visuals (UNCHANGED) ---
        label = class_names[index.item()]
        category, color = get_smart_city_decision(label)

        cv_img = cv2.imread(str(img_path))
        if cv_img is None: continue
        
        cv_img = cv2.resize(cv_img, (600, 600))
        
        text_main = f"{category}: {label}"
        text_conf = f"Confidence: {conf*100:.1f}%"
        
        cv2.putText(cv_img, text_main, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.putText(cv_img, text_conf, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        save_path = OUTPUT_FOLDER / f"graded_{img_path.name}"
        cv2.imwrite(str(save_path), cv_img)
        print(f"   ✅ Saved: graded_{img_path.name}")

    print(f"\n🎉 Done! Check the folder: {OUTPUT_FOLDER}")

if __name__ == '__main__':
    grade()
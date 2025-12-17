import torch
import torch.nn as nn
from torchvision import models, transforms
from picamera2 import Picamera2
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from collections import deque, Counter
import time # Added for potential future timing/FPS calculation

# --- CONFIG ---
NUM_CLASSES = 12
MODEL_PATH = Path("../model/waste_model.hef")

# Smoothing parameters
SMOOTHING_WINDOW = 15        # number of frames to store
MIN_CONFIDENCE_COUNT = 9    # how many frames must agree to "stabilize" the prediction

# Display color (Green for stable detection)
COLOR_STABLE = (0, 255, 0)
# Display color (Yellow for unstable/detecting)
COLOR_UNSTABLE = (0, 255, 255)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOAD MODEL ---
def load_model(path: Path, num_classes: int, device):
    """Initializes and loads the MobileNetV2 model."""
    try:
        model = models.mobilenet_v2(weights=None) # Start without pre-trained weights
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {path}")
        exit()

model = load_model(MODEL_PATH, NUM_CLASSES, device)

# Classes
classes = [
    "cardboard", "glass", "metal", "paper", "plastic", "trash",
    "battery", "electronics", "organic", "clothes", "styrofoam", "other"
]

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- CAMERA SETUP ---
def setup_camera():
    """Configures and starts the Picamera2."""
    picam2 = Picamera2()
    # Using a 720p resolution is often a good balance between speed and quality
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (1280, 720)} # Increased resolution for better detection
    )
    picam2.configure(config)
    picam2.start()
    return picam2

picam2 = setup_camera()
print("Camera started. Press 'q' to quit.")

# --- INFERENCE LOOP ---
prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)
stable_label = "Detecting..."
stable_confidence = 0.0

try:
    while True:
        # Capture frame
        frame = picam2.capture_array()
        img_rgb = Image.fromarray(frame)

        # Preprocess and prepare tensor
        input_tensor = preprocess(img_rgb).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            
            # 1. Apply Softmax to get probabilities (CRITICAL FIX)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # 2. Get the confidence and the predicted index
            confidence, pred = torch.max(probabilities, 1) 
            current_label = classes[pred.item()]
            current_confidence = confidence.item()

        # Add to rolling buffer
        prediction_buffer.append(current_label)

        # Majority vote and Smoothing Logic
        most_common, count = Counter(prediction_buffer).most_common(1)[0]
        
        # Check against minimum confidence count, ensuring buffer is at least that long
        if count >= MIN_CONFIDENCE_COUNT and len(prediction_buffer) >= MIN_CONFIDENCE_COUNT:
            # Prediction is stable
            stable_label = most_common
            
            # Update stable confidence: Get the probability of the stable label
            # from the current frame's output.
            stable_label_index = classes.index(stable_label)
            stable_confidence = probabilities[0, stable_label_index].item()
            display_color = COLOR_STABLE
        else:
            # Prediction is unstable or still filling buffer
            stable_label = "Detecting..."
            stable_confidence = current_confidence # Show raw confidence while detecting
            display_color = COLOR_UNSTABLE

        # Display Prediction and Confidence
        confidence_percent = stable_confidence * 100
        
        # Display Stable Label and its Confidence
        display_text = f"Prediction: {stable_label}"
        cv2.putText(
            frame,
            display_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            display_color,
            2
        )
        
        # Display Confidence Percentage
        confidence_text = f"Confidence: {confidence_percent:.1f}%"
        cv2.putText(
            frame,
            confidence_text,
            (10, 70), # Moved down to a new line
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            display_color,
            2
        )


        cv2.imshow("Waste Classifier", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # --- RESOURCE CLEANUP (Improved Robustness) ---
    print("\nStopping camera and cleaning up resources...")
    picam2.stop()
    cv2.destroyAllWindows()
    # It's good practice to explicitly release other resources if possible
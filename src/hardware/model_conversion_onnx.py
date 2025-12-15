import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path 

MODEL_PATH = Path("src/model/waste_model.pth") 

# Setup
device = torch.device("cpu") 
CURRENT_DIR = Path(__file__).parent.resolve()
MODEL_PATH = CURRENT_DIR.parent / "model" / "waste_model.pth" 
NUM_CLASSES = 12 
ONNX_OUTPUT_PATH = CURRENT_DIR / "waste_model.onnx"

# Load the model architecture
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


dummy_input = torch.randn(1, 3, 224, 224, device=device) 

#create model 
torch.onnx.export(
    model, 
    dummy_input, 
    ONNX_OUTPUT_PATH, 
    export_params=True, 
    opset_version=17,        
    do_constant_folding=True,
    input_names=['input'], 
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}} 
)

print(f"🎉 Successfully exported model to {ONNX_OUTPUT_PATH}")
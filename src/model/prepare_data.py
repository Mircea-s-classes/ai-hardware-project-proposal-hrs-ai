import os
import shutil
import random
from pathlib import Path

# --- CONFIGURATION (Relative Paths) ---
# Get the location of THIS script (src/model)
CURRENT_DIR = Path(__file__).parent.resolve()

# Go up two levels to find 'data' (../../data)
PROJECT_ROOT = CURRENT_DIR.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "garbage_classification"
OUTPUT_DIR = PROJECT_ROOT / "data" / "split_dataset"

SPLIT_RATIO = 0.8  # 80% Training, 20% Validation

def prepare_data():
    print(f"📂 Looking for data at: {RAW_DATA_DIR}")

    if not RAW_DATA_DIR.exists():
        print(f"❌ Error: Could not find '{RAW_DATA_DIR}'")
        print("   Please make sure you moved the 'garbage_classification' folder into 'data'!")
        return

    # Create Train/Val folders
    train_dir = OUTPUT_DIR / "train"
    val_dir = OUTPUT_DIR / "val"

    # Reset output folder if it exists
    if OUTPUT_DIR.exists():
        print("   Cleaning old split data...")
        shutil.rmtree(OUTPUT_DIR)

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Get classes (battery, clothes, etc.)
    classes = [d.name for d in RAW_DATA_DIR.iterdir() if d.is_dir()]
    print(f"✅ Found {len(classes)} classes: {classes}")

    total_images = 0

    for class_name in classes:
        # Create subfolders
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)

        # Get images
        source_path = RAW_DATA_DIR / class_name
        images = [f for f in source_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        # Shuffle & Split
        random.shuffle(images)
        split_idx = int(len(images) * SPLIT_RATIO)
        
        train_files = images[:split_idx]
        val_files = images[split_idx:]

        # Copy files
        for img in train_files:
            shutil.copy(img, train_dir / class_name / img.name)
        for img in val_files:
            shutil.copy(img, val_dir / class_name / img.name)

        print(f"   Processed {class_name}: {len(train_files)} train, {len(val_files)} val")
        total_images += len(images)

    print("-" * 30)
    print(f"🎉 Done! Processed {total_images} images.")
    print(f"   Ready for training at: {OUTPUT_DIR}")

if __name__ == "__main__":
    prepare_data()
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

CURRENT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CURRENT_DIR.parent.parent

MODEL_PATH = PROJECT_ROOT / "src" / "model" / "waste_model.pth"
LABEL_PATH = PROJECT_ROOT / "src" / "model" / "labels.txt"
INPUT_FOLDER = PROJECT_ROOT / "data" / "test_images"
FIGURES_FOLDER = PROJECT_ROOT / "report" / "figures"

if not FIGURES_FOLDER.exists():
    FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)

def benchmark():
    print("--- 🚀 STARTING FULL SYSTEM BENCHMARK ---")
    
    device = torch.device("cpu")
    print(f"⚙️  Hardware: {device.type.upper()} (Laptop/PC Baseline)")

    if not MODEL_PATH.exists():
        print("❌ Error: Model not found. Run training first.")
        return

    print("🧠 Loading AI Model...")
    with open(LABEL_PATH, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except:
        print("❌ Error: Could not load weights.")
        return
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_files = [f for f in INPUT_FOLDER.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    if not image_files:
        print("❌ No images found in data/test_images.")
        return

    benchmark_samples = image_files * (50 // len(image_files) + 1)
    benchmark_samples = benchmark_samples[:50]
    
    print(f"📂 Benchmarking on {len(benchmark_samples)} inference passes...")

    input_batch = []
    for img_path in benchmark_samples:
        try:
            pil_img = Image.open(img_path).convert('RGB')
            input_tensor = preprocess(pil_img).unsqueeze(0)
            input_batch.append(input_tensor)
        except:
            pass

    print("🔥 Warming up engine...")
    with torch.no_grad():
        for _ in range(5): _ = model(input_batch[0])

    print("⏱️  Running Inference Loops...")
    latencies = []

    with torch.no_grad():
        for img_tensor in input_batch:
            start = time.time()
            _ = model(img_tensor)
            end = time.time()
            latencies.append((end - start) * 1000) 

    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    throughput = 1000 / avg_latency 

    print("\n" + "="*40)
    print("       📊 PERFORMANCE REPORT       ")
    print("="*40)
    print(f"Model:           MobileNetV2")
    print(f"Device:          CPU (Baseline)")
    print(f"Samples:         {len(latencies)}")
    print("-" * 40)
    print(f"⏱️  Avg Latency:  {avg_latency:.2f} ms")
    print(f"⚡ Throughput:   {throughput:.2f} FPS")
    print("-" * 40)
    print(f"Min Latency:     {min_latency:.2f} ms")
    print(f"Max Latency:     {max_latency:.2f} ms")
    print("="*40)

    print("\n🎨 Generating Graphs...")
    plt.figure(figsize=(10, 6))
    plt.hist(latencies, bins=15, color='#4CAF50', edgecolor='black', alpha=0.7)
    plt.axvline(avg_latency, color='red', linestyle='dashed', linewidth=2, label=f'Avg: {avg_latency:.1f}ms')
    plt.title('CPU Inference Latency Distribution', fontsize=16)
    plt.xlabel('Latency (ms)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(FIGURES_FOLDER / "latency_histogram.png")
    plt.close()
    print("   ✅ Saved: latency_histogram.png")

    metrics = ['Min', 'Avg', 'Max']
    values = [min_latency, avg_latency, max_latency]
    colors = ['#2196F3', '#FF9800', '#F44336']

    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics, values, color=colors)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f} ms", ha='center', va='bottom', fontweight='bold')
    plt.title('Latency Statistics (CPU)', fontsize=16)
    plt.ylabel('Time (ms)', fontsize=12)
    plt.ylim(0, max_latency * 1.2)
    plt.savefig(FIGURES_FOLDER / "latency_stats.png")
    plt.close()
    print("   ✅ Saved: latency_stats.png")

    dpu_est = 2.0 
    speedup = avg_latency / dpu_est
    
    plt.figure(figsize=(8, 6))
    plt.bar(['Raspberry Pi 5 (CPU)', 'AI HAT+ (DPU Estimate)'], [avg_latency, dpu_est], color=['#9E9E9E', '#673AB7'])
    plt.title('Hardware Acceleration Comparison', fontsize=16)
    plt.ylabel('Latency (Lower is Better)', fontsize=12)
    plt.text(0.5, (avg_latency + dpu_est)/2, f"⚡ {speedup:.1f}x Faster", fontsize=14, color='red', fontweight='bold', ha='center')
    plt.savefig(FIGURES_FOLDER / "projected_speedup.png")
    plt.close()
    print("   ✅ Saved: projected_speedup.png")

    print(f"\n🎉 Done! All figures saved to: {FIGURES_FOLDER}")

if __name__ == "__main__":
    benchmark()


This final `README.md` serves as a complete project report and technical guide for **Trash-E**. It integrates the provided Python scripts, performance data from the benchmarking, and the hardware deployment steps for the **Raspberry Pi 5** and **Hailo-8L**.

---
## 👥 Team Information

* **Team Name**: HRS AI
* **Team Members**:
* Raul Cancho
* Salina Tran
* Hannah Duong


# ♻️ Trash-E: Real-Time Edge AI Waste Classification

**Trash-E** is an end-to-end system designed to classify waste into distinct categories. This repository guides you from initial data preparation to deploying a high-speed, hardware-accelerated model on the edge.

## 📂 Project Repository Structure

The repository is split into two primary environments: the **Model Directory** (training and software evaluation) and the **Hardware Directory** (model conversion and edge deployment).

### 🧠 Model Directory (`src/model/`)

These scripts handle the data engineering and training phases:

This project is powered by a modular codebase where each script fulfills a specific role in the machine learning lifecycle:

* **`prepare_data.py`**: Handles initial data engineering by performing an **80/20 train/validation split** on raw garbage images to ensure a robust and unbiased training set.


* **`train_model.py`**: Trains the **MobileNetV2** model using **Transfer Learning**. It freezes the pre-trained feature extractor and trains a custom head for our specific classes.


* **`bench_perform.py`**: Evaluates the model on a CPU to establish a baseline. It measures and visualizes latency statistics, including min, average, and max times.

* **`grade_images.py`**: A post-processing and policy script that assigns "Smart City Decisions" to classified images.

### 🛠️ Hardware Directory (`src/hardware/`)

These files are used for hardware conversion and real-time execution:

* **`model_conversion_onnx.py`**: Converts the PyTorch `.pth` weights into an **ONNX format (Opset 17)**, a necessary intermediary step for hardware compilation.

* **`new_test.py`**: The final production script for real-time monitoring on the Raspberry Pi. It implements a **15-frame smoothing buffer** to ensure stable classification results in live video.



---

## 📊 Performance & Results

The system achieved high-performance benchmarks across both software and hardware platforms.

| Metric | CPU Baseline (Software) | Hardware Accelerated (Hailo-8L) | % Change (Improvement)
| --- | --- | --- | --- |
| **Avg Latency** | **28.2 ms** | **~17.8 ms** | **36.9% (Faster)** |
| **Throughput** | **35.5 FPS** | **30.0–40.0 FPS** | **12.7 % (Peak)** |
| **Max Latency** | **105.0 ms** | **26.3 ms** (Total Throughput) | **-75.0% (More Stable)**|
| **Accuracy** | **~85.0%** (Validation) | **~40.0–50.0%** (Real-world)* | **-47.1% (Real-world)**|

*Note: Real-world accuracy was impacted by the camera quality of the PiCam V2.1.

## ⚠️ Known Flaws & System Limitations

* **Camera Quality**: The **Raspberry Pi Camera V2.1** has significant quality issues; it is often **blurry** and lacks good visualization, which directly caused the drop in real-world accuracy.


* **Connectivity**: The camera hardware suffers from intermittent **connectivity issues**, requiring careful ribbon cable management during setup.


## 🛠️ Setup & Implementation "How-To"

### Phase 1: Software Development & Model Creation

Follow these steps to train and prepare your model for conversion.

1. **Prepare Data:**
```bash
python prepare_data.py
```


2. **Train the Model:**
```bash
python train_model.py
```
This generates `waste_model.pth` and `labels.txt`.


3. **Benchmarking (Optional):**
```bash
python bench_perform.py
```


4. **Export to ONNX:**
```bash
python model_conversion_onnx.py
```
This produces the intermediary `waste_model.onnx` file.


### Phase 2: Hardware Compilation (Host PC)

Convert your model into the **Hailo Execution Format (.hef)**.

1. **Load Compiler:** Activate your Hailo environment:
```bash
source /path/to/hailo_sdk/activate.sh
```


2. **Generate HEF:** Use the Hailo compiler with your `.onnx` file, a `.yaml` config, and a small sample of **calibration data** from your software folder.
* *Result:* This produces **`waste_model.hef`**, which is ready for the Hailo-8L.


### Phase 3: Hardware Platform Deployment (Raspberry Pi 5)

1. **Physical Setup:** Boot the **Raspberry Pi 5** and connect your monitor, keyboard, mouse, and the **PiCam V2.1**.


2. **Initialize Hardware:** Ensure the Hailo-8L AI HAT is properly installed and detected.


3. **Deploy Repo:** Pull the latest code from GitHub to the Pi.
4. **Run Real-Time Inference:**
```bash
python new_test.py
```
* The system will monitor the camera in real time, displaying a **prediction** and its **confidence interval** on screen. A stable prediction (Green) is only shown if the same label is detected for at least **9 out of 15** consecutive frames.


---

This final `README.md` serves as a complete project report and technical guide for **Trash-E**. It integrates the provided Python scripts, performance data from the benchmarking PDF, and the hardware deployment steps for the **Raspberry Pi 5** and **Hailo-8L**.

---

# ♻️ Trash-E: Real-Time Edge AI Waste Classification

**Trash-E** is an end-to-end system designed to classify waste into 12 distinct categories. This repository guides you from initial data preparation to deploying a high-speed, hardware-accelerated model on the edge.

## 📂 Project Repository Structure

This project is powered by a modular codebase where each script fulfills a specific role in the machine learning lifecycle:

* 
**`prepare_data.py`**: Handles initial data engineering by performing an **80/20 train/validation split** on raw garbage images to ensure a robust and unbiased training set.


* **`train_model.py`**: Trains the **MobileNetV2** model using **Transfer Learning**. It freezes the pre-trained feature extractor and trains a custom head for our specific classes.


* **`bench_perform.py`**: Evaluates the model on a CPU to establish a baseline. It measures and visualizes latency statistics, including min, average, and max times.


* 
**`model_conversion_onnx.py`**: Converts the PyTorch `.pth` weights into an **ONNX format (Opset 17)**, a necessary intermediary step for hardware compilation.


* 
**`grade_images.py`**: A post-processing and policy script that assigns "Smart City Decisions" (e.g., RECYCLING, HAZARDOUS) to classified images.


* **`new_test.py`**: The final production script for real-time monitoring on the Raspberry Pi. It implements a **15-frame smoothing buffer** to ensure stable classification results in live video.



---

## 📊 Performance & Results

The system achieved high-performance benchmarks across both software and hardware platforms.

| Metric | CPU Baseline (Software) | Hardware Accelerated (Hailo-8L) |
| --- | --- | --- |
| **Avg Latency** | <br>**28.2 ms** 

 | <br>**~17.8 ms** 

 |
| **Throughput** | **35.5 FPS** (Calculated) | <br>**30.0–40.0 FPS** 

 |
| **Max Latency** | <br>**105.0 ms** 

 | <br>**26.3 ms** (Total Throughput) 

 |
| **Accuracy** | **~85.0%** (Validation) | **~40.0–50.0%** (Real-world)* |

*Note: Real-world accuracy was impacted by the camera quality of the PiCam V2.1.

---

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
* 
*Result:* This produces **`waste_model.hef`**, which is ready for the Hailo-8L.





### Phase 3: Hardware Platform Deployment (Raspberry Pi 5)

1. 
**Physical Setup:** Boot the **Raspberry Pi 5** and connect your monitor, keyboard, mouse, and the **PiCam V2.1**.


2. 
**Initialize Hardware:** Ensure the Hailo-8L AI HAT is properly installed and detected.


3. **Deploy Repo:** Pull the latest code from GitHub to the Pi.
4. **Run Real-Time Inference:**
```bash
python new_test.py

```


* The system will monitor the camera in real time, displaying a **prediction** and its **confidence interval** on screen. A stable prediction (Green) is only shown if the same label is detected for at least **9 out of 15** consecutive frames.





---

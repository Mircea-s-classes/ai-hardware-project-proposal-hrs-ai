# University of Virginia
## Department of Electrical and Computer Engineering

**Course:** ECE 4332 / ECE 6332 — AI Hardware Design and Implementation  
**Semester:** Fall 2025  
**Proposal Deadline:** November 5, 2025 — 11:59 PM  
**Submission:** Upload to Canvas (PDF) and to GitHub (`/docs` folder)

---

# AI Hardware Project Proposal

## 1. Project Title
Name of the Team
HRS AI

List of students in the team
- Raul Cancho
- Hannah Duong
- Salina Tran

TRASH-E

## 2. Platform Selection
Edge-AI
Kria KV260 Vision AI Kit
The purpose of this project is real-time object identification for an automated sorting system, specifically with pieces of waste. In order to achieve the high throughput for continuous sorting processes, it requires hardware acceleration. The Kria KV260 with its integrated FPGA logic is an ideal choice to display this necessity as it allows us to quantify efficiency in a task important to smart cities or other applicable areas.

## 3. Problem Definition
This project aims to address the problem of sorting accuracy and high operational latency in waste management systems. Whereas traditional waste sorting can be time-consuming and manual, AI hardware usage allows for simplification if implemented efficiently. 

The specific problems we will use AI hardware to address are:
- Sorting Latency: Can the FPGA-accelerated Deep Processing Unit (DPU) reduce inference latency sufficiently to classify waste?
- Classification Efficiency: What differences does DPU vs. CPU execution have efficiency?
- Scalability:  How does the system perform/scale given various architectures such as YOLOv3, TinyYOLO, or another from Vitis AI Model Zoo


## 4. Technical Objectives
Deployment and Classification: 
    - Objective: Successfully deploy and run an object detection model that can classify at least three different waste            categories like plastic, paper, or non-recyclables
Measurement of Throughput: 
    - Objective: Quantify the speedup factor of the DPU over the CPU for the waste classification tasks
Determine the Sorting Speed: 
    - Objective: Measure the processing speed for the waste classification tasks 

## 5. Methodology
The hardware setup will utilize theKria KV260 Vision AI starter Kit, which is the Edge AI platform. An external USB camera will be connected to provide a live video stream of waste objects. The software and model design will center on the Vitis AI framework, which manages the deployment of the model onto the Zynq MPSoC arhcitecture. It will use an object detection model from the Vitis AI Model Zoo for waste categories. We will use the model to classify common household objects as proxies to waste types. The application will be the modification of the pre-built "Smart Camera" app to focus on the desired classes and output the correct classification. For performance metrics, we will collect FPS and inference latency for two different cases, which include DPU accelration and CPU acceleration for the waste model. In order to validate it, we will calculate the performance-per-watt for DPU acceleration case and compare it to the CPU baseline case. 

## 6. Expected Deliverables
The working demonstration will include a demo on the Kria KV260 classifying a physical object as "Plastic/Recylable" in real-time. The Github repository will include the setup instruction, scripts for model swapping, and raw data logs. The report will detail the architecture, model optimization process, and a full analysis that looks at the efficiency of the DPU and CPU for the waste sorting application. The presentation slides will summarize our process, results, and the feasability of using the KV260 in an automated sorting system. 

## 7. Team Responsibilities

| Name | Role  | Responsibilities |
|------|-------|------------------|
| Raul Cancho  | Team Lead | Coordination, documentation | Testing, benchmarking |
| Hannah Duong | Hardware | Setup, integration |
| Salina Tran  | Software | Model training, inference |

## 8. Timeline and Milestones

| Week | Milestone | Deliverable |
|------|------------|-------------|
| 2 | Proposal | PDF + GitHub submission |
| 4 | Midterm presentation | Slides, preliminary results |
| 6 | Integration & testing | Working prototype |
| Dec. 18 | Final presentation | Report, demo, GitHub archive |

## 9. Resources Required
Hardware Resources
 - Kria KV260 Vision AI Kit 
 - Smart Camera / camera sensor 
Software Resources
 - Vitis AI tools 
 - Smart Camera Application
 - Python / TensorFlow / PyTorch frameworks 
Data Resources
 - Labeled training and test datasets 
 - pre-trained models from the Vitis AI Model Zoo.

## 10. References
AMD Kria Documentation (https://xilinx.github.io/kria-apps-docs/home/build/html/index.html)


Smart Camera Application Guide (https://xilinx.github.io/kria-apps-docs/kv260/2022.1/build/html/docs/smartcamera/docs/app_deployment.html) 


Vitis AI Model Zoo (https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo)


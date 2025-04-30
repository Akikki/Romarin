# 🧭 RomarinCV

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Arduino](https://img.shields.io/badge/-Compatible-00979D?logo=arduino&logoColor=white&label=Arduino&labelColor=gray)

**RomarinCV** is the AI vision system developed to control **Romarin**, our custom-built submarine robot. This project combines Python and Arduino code to process camera input, recognize underwater objects, and navigate accordingly. This project was made as part of our studies at Sorbonne University.

---

## 📁 Project Structure

Below is a simplified overview of the most important folders and scripts in the project:

```
RomarinCV/
├── AI_vision/
│   └── finalRomarin.py                # Current execution script
│
├── Data/
│   ├── analyze.py              # Show a full model analysis with stats
│   ├── summary.py              # Shows a summary of the models
│   └── model.py                # Shows a web graph of the model
│
├── models/
│   ├── best.pt                 # Trained YOLO model used for object detection
│   └── best.onnx               # Converted ONNX version for lighter deployment
│
├── CameraScripts/
│   ├── cleanCamera.py          # Main camera testing using pycams, with a Inference output & a blank output (both video)
│   └── piCamera.py             # Standard scripts for pycamera
```

👉 *This only includes the most important or representative scripts. Each folder may contain additional files or helpers.*


---

## 🤖 About the Submarine

**Romain** is an autonomous underwater robot designed for tasks requiring object detection, navigation, and control in aquatic environments. The RomarinCV software enables it to:
- Interpret real-time camera input
- Identify objects via AI models
- Make movement decisions based on detection
- Operate either fully or partially autonomous

---

## 🛠️ Technologies

- Python 3.9+
- Arduino C/C++
- YOLOv8 / PyTorch-based AI models
- Custom made AI model
- OpenCV for vision processing

---
## ⚙️ Installation

To get started with RomarinCV on your local machine, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/RomarinCV.git
cd RomarinCV
```

> Replace `your-username` with your GitHub username or the actual repo link.

---

### 2. Set Up the Python Environment

It's recommended to use a virtual environment:

```bash
python -m venv .venv --system-site-packages # Must include site package if using picamera !
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Install torch using a torch install command found on : https://pytorch.org/get-started/locally/
---

### 3. Run the Main Program

To launch the main AI + control loop (submarine in action):

```bash
python AI_vision/webcamV2.py
```

To run detection only (no movement):

```bash
python CameraScripts/cleanCamera.py
```

---

## 🚧 Work in Progress

This project is under active development. Contributions, ideas, and feedback are welcome!

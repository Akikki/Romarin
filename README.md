# 🧭 RomarinCV

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Arduino](https://img.shields.io/badge/-Compatible-00979D?logo=arduino&logoColor=white&label=Arduino&labelColor=gray)

**RomarinCV** is the AI vision system developed to control **Romain**, our custom-built submarine robot. This project combines Python and Arduino code to process camera input, recognize underwater objects, and navigate accordingly.

---

## 📁 Project Structure

Below is a simplified overview of the most important folders and scripts in the project:

```
RomarinCV/
├── **AI_vision/**
│   └── webcamV2                # Current execution script
│
├── **Data/**
│   ├── analyze.py              # Show a full model analysis with stats
│   ├── summary.py              # Shows a summary of the models
│   └── model.py                # Shows a web graph of the model
│
├── **models/**
│   ├── best.pt                 # Trained YOLO model used for object detection
│   └── best.onnx               # Converted ONNX version for lighter deployment
│
├── **CameraScripts/**
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
- YOLOv5 / PyTorch-based AI models
- OpenCV for vision processing

---

## 🚧 Work in Progress

This project is under active development. Contributions, ideas, and feedback are welcome!

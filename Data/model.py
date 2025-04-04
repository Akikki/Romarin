from ultralytics import YOLO
import os

# Load the YOLOv8 model
model = YOLO("../models/best.pt")

# Export to ONNX format
exported_model = model.export(format="onnx", opset=12)

# Get the path to the ONNX file
onnx_path = exported_model if isinstance(exported_model, str) else exported_model[0]

print(f"✅ Model exported to ONNX format: {onnx_path}")

# OPTIONAL: Launch Netron to visualize the model
try:
    import netron
    print("🔍 Launching Netron...")
    netron.start(onnx_path)
except ImportError:
    print("ℹ️ Netron not installed. Run 'pip install netron' to visualize the model.")

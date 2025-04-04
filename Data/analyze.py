import torch
from ultralytics import YOLO
from torch.profiler import profile, record_function, ProfilerActivity

# Load model and input
model = YOLO("../models/best.pt")
model = model.model.eval()  # Under the hood: model.model.model is the core module

# Create dummy input
imgsz = model.args['imgsz']
imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
dummy_input = torch.randn(1, 3, *imgsz).to("cpu")  # Switch to "cuda" if using GPU

# Profile the model
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("yolo_inference"):
        model(dummy_input)

# Print summary
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25))

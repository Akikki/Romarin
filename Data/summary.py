import torch
from ultralytics import YOLO
from torchinfo import summary

# Load YOLO model
model = YOLO("../models/best.pt")
core_model = model.model  # This is a DetectionModel instance

# Get input shape
imgsz = core_model.args['imgsz']
imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
input_shape = (1, 3, *imgsz)

# Safe wrapper using plain forward pass
class CleanForwardWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Inference mode to avoid postprocessing (like NMS)
        return self.model.forward(x, augment=False, visualize=False)

wrapped_model = CleanForwardWrapper(core_model)

# Generate the summary
summary(
    wrapped_model,
    input_size=input_shape,
    col_names=["input_size", "output_size", "num_params", "trainable"],
    depth=5,
    row_settings=["var_names"]
)

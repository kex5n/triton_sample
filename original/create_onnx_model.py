import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

model = fasterrcnn_resnet50_fpn(weights="COCO_V1")
model.eval()
dummpy_input = torch.randn(1, 3, 216, 384)
torch.onnx.export(model, dummpy_input, "model.onnx", opset_version=12, input_names=["input"],output_names=["boxes", "labels", "scores"])

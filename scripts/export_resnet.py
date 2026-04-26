import torch
import torchvision
import os

model = torchvision.models.resnet18(pretrained=False)
model.eval()
dummy = torch.randn(1, 3, 224, 224)

os.makedirs("models", exist_ok=True)
torch.onnx.export(
    model,
    dummy,
    "models/resnet18.onnx",
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
)
print("exported models/resnet18.onnx")

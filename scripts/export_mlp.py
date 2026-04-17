"""Export a simple 5-layer MLP to ONNX for ferroflow testing.

Usage:
    python scripts/export_mlp.py
"""

import torch
import os

os.makedirs("models", exist_ok=True)

model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10),
)
model.eval()

dummy = torch.randn(1, 784)

torch.onnx.export(
    model,
    dummy,
    "models/mlp.onnx",
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)

print("exported models/mlp.onnx")

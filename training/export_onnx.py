from __future__ import annotations

import torch
import torchvision

from app.config import IMAGE_SIZE, MODEL_PATH, NUM_CLASSES, TORCH_MODEL_PATH


def export() -> None:
    if not TORCH_MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model weights not found: {TORCH_MODEL_PATH}")

    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    state_dict = torch.load(TORCH_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    dummy = torch.randn(1, 3, IMAGE_SIZE[1], IMAGE_SIZE[0], dtype=torch.float32)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        MODEL_PATH,
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

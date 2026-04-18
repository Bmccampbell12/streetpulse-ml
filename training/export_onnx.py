from __future__ import annotations

import shutil
import torch
import torchvision

from app.config import IMAGE_SIZE, MODEL_PATH, MODEL_VERSION_PATH, NUM_CLASSES, TORCH_MODEL_PATH


def export(model_version: str | None = None) -> dict[str, str]:
    if not TORCH_MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model weights not found: {TORCH_MODEL_PATH}")

    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    state_dict = torch.load(TORCH_MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    dummy = torch.randn(1, 3, IMAGE_SIZE[1], IMAGE_SIZE[0], dtype=torch.float32)

    version = model_version or "baseline"
    versioned_path = MODEL_PATH.parent / f"{version}.onnx"
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        versioned_path,
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

    shutil.copyfile(versioned_path, MODEL_PATH)
    MODEL_VERSION_PATH.write_text(version, encoding="utf-8")
    return {"model_version": version, "onnx_path": str(versioned_path)}

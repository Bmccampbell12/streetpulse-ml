from __future__ import annotations

from typing import BinaryIO

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.config import IMAGE_SIZE, LABELS
from app.model_loader import load_model

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def preprocess(image_file: BinaryIO) -> np.ndarray:
    try:
        if hasattr(image_file, "seek"):
            image_file.seek(0)
        img = Image.open(image_file).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Unsupported or invalid image file") from exc

    img = img.resize(IMAGE_SIZE)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    arr = (arr - _MEAN) / _STD
    return arr


def predict(image_file: BinaryIO) -> dict[str, float | str]:
    session = load_model()
    input_data = preprocess(image_file)
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: input_data})
    logits = np.asarray(outputs[0][0], dtype=np.float32)
    probabilities = _softmax(logits)

    idx = int(np.argmax(probabilities))
    return {
        "label": LABELS[idx],
        "confidence": float(probabilities[idx]),
    }

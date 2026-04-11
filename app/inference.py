from __future__ import annotations

import json
from typing import BinaryIO

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.config import CALIBRATION_PATH, CALIBRATION_TEMPERATURE_DEFAULT, IMAGE_SIZE, LABELS, MODEL_VERSION_PATH, UNCERTAIN_THRESHOLD
from app.model_loader import load_model

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _load_calibration_temperature() -> float:
    if not CALIBRATION_PATH.exists():
        return CALIBRATION_TEMPERATURE_DEFAULT
    try:
        data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        value = float(data.get("temperature", CALIBRATION_TEMPERATURE_DEFAULT))
        return max(0.05, value)
    except (OSError, ValueError, json.JSONDecodeError, TypeError):
        return CALIBRATION_TEMPERATURE_DEFAULT


def _load_model_version() -> str:
    if not MODEL_VERSION_PATH.exists():
        return "unknown"
    try:
        value = MODEL_VERSION_PATH.read_text(encoding="utf-8").strip()
        return value or "unknown"
    except OSError:
        return "unknown"


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
    temperature = _load_calibration_temperature()
    probabilities = _softmax(logits / temperature)

    idx = int(np.argmax(probabilities))
    confidence = float(probabilities[idx])
    label = LABELS[idx] if confidence >= UNCERTAIN_THRESHOLD else "uncertain"

    return {
        "label": label,
        "confidence": confidence,
        "temperature": float(temperature),
        "model_version": _load_model_version(),
    }

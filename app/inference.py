from __future__ import annotations

import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

# Ensure the project root is on sys.path when this module is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.config import (
    ACTIVE_LEARNING_QUEUE_PATH,
    ACTIVE_LEARNING_THRESHOLD,
    AUTO_LABEL_QUEUE_PATH,
    AUTO_LABEL_THRESHOLD,
    CALIBRATION_PATH,
    CALIBRATION_TEMPERATURE_DEFAULT,
    CLASS_INDEX_PATH,
    IMAGE_SIZE,
    INFERENCE_LOG_PATH,
    LABELS,
    MODEL_BACKEND,
    MODEL_VERSION_PATH,
    UNCERTAIN_THRESHOLD,
)
from app.model_loader import load_model
from app.versioning import get_dataset_version

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
_LOG_LOCK = threading.Lock()


class _ModelBackend:
    def run_logits(self, input_data: np.ndarray) -> np.ndarray:  # pragma: no cover - interface method
        raise NotImplementedError


class _OnnxModelBackend(_ModelBackend):
    def run_logits(self, input_data: np.ndarray) -> np.ndarray:
        session = load_model()
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_data})
        return np.asarray(outputs[0][0], dtype=np.float32)


def _get_backend() -> _ModelBackend:
    if MODEL_BACKEND == "onnx":
        return _OnnxModelBackend()
    raise ValueError(f"Unsupported model backend: {MODEL_BACKEND}")


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


def _append_jsonl(path, payload: dict[str, object]) -> None:
    line = json.dumps(payload, ensure_ascii=True)
    with _LOG_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _load_class_map() -> dict[int, str]:
    if not CLASS_INDEX_PATH.exists():
        return {idx: label for idx, label in enumerate(LABELS)}

    try:
        raw = json.loads(CLASS_INDEX_PATH.read_text(encoding="utf-8"))
        idx_to_label = {int(value): key for key, value in raw.items()}
        if not idx_to_label:
            raise ValueError("empty class_to_idx mapping")
        return idx_to_label
    except (OSError, ValueError, json.JSONDecodeError, TypeError):
        return {idx: label for idx, label in enumerate(LABELS)}

_IDX_TO_LABEL = _load_class_map()


def _record_inference_event(*, source_id: str | None, label: str, confidence: float, model_version: str) -> None:
    dataset_version = get_dataset_version()
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_id": source_id,
        "label": label,
        "confidence": confidence,
        "model_version": model_version,
        "dataset_version": dataset_version,
        "backend": MODEL_BACKEND,
    }
    _append_jsonl(INFERENCE_LOG_PATH, event)

    queue_event = dict(event)
    if label != "uncertain" and confidence >= AUTO_LABEL_THRESHOLD:
        _append_jsonl(AUTO_LABEL_QUEUE_PATH, queue_event)
    elif label == "uncertain" or confidence < ACTIVE_LEARNING_THRESHOLD:
        _append_jsonl(ACTIVE_LEARNING_QUEUE_PATH, queue_event)


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


def predict(image_file: BinaryIO, source_id: str | None = None) -> dict[str, float | str]:
    backend = _get_backend()
    input_data = preprocess(image_file)
    logits = backend.run_logits(input_data)
    temperature = _load_calibration_temperature()
    probabilities = _softmax(logits / temperature)

    idx = int(np.argmax(probabilities))
    confidence = float(probabilities[idx])
    label = _IDX_TO_LABEL.get(idx, LABELS[idx] if idx < len(LABELS) else "unknown")
    if confidence < UNCERTAIN_THRESHOLD:
        label = "uncertain"
    model_version = _load_model_version()

    _record_inference_event(source_id=source_id, label=label, confidence=confidence, model_version=model_version)

    return {
        "label": label,
        "confidence": confidence,
        "temperature": float(temperature),
        "model_version": model_version,
    }

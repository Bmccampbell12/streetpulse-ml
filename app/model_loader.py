from __future__ import annotations

import threading

import onnxruntime as ort

from app.config import MODEL_PATH

_session: ort.InferenceSession | None = None
_lock = threading.Lock()


def load_model() -> ort.InferenceSession:
    global _session
    if _session is not None:
        return _session

    with _lock:
        if _session is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")
            _session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])

    return _session


def reload_model() -> ort.InferenceSession:
    """Discard the cached session and load a fresh one (call after retraining)."""
    global _session
    with _lock:
        _session = None
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")
        _session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    return _session

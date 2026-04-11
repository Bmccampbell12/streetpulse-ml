from __future__ import annotations

import json
import threading
from datetime import datetime, timezone

from app.config import METADATA_PATH

_lock = threading.Lock()


def record_label(
    filename: str,
    label: str,
    *,
    source: str = "stream",
    predicted_label: str | None = None,
    confidence: float | None = None,
    device: str | None = None,
    model_version: str | None = None,
) -> None:
    """Append a labeling event to the persistent dataset index."""
    entry: dict[str, object] = {
        "filename": filename,
        "label": label,
        "source": source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "predicted_label": predicted_label,
        "confidence": confidence,
        "device": device,
        "model_version": model_version,
    }
    with _lock:
        data: list[dict[str, object]] = []
        if METADATA_PATH.exists():
            try:
                data = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
                if not isinstance(data, list):
                    data = []
            except (json.JSONDecodeError, OSError):
                data = []
        data.append(entry)
        METADATA_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def record_hard_negative(
    *,
    filename: str,
    predicted_label: str,
    true_label: str,
    confidence: float | None,
    model_version: str | None,
) -> None:
    """Append a hard-negative mining event for later retraining analysis."""
    entry: dict[str, object] = {
        "type": "hard_negative",
        "filename": filename,
        "predicted_label": predicted_label,
        "true_label": true_label,
        "confidence": confidence,
        "model_version": model_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with _lock:
        data: list[dict[str, object]] = []
        if METADATA_PATH.exists():
            try:
                data = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
                if not isinstance(data, list):
                    data = []
            except (json.JSONDecodeError, OSError):
                data = []
        data.append(entry)
        METADATA_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def count_labeled() -> int:
    """Return the total number of labeling events recorded."""
    with _lock:
        if not METADATA_PATH.exists():
            return 0
        try:
            data = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            return len(data) if isinstance(data, list) else 0
        except (json.JSONDecodeError, OSError):
            return 0


def load_index() -> list[dict[str, object]]:
    """Return the full dataset index."""
    with _lock:
        if not METADATA_PATH.exists():
            return []
        try:
            data = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

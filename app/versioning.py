from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.config import DATASET_DIR, DATASET_VERSION_PATH

_lock = threading.Lock()
_VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def get_dataset_version() -> str:
    if not DATASET_VERSION_PATH.exists():
        return "v0"
    try:
        data = json.loads(DATASET_VERSION_PATH.read_text(encoding="utf-8"))
        return str(data.get("version", "v0"))
    except (OSError, ValueError, json.JSONDecodeError):
        return "v0"


def _dataset_fingerprint() -> str:
    hasher = hashlib.sha256()
    roots = [DATASET_DIR / "labeled", DATASET_DIR / "curated", DATASET_DIR / "hard_negatives"]

    for root in roots:
        if not root.exists():
            continue
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file() or file_path.suffix.lower() not in _VALID_EXTENSIONS:
                continue
            stat = file_path.stat()
            rel = file_path.resolve().relative_to(DATASET_DIR.resolve()).as_posix()
            hasher.update(rel.encode("utf-8"))
            hasher.update(str(stat.st_size).encode("utf-8"))
            hasher.update(str(stat.st_mtime_ns).encode("utf-8"))

    return hasher.hexdigest()


def snapshot_dataset_version(reason: str = "pipeline") -> dict[str, str]:
    """Compute and persist the current dataset version snapshot."""
    now = datetime.now(timezone.utc).isoformat()
    fingerprint = _dataset_fingerprint()
    version_id = f"ds_{fingerprint[:12]}"
    payload = {
        "version": version_id,
        "fingerprint": fingerprint,
        "updated_at": now,
        "reason": reason,
    }

    with _lock:
        DATASET_VERSION_PATH.parent.mkdir(parents=True, exist_ok=True)
        DATASET_VERSION_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {"version": version_id, "updated_at": now}


def get_dataset_version() -> str:
    if not DATASET_VERSION_PATH.exists():
        return snapshot_dataset_version(reason="bootstrap")["version"]

    with _lock:
        try:
            payload = json.loads(DATASET_VERSION_PATH.read_text(encoding="utf-8"))
            value = str(payload.get("version", "")).strip()
            if value:
                return value
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass

    return snapshot_dataset_version(reason="recover")["version"]

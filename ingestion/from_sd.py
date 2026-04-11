from __future__ import annotations

import logging
import shutil
from pathlib import Path

from app.config import RAW_DIR, SD_CARD_PATH

log = logging.getLogger(__name__)

_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def import_sd(sd_path: Path | None = None) -> int:
    source_dir = sd_path or SD_CARD_PATH
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists():
        log.warning("SD ingestion skipped: path not found: %s", source_dir)
        return 0

    copied = 0
    for item in source_dir.iterdir():
        if not item.is_file() or item.suffix.lower() not in _ALLOWED_EXTENSIONS:
            continue

        destination = RAW_DIR / item.name
        if destination.exists():
            continue

        shutil.copy2(item, destination)
        copied += 1

    return copied

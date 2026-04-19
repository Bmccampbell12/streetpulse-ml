from __future__ import annotations

import logging
import shutil
from pathlib import Path

from app.config import RAW_DIR, SD_CARD_PATH

log = logging.getLogger(__name__)

_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _destination_name(source_file: Path, source_dir: Path) -> str:
    relative = source_file.relative_to(source_dir)
    # Flatten nested SD-card folders into a stable filename for RAW_DIR.
    return "__".join(relative.parts)


def import_sd(sd_path: Path | None = None) -> int:
    source_dir = sd_path or SD_CARD_PATH
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists():
        log.warning("SD ingestion skipped: path not found: %s", source_dir)
        return 0

    copied = 0
    failed = 0
    for item in source_dir.rglob("*"):
        if not item.is_file() or item.suffix.lower() not in _ALLOWED_EXTENSIONS:
            continue

        destination = RAW_DIR / _destination_name(item, source_dir)
        if destination.exists():
            continue

        try:
            shutil.copy2(item, destination)
            copied += 1
        except OSError as exc:
            failed += 1
            log.warning("SD import skipped for %s: %s", item, exc)

    if failed:
        log.warning("SD ingestion completed with failures. copied=%d failed=%d", copied, failed)

    return copied

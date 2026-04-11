from __future__ import annotations

from PIL import Image, UnidentifiedImageError

from app.config import CURATED_DIR, RAW_DIR

_MIN_WIDTH = 100
_MIN_HEIGHT = 100
_VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}


def filter_images() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CURATED_DIR.mkdir(parents=True, exist_ok=True)

    curated_count = 0
    for image_path in RAW_DIR.iterdir():
        if not image_path.is_file() or image_path.suffix.lower() not in _VALID_SUFFIXES:
            continue

        output_path = CURATED_DIR / image_path.name

        try:
            with Image.open(image_path) as img:
                if img.width < _MIN_WIDTH or img.height < _MIN_HEIGHT:
                    continue
                img.convert("RGB").save(output_path)
                curated_count += 1
        except (UnidentifiedImageError, OSError):
            continue

    return curated_count

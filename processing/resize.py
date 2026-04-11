from __future__ import annotations

from PIL import Image

from app.config import IMAGE_SIZE, LABELED_DIR

_VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}


def resize_all() -> int:
    resized = 0
    for image_path in LABELED_DIR.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in _VALID_SUFFIXES:
            continue

        with Image.open(image_path) as img:
            resized_image = img.convert("RGB").resize(IMAGE_SIZE)
            resized_image.save(image_path)
            resized += 1

    return resized

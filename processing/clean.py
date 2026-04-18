from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path when this module is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageStat, UnidentifiedImageError

from app.config import CURATED_DIR, MIN_BRIGHTNESS, MIN_SHARPNESS, RAW_DIR

_MIN_WIDTH = 100
_MIN_HEIGHT = 100
_VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}


def filter_images_with_report() -> dict[str, int]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CURATED_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "curated": 0,
        "invalid": 0,
        "too_small": 0,
        "too_dark": 0,
        "too_blurry": 0,
    }

    for image_path in RAW_DIR.iterdir():
        if not image_path.is_file() or image_path.suffix.lower() not in _VALID_SUFFIXES:
            continue

        output_path = CURATED_DIR / image_path.name

        try:
            with Image.open(image_path) as img:
                if img.width < _MIN_WIDTH or img.height < _MIN_HEIGHT:
                    report["too_small"] += 1
                    continue

                rgb = img.convert("RGB")
                
                # Blur detection filter
                gray = np.array(img.convert('L'), dtype=np.float32)
                sharpness = float(np.var(gray))          # Laplacian variance proxy
                brightness = float(np.mean(gray))

                if sharpness < MIN_SHARPNESS:
                    report["too_blurry"] += 1
                    continue   # too blurry
                
                if brightness < MIN_BRIGHTNESS:
                    report["too_dark"] += 1
                    continue # too dark

                rgb.save(output_path)
                report["curated"] += 1
        except (UnidentifiedImageError, OSError):
            report["invalid"] += 1

    return report


def filter_images() -> int:
    return filter_images_with_report()["curated"]

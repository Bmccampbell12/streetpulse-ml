from __future__ import annotations

import logging
from pathlib import Path

import requests

from app.config import API_URL, RAW_DIR

log = logging.getLogger(__name__)


def fetch_images(api_url: str | None = None) -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    source_url = api_url or API_URL

    try:
        response = requests.get(source_url, timeout=20)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.ConnectionError:
        log.warning("API ingestion skipped: cannot reach %s (server not running)", source_url)
        return 0
    except requests.exceptions.RequestException as exc:
        log.warning("API ingestion skipped: %s", exc)
        return 0

    if not isinstance(data, list):
        raise ValueError("API response must be a JSON array")

    downloaded = 0
    for item in data:
        image_id = str(item.get("id", "")).strip()
        image_url = item.get("url")
        if not image_id or not image_url:
            continue

        filename = RAW_DIR / f"{image_id}.jpg"
        if filename.exists():
            continue

        content = requests.get(image_url, timeout=30)
        content.raise_for_status()
        filename.write_bytes(content.content)
        downloaded += 1

    return downloaded

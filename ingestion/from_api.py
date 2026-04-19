from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from app.config import API_URL, PIPELINE_MAX_RETRIES, PIPELINE_RETRY_DELAY_SECONDS, RAW_DIR

log = logging.getLogger(__name__)


def _request_with_retry(url: str, *, timeout: int, expect_json: bool, retries: int) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            if expect_json:
                response.json()
            return response
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(PIPELINE_RETRY_DELAY_SECONDS)
                continue
            raise

    raise RuntimeError(f"Unexpected retry state for URL: {url}") from last_error


def fetch_images(api_url: str | None = None) -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    source_url = api_url or API_URL

    try:
        response = _request_with_retry(source_url, timeout=20, expect_json=True, retries=PIPELINE_MAX_RETRIES)
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
    failed = 0
    for item in data:
        image_id = str(item.get("id", "")).strip()
        image_url = item.get("url")
        if not image_id or not image_url:
            continue

        filename = RAW_DIR / f"{image_id}.jpg"
        if filename.exists():
            continue

        try:
            content = _request_with_retry(str(image_url), timeout=30, expect_json=False, retries=PIPELINE_MAX_RETRIES)
            filename.write_bytes(content.content)
            downloaded += 1
        except requests.exceptions.RequestException as exc:
            failed += 1
            log.warning("Failed downloading image %s from %s: %s", image_id, image_url, exc)

    if failed:
        log.warning("API ingestion completed with failures. downloaded=%d failed=%d", downloaded, failed)

    return downloaded

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
RAW_DIR = DATASET_DIR / "raw"
STREAM_RAW_DIR = RAW_DIR / "stream"
CURATED_DIR = DATASET_DIR / "curated"
CURATED_STREAM_DIR = CURATED_DIR / "stream"
LABELED_DIR = DATASET_DIR / "labeled"
MODELS_DIR = BASE_DIR / "models"
METADATA_PATH = DATASET_DIR / "index.json"

MODEL_PATH = MODELS_DIR / "latest_model.onnx"
TORCH_MODEL_PATH = MODELS_DIR / "model.pth"
CLASS_INDEX_PATH = MODELS_DIR / "class_to_idx.json"

SD_CARD_PATH = BASE_DIR / "sd_card"
API_URL = os.getenv("STREETPULSE_IMAGE_API", "http://localhost:3000/api/images")

IMAGE_SIZE = (224, 224)
LABELS = ["pothole", "crack", "normal", "manhole"]
NUM_CLASSES = len(LABELS)

BATCH_SIZE = int(os.getenv("STREETPULSE_BATCH_SIZE", "16"))
EPOCHS = int(os.getenv("STREETPULSE_EPOCHS", "5"))
LEARNING_RATE = float(os.getenv("STREETPULSE_LR", "0.001"))
RANDOM_SEED = int(os.getenv("STREETPULSE_SEED", "42"))
AUTO_TRAIN_THRESHOLD = int(os.getenv("STREETPULSE_AUTO_TRAIN_THRESHOLD", "50"))
STREAM_STALE_HOURS = float(os.getenv("STREETPULSE_STALE_HOURS", "24"))


def ensure_directories() -> None:
    for path in [RAW_DIR, STREAM_RAW_DIR, CURATED_DIR, CURATED_STREAM_DIR, LABELED_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)

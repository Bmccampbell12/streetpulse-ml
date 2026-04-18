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
HARD_NEGATIVE_DIR = DATASET_DIR / "hard_negatives"
MODELS_DIR = BASE_DIR / "models"
METADATA_PATH = DATASET_DIR / "index.json"
VERSIONING_DIR = DATASET_DIR / "versioning"
DATASET_VERSION_PATH = VERSIONING_DIR / "dataset_version.json"
AUTO_LABEL_QUEUE_PATH = DATASET_DIR / "auto_labels_pending.jsonl"
ACTIVE_LEARNING_QUEUE_PATH = DATASET_DIR / "active_learning_queue.jsonl"

MODEL_PATH = MODELS_DIR / "latest_model.onnx"
TORCH_MODEL_PATH = MODELS_DIR / "model.pth"
MODEL_VERSION_PATH = MODELS_DIR / "latest_model.version"
CALIBRATION_PATH = MODELS_DIR / "calibration.json"
CLASS_INDEX_PATH = MODELS_DIR / "class_to_idx.json"
INFERENCE_LOG_PATH = MODELS_DIR / "inference_log.jsonl"
PIPELINE_RUNS_DIR = BASE_DIR / "pipeline_runs"
PIPELINE_STATE_PATH = PIPELINE_RUNS_DIR / "pipeline_state.json"

SD_CARD_PATH = Path(os.getenv("STREETPULSE_SD_CARD_PATH", str(BASE_DIR / "sd_card"))).expanduser()
API_URL = os.getenv("STREETPULSE_IMAGE_API", "http://localhost:3000/api/images")

IMAGE_SIZE = (224, 224)
LABELS = ["pothole", "crack", "normal", "manhole"]
NUM_CLASSES = len(LABELS)

BATCH_SIZE = int(os.getenv("STREETPULSE_BATCH_SIZE", "16"))
EPOCHS = int(os.getenv("STREETPULSE_EPOCHS", "30"))
LEARNING_RATE = float(os.getenv("STREETPULSE_LR", "0.001"))
RANDOM_SEED = int(os.getenv("STREETPULSE_SEED", "42"))
AUTO_TRAIN_THRESHOLD = int(os.getenv("STREETPULSE_AUTO_TRAIN_THRESHOLD", "50"))
STREAM_STALE_HOURS = float(os.getenv("STREETPULSE_STALE_HOURS", "24"))
UNCERTAIN_THRESHOLD = float(os.getenv("STREETPULSE_UNCERTAIN_THRESHOLD", "0.6"))
CALIBRATION_TEMPERATURE_DEFAULT = float(os.getenv("STREETPULSE_CALIBRATION_T", "1.0"))
USE_WEIGHTED_SAMPLER = os.getenv("STREETPULSE_WEIGHTED_SAMPLER", "false").lower() == "true"
HARD_NEGATIVE_CONFIDENCE = float(os.getenv("STREETPULSE_HARD_NEGATIVE_CONFIDENCE", "0.7"))
MODEL_BACKEND = os.getenv("STREETPULSE_MODEL_BACKEND", "onnx").strip().lower()
PIPELINE_MAX_RETRIES = int(os.getenv("STREETPULSE_PIPELINE_MAX_RETRIES", "2"))
PIPELINE_RETRY_DELAY_SECONDS = float(os.getenv("STREETPULSE_PIPELINE_RETRY_DELAY_SECONDS", "1.5"))
INFER_BATCH_SIZE = int(os.getenv("STREETPULSE_INFER_BATCH_SIZE", "16"))
AUTO_LABEL_THRESHOLD = float(os.getenv("STREETPULSE_AUTO_LABEL_THRESHOLD", "0.9"))
ACTIVE_LEARNING_THRESHOLD = float(os.getenv("STREETPULSE_ACTIVE_LEARNING_THRESHOLD", "0.75"))
MIN_BRIGHTNESS = float(os.getenv("STREETPULSE_MIN_BRIGHTNESS", "20.0"))
MIN_SHARPNESS = float(os.getenv("STREETPULSE_MIN_SHARPNESS", "15.0"))


def ensure_directories() -> None:
    for path in [
        RAW_DIR,
        STREAM_RAW_DIR,
        CURATED_DIR,
        CURATED_STREAM_DIR,
        LABELED_DIR,
        HARD_NEGATIVE_DIR,
        MODELS_DIR,
        VERSIONING_DIR,
        PIPELINE_RUNS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

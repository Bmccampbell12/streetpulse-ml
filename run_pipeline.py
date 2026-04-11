from __future__ import annotations

import argparse

from app.config import ensure_directories
from ingestion.from_api import fetch_images
from ingestion.from_sd import import_sd
from processing.clean import filter_images
from processing.resize import resize_all
from training.export_onnx import export
from training.train import train


def run(source: str = "api", resize_labeled: bool = False) -> None:
    ensure_directories()

    print("[1] Ingestion...")
    if source in {"api", "both"}:
        downloaded = fetch_images()
        print(f"  - API downloaded: {downloaded}")
    if source in {"sd", "both"}:
        copied = import_sd()
        print(f"  - SD imported: {copied}")

    print("[2] Cleaning dataset...")
    curated = filter_images()
    print(f"  - Curated images: {curated}")

    if resize_labeled:
        print("[3] Resizing labeled dataset...")
        resized = resize_all()
        print(f"  - Resized labeled images: {resized}")

    print("[4] Training model...")
    train_result = train()

    print("[5] Exporting ONNX...")
    export(model_version=str(train_result.get("model_version", "baseline")))

    print("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StreetPulse ML pipeline")
    parser.add_argument("--source", choices=["api", "sd", "both"], default="api")
    parser.add_argument("--resize-labeled", action="store_true")
    args = parser.parse_args()

    run(source=args.source, resize_labeled=args.resize_labeled)

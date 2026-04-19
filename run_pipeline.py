from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from app.config import (
    CURATED_DIR,
    INFER_BATCH_SIZE,
    LABELED_DIR,
    LABELS,
    PIPELINE_MAX_RETRIES,
    PIPELINE_RETRY_DELAY_SECONDS,
    PIPELINE_RUNS_DIR,
    PIPELINE_STATE_PATH,
    ensure_directories,
)
from app.inference import predict
from app.versioning import snapshot_dataset_version
from ingestion.from_api import fetch_images
from ingestion.from_sd import import_sd
from pipeline.state_machine import PipelineRunner, StageStatus
from processing.clean import filter_images_with_report
from processing.resize import resize_all


_VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _iter_curated_images() -> list[Path]:
    if not CURATED_DIR.exists():
        return []
    return [
        path
        for path in sorted(CURATED_DIR.rglob("*"))
        if path.is_file() and path.suffix.lower() in _VALID_EXTENSIONS
    ]


def _missing_labeled_classes() -> list[str]:
    missing: list[str] = []
    valid_ext = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

    for label in LABELS:
        label_dir = LABELED_DIR / label
        has_images = label_dir.is_dir() and any(
            f.is_file() and f.suffix.lower() in valid_ext
            for f in label_dir.iterdir()
        )
        if not has_images:
            missing.append(str(label_dir))
    return missing


def run(source: str = "api", resize_labeled: bool = False, sd_path: Path | None = None) -> None:
    ensure_directories()
    run_id = f"pipeline_{_now_stamp()}"
    run_state_path = PIPELINE_RUNS_DIR / f"{run_id}_state.json"
    runner = PipelineRunner(
        run_id=run_id,
        state_path=run_state_path,
        latest_state_path=PIPELINE_STATE_PATH,
        max_retries=PIPELINE_MAX_RETRIES,
        retry_delay_seconds=PIPELINE_RETRY_DELAY_SECONDS,
    )
    stage_outputs: dict[str, dict[str, object]] = {}

    def _stage_ingest() -> dict[str, object]:
        api_downloaded = 0
        sd_imported = 0
        if source in {"api", "both"}:
            api_downloaded = fetch_images()
        if source in {"sd", "both"}:
            sd_imported = import_sd(sd_path=sd_path)
        return {
            "source": source,
            "sd_path": str(sd_path) if sd_path else None,
            "api_downloaded": api_downloaded,
            "sd_imported": sd_imported,
            "total_ingested": api_downloaded + sd_imported,
        }

    def _stage_validate() -> dict[str, object]:
        report = filter_images_with_report()
        dataset_version = snapshot_dataset_version(reason="validate_stage")
        out = {**report, "dataset_version": dataset_version["version"]}
        return out

    def _stage_infer() -> dict[str, object]:
        files = _iter_curated_images()
        total = 0
        failed = 0
        uncertain = 0
        high_conf = 0

        for i in range(0, len(files), INFER_BATCH_SIZE):
            batch = files[i : i + INFER_BATCH_SIZE]
            for image_path in batch:
                total += 1
                try:
                    with image_path.open("rb") as handle:
                        result = predict(handle, source_id=str(image_path.resolve().relative_to(Path.cwd().resolve())))
                    label = str(result["label"])
                    confidence = float(result["confidence"])
                    if label == "uncertain":
                        uncertain += 1
                    if confidence >= 0.9 and label != "uncertain":
                        high_conf += 1
                except Exception:
                    failed += 1

        return {
            "processed": total,
            "failed": failed,
            "uncertain": uncertain,
            "high_confidence": high_conf,
            "batch_size": INFER_BATCH_SIZE,
        }

    def _stage_postprocess() -> dict[str, object]:
        resized = 0
        if resize_labeled:
            resized = resize_all()

        missing = _missing_labeled_classes()
        if missing:
            # Keep pipeline usable during data collection: skip training until all classes exist.
            dataset_version = snapshot_dataset_version(reason="postprocess_skipped")
            return {
                "resized_labeled": resized,
                "skipped": True,
                "reason": "missing_labeled_classes",
                "missing_dirs": missing,
                "dataset_version": dataset_version["version"],
            }

        from training.export_onnx import export
        from training.train import train

        train_result = train()
        model_version = str(train_result.get("model_version", "baseline"))
        export_result = export(model_version=model_version)
        dataset_version = snapshot_dataset_version(reason="postprocess_stage")

        return {
            "resized_labeled": resized,
            "train": train_result,
            "export": export_result,
            "dataset_version": dataset_version["version"],
        }

    def _stage_send() -> dict[str, object]:
        summary_path = PIPELINE_RUNS_DIR / f"{run_id}_summary.json"
        summary = {
            "run_id": run_id,
            "stages": stage_outputs,
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {"summary_path": str(summary_path)}

    stage_map = [
        ("INGEST", _stage_ingest),
        ("VALIDATE", _stage_validate),
        ("INFER", _stage_infer),
        ("POSTPROCESS", _stage_postprocess),
        ("SEND", _stage_send),
    ]

    try:
        for stage_name, stage_action in stage_map:
            print(f"[{stage_name}] running...")
            result = runner.run_stage(stage_name, stage_action)
            if result.status != StageStatus.success:
                runner.finalize(success=False, error=result.error)
                raise RuntimeError(f"Stage {stage_name} failed after {result.attempts} attempts: {result.error}")
            stage_outputs[stage_name] = result.output or {}
            print(f"[{stage_name}] success")

        runner.finalize(success=True)
        print(f"DONE ({run_id})")
    except Exception:
        if (PIPELINE_STATE_PATH.exists() and json.loads(PIPELINE_STATE_PATH.read_text(encoding="utf-8")).get("status") != "failed"):
            runner.finalize(success=False, error="Pipeline terminated unexpectedly")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StreetPulse ML pipeline")
    parser.add_argument("--source", choices=["api", "sd", "both"], default="api")
    parser.add_argument("--resize-labeled", action="store_true")
    parser.add_argument("--sd-path", type=Path, default=None, help="Optional SD folder path override (e.g., D:/ride_002)")
    args = parser.parse_args()

    run(source=args.source, resize_labeled=args.resize_labeled, sd_path=args.sd_path)

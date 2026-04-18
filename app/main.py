from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from urllib.parse import quote

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import (
    AUTO_TRAIN_THRESHOLD,
    CURATED_DIR,
    CURATED_STREAM_DIR,
    DATASET_DIR,
    HARD_NEGATIVE_CONFIDENCE,
    HARD_NEGATIVE_DIR,
    LABELED_DIR,
    LABELS,
    ACTIVE_LEARNING_QUEUE_PATH,
    AUTO_LABEL_QUEUE_PATH,
    MODEL_VERSION_PATH,
    PIPELINE_RUNS_DIR,
    PIPELINE_STATE_PATH,
    RAW_DIR,
    STREAM_RAW_DIR,
    STREAM_STALE_HOURS,
    ensure_directories,
)
from app import metadata as meta
from app.inference import predict
from app.model_loader import load_model, reload_model
from app.versioning import snapshot_dataset_version
from ingestion.from_sd import import_sd

log = logging.getLogger(__name__)


class StreamConnectionManager:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.discard(websocket)

    async def broadcast(self, message: dict[str, object]) -> None:
        stale: list[WebSocket] = []
        for websocket in list(self._connections):
            try:
                await websocket.send_json(message)
            except Exception:
                stale.append(websocket)
        for websocket in stale:
            self.disconnect(websocket)


class LabelRequest(BaseModel):
    filename: str
    label: str
    source_id: str | None = None
    path: str | None = None
    predicted_label: str | None = None
    confidence: float | None = None
    model_version: str | None = None


class SdIngestRequest(BaseModel):
    path: str | None = None


ML_ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "http://localhost:5173")
ML_ADMIN_API_KEY = os.getenv("ML_ADMIN_API_KEY")

app = FastAPI(title="StreetPulse ML Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ML_ALLOWED_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ensure_directories()
app.mount("/stream/image", StaticFiles(directory=str(CURATED_STREAM_DIR)), name="stream_image")


def _require_admin_key(x_admin_key: str | None = Header(None, alias="x-admin-key")) -> None:
    if ML_ADMIN_API_KEY and x_admin_key != ML_ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing admin key")

stream_connections = StreamConnectionManager()

_VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@app.on_event("startup")
def startup() -> None:
    ensure_directories()
    try:
        load_model()
        log.info("Model warm-loaded on startup.")
    except FileNotFoundError:
        log.warning("ONNX model not found — inference will be unavailable until a model is placed in models/.")


def _resolve_dataset_path(relative_path: str) -> Path:
    normalized = str(relative_path).strip().replace("\\", "/")
    if normalized.startswith("dataset/"):
        normalized = normalized[len("dataset/") :]

    candidate = (DATASET_DIR / normalized).resolve()
    dataset_root = DATASET_DIR.resolve()
    try:
        candidate.relative_to(dataset_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid dataset path")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return candidate


def _safe_relative(path: Path) -> str:
    return path.resolve().relative_to(DATASET_DIR.resolve()).as_posix()


def _collect_images(directory: Path) -> list[dict[str, str]]:
    if not directory.exists():
        return []

    rows: list[dict[str, str]] = []
    for file_path in sorted(directory.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in _VALID_EXTENSIONS:
            continue
        rows.append(
            {
                "name": file_path.name,
                "path": _safe_relative(file_path),
                "folder": file_path.parent.name,
            }
        )
    return rows


def _existing_labeled_path(filename: str) -> Path | None:
    name = Path(filename).name
    if not name:
        return None
    for label in LABELS:
        candidate = LABELED_DIR / label / name
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _resolve_by_filename(filename: str) -> Path | None:
    name = Path(filename).name
    if not name:
        return None
    for base in (CURATED_DIR, RAW_DIR, CURATED_STREAM_DIR):
        candidate = base / name
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "StreetPulse ML Backend Running"}


@app.get("/health")
def health() -> dict[str, str]:
    try:
        load_model()
        model_version = MODEL_VERSION_PATH.read_text(encoding="utf-8").strip() if MODEL_VERSION_PATH.exists() else "unknown"
        return {"status": "ok", "model": "loaded", "model_version": model_version}
    except FileNotFoundError:
        return {"status": "degraded", "model": "missing", "model_version": "unknown"}


@app.get("/images")
def get_images(source: str = "curated") -> dict[str, object]:
    source_map = {
        "raw": RAW_DIR,
        "curated": CURATED_DIR,
        "labeled": LABELED_DIR,
    }
    if source not in source_map:
        raise HTTPException(status_code=400, detail="source must be raw, curated, or labeled")

    images = _collect_images(source_map[source])
    return {"source": source, "count": len(images), "images": images}


@app.get("/images/file")
def get_image_file(path: str) -> FileResponse:
    image_path = _resolve_dataset_path(path)
    return FileResponse(image_path)


@app.post("/ingest/sd", dependencies=[Depends(_require_admin_key)])
def ingest_sd(payload: SdIngestRequest) -> dict[str, object]:
    source_path: Path | None = None
    if payload.path is not None:
        candidate = Path(payload.path).expanduser()
        if not candidate.exists() or not candidate.is_dir():
            raise HTTPException(status_code=400, detail=f"SD path not found or not a directory: {candidate}")
        source_path = candidate

    copied = import_sd(sd_path=source_path)
    return {
        "status": "success",
        "source_path": str(source_path) if source_path else None,
        "copied": copied,
    }


@app.post("/label", dependencies=[Depends(_require_admin_key)])
async def label_image(payload: LabelRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    if payload.label not in LABELS:
        raise HTTPException(status_code=400, detail=f"label must be one of: {', '.join(LABELS)}")

    normalized_name = Path(payload.filename).name
    if normalized_name:
        already_labeled = _existing_labeled_path(normalized_name)
        if already_labeled is not None:
            new_path = Path("dataset") / already_labeled.resolve().relative_to(DATASET_DIR.resolve())
            return {"status": "success", "new_path": str(new_path).replace("\\", "/")}

    source_file: Path | None = None
    source_name = normalized_name

    if payload.path:
        try:
            source_file = _resolve_dataset_path(payload.path)
            source_name = source_file.name
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
    elif payload.source_id:
        resolved = _source_id_to_dataset_path(payload.source_id)
        if resolved is not None:
            source_file = _resolve_dataset_path(resolved)
            source_name = source_file.name
    else:
        if not normalized_name:
            raise HTTPException(status_code=400, detail="filename is required")

        curated_root = CURATED_STREAM_DIR.resolve()
        stream_file = (CURATED_STREAM_DIR / normalized_name).resolve()
        try:
            stream_file.relative_to(curated_root)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid filename")

        if not stream_file.exists() or not stream_file.is_file():
            raise HTTPException(status_code=404, detail="Source curated stream image not found")
        source_file = stream_file
        source_name = normalized_name

    if source_file is None and source_name:
        source_file = _resolve_by_filename(source_name)

    if source_file is None:
        already_labeled = _existing_labeled_path(source_name)
        if already_labeled is not None:
            new_path = Path("dataset") / already_labeled.resolve().relative_to(DATASET_DIR.resolve())
            return {"status": "success", "new_path": str(new_path).replace("\\", "/")}
        raise HTTPException(status_code=404, detail="Source image not found")

    destination_dir = LABELED_DIR / payload.label
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source_name

    if destination.exists():
        new_path = Path("dataset") / destination.resolve().relative_to(DATASET_DIR.resolve())
        return {"status": "success", "new_path": str(new_path).replace("\\", "/")}

    try:
        shutil.move(str(source_file), str(destination))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to move file for labeling") from exc

    new_path = Path("dataset") / "labeled" / payload.label / source_name

    background_tasks.add_task(
        _post_label_tasks,
        filename=source_name,
        label=payload.label,
        source_id=payload.source_id or payload.path or source_name,
        predicted_label=payload.predicted_label,
        confidence=payload.confidence,
        model_version=payload.model_version,
        labeled_path=destination,
    )

    return {"status": "success", "new_path": str(new_path).replace("\\", "/")}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)) -> dict[str, float | str]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        return predict(file.file, source_id=file.filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Inference failed") from exc


@app.websocket("/ws/stream")
async def stream_websocket(websocket: WebSocket) -> None:
    await stream_connections.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        stream_connections.disconnect(websocket)
    except Exception:
        stream_connections.disconnect(websocket)


@app.post("/stream-image")
async def stream_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> dict[str, str]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _VALID_EXTENSIONS:
        suffix = ".jpg"

    filename = f"{uuid4().hex}{suffix}"
    saved_path = STREAM_RAW_DIR / filename

    try:
        STREAM_RAW_DIR.mkdir(parents=True, exist_ok=True)
        contents = await file.read()
        saved_path.write_bytes(contents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to save uploaded image") from exc

    background_tasks.add_task(_process_stream_image, saved_path, filename)
    return {"status": "accepted", "filename": filename}


async def _process_stream_image(saved_path: Path, filename: str) -> None:
    """Background task: infer from disk, move raw→curated/stream, broadcast WS."""
    loop = asyncio.get_running_loop()
    try:
        image_bytes = saved_path.read_bytes()
        from io import BytesIO
        prediction = await loop.run_in_executor(None, predict, BytesIO(image_bytes), filename)

        CURATED_STREAM_DIR.mkdir(parents=True, exist_ok=True)
        dest = CURATED_STREAM_DIR / filename
        shutil.move(str(saved_path), str(dest))

        payload: dict[str, object] = {
            "type": "new_image",
            "image_url": f"/stream/image/{filename}",
            "prediction": {
                "label": prediction["label"],
                "confidence": prediction["confidence"],
                "model_version": prediction.get("model_version", "unknown"),
            },
        }
        await stream_connections.broadcast(payload)
    except Exception:
        log.exception("Stream image processing failed for %s", filename)
        if saved_path.exists():
            saved_path.unlink(missing_ok=True)


async def _post_label_tasks(
    *,
    filename: str,
    label: str,
    source_id: str | None,
    predicted_label: str | None,
    confidence: float | None,
    model_version: str | None,
    labeled_path: Path,
) -> None:
    """Broadcast labeled event, persist metadata, and trigger auto-train if threshold reached."""
    await stream_connections.broadcast({"type": "labeled", "filename": filename})

    meta.record_label(
        filename,
        label,
        source="stream",
        predicted_label=predicted_label,
        confidence=confidence,
        model_version=model_version,
    )
    snapshot_dataset_version(reason="manual_label")
    if source_id:
        _remove_from_queues(source_id)

    if _should_promote_hard_negative(predicted_label=predicted_label, true_label=label, confidence=confidence):
        _copy_to_hard_negative(labeled_path, label)
        if predicted_label is not None:
            meta.record_hard_negative(
                filename=filename,
                predicted_label=predicted_label,
                true_label=label,
                confidence=confidence,
                model_version=model_version,
            )

    if predicted_label is not None and predicted_label != label:
        meta.record_autolabel_correction(
            filename=filename,
            predicted_label=predicted_label,
            true_label=label,
            confidence=confidence,
            model_version=model_version,
        )

    total = meta.count_labeled()
    if total > 0 and total % AUTO_TRAIN_THRESHOLD == 0:
        log.info("Auto-train threshold reached (%d labeled images). Starting background training.", total)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _run_auto_train)


def _run_auto_train() -> None:
    try:
        from training.train import train
        from training.export_onnx import export

        result = train()
        model_version = str(result.get("model_version", "baseline"))
        export(model_version=model_version)
        reload_model()
        log.info("Auto-train complete. Model %s reloaded.", model_version)
    except Exception:
        log.exception("Auto-train failed.")


def _should_promote_hard_negative(*, predicted_label: str | None, true_label: str, confidence: float | None) -> bool:
    if predicted_label is None or confidence is None:
        return False
    if predicted_label == true_label:
        return False
    if confidence < HARD_NEGATIVE_CONFIDENCE:
        return False

    # Focus on high-confidence normal mistakes and other high-confidence misclassifications.
    if predicted_label == "normal" and true_label != "normal":
        return True
    return True


def _copy_to_hard_negative(image_path: Path, true_label: str) -> None:
    destination_dir = HARD_NEGATIVE_DIR / true_label
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / image_path.name
    if destination.exists():
        destination = destination_dir / f"{image_path.stem}_{uuid4().hex[:8]}{image_path.suffix}"
    shutil.copy2(image_path, destination)


def _source_id_to_dataset_path(source_id: str | None) -> str | None:
    if not source_id:
        return None

    raw = str(source_id).strip().replace("\\", "/")
    if not Path(raw).name:
        return None

    dataset_root = DATASET_DIR.resolve()

    normalized = raw
    lower = normalized.lower()

    # source_id often includes "dataset/..."; strip it for /images/file path.
    if lower.startswith("dataset/"):
        normalized = normalized[len("dataset/") :]
        lower = normalized.lower()

    # For absolute source IDs, avoid expensive resolve() on external drives.
    # Only accept paths that clearly include a dataset segment.
    marker = "/dataset/"
    marker_idx = lower.find(marker)
    if marker_idx >= 0:
        normalized = normalized[marker_idx + len(marker) :]
        lower = normalized.lower()

    if lower.startswith(("curated/", "raw/", "labeled/", "hard_negatives/")):
        direct_abs = (dataset_root / Path(normalized)).resolve()
        try:
            direct_abs.relative_to(dataset_root)
            if direct_abs.exists() and direct_abs.is_file():
                return direct_abs.relative_to(dataset_root).as_posix()
        except ValueError:
            pass

    stream_guess = CURATED_STREAM_DIR / Path(raw).name
    if stream_guess.exists() and stream_guess.is_file():
        return stream_guess.resolve().relative_to(dataset_root).as_posix()

    return None


def _queue_items(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []

    by_source: dict[str, dict[str, object]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id") or "").strip()
        if not source_id:
            continue

        image_path = _source_id_to_dataset_path(source_id)

        image_url = None
        if image_path:
            image_url = f"/images/file?path={quote(image_path)}"
        else:
            stream_name = Path(source_id).name
            if stream_name:
                image_url = f"/stream/image/{quote(stream_name)}"

        if image_url is None:
            # Skip unresolved queue records; Ops Center can only preview items with a valid image URL.
            continue

        by_source[source_id] = {
            "source_id": source_id,
            "image_path": image_path,
            "image_url": image_url,
            "prediction": item.get("label"),
            "confidence": item.get("confidence"),
            "model_version": item.get("model_version"),
            "timestamp": item.get("timestamp"),
        }

    # Most recent first by timestamp string (ISO format).
    rows = list(by_source.values())
    rows.sort(key=lambda row: str(row.get("timestamp") or ""), reverse=True)
    return rows


def _remove_from_queue(path: Path, source_id: str) -> None:
    if not path.exists():
        return
    out: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        if str(item.get("source_id") or "") == source_id:
            continue
        out.append(json.dumps(item, ensure_ascii=True))
    path.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")


def _remove_from_queues(source_id: str) -> None:
    _remove_from_queue(AUTO_LABEL_QUEUE_PATH, source_id)
    _remove_from_queue(ACTIVE_LEARNING_QUEUE_PATH, source_id)


@app.get("/queue/active")
def queue_active() -> dict[str, object]:
    items = _queue_items(ACTIVE_LEARNING_QUEUE_PATH)
    return {"count": len(items), "items": items}


@app.get("/queue/auto-label")
def queue_auto_label() -> dict[str, object]:
    items = _queue_items(AUTO_LABEL_QUEUE_PATH)
    return {"count": len(items), "items": items}


def _pipeline_progress_from_state(state: dict[str, object]) -> dict[str, object]:
    stages = state.get("stages")
    stages_dict = stages if isinstance(stages, dict) else {}
    total = 5
    completed = sum(1 for v in stages_dict.values() if isinstance(v, dict) and v.get("status") == "success")

    infer = stages_dict.get("INFER") if isinstance(stages_dict.get("INFER"), dict) else {}
    validate = stages_dict.get("VALIDATE") if isinstance(stages_dict.get("VALIDATE"), dict) else {}
    infer_output = infer.get("output") if isinstance(infer.get("output"), dict) else {}
    validate_output = validate.get("output") if isinstance(validate.get("output"), dict) else {}

    rejected = 0
    for key in ("invalid", "too_small", "too_dark", "too_blurry"):
        val = validate_output.get(key)
        if isinstance(val, int):
            rejected += val

    metrics = {
        "images_processed": int(infer_output.get("processed", 0) or 0),
        "rejected_images": rejected,
        "auto_label_count": len(_queue_items(AUTO_LABEL_QUEUE_PATH)),
        "active_learning_count": len(_queue_items(ACTIVE_LEARNING_QUEUE_PATH)),
    }

    return {
        "run_id": state.get("run_id"),
        "status": state.get("status", "unknown"),
        "current_stage": state.get("current_stage"),
        "progress": {
            "completed_stages": completed,
            "total_stages": total,
            "percent": round((completed / total) * 100, 1),
        },
        "metrics": metrics,
        "stages": stages_dict,
        "started_at": state.get("started_at"),
        "finished_at": state.get("finished_at"),
        "error": state.get("error"),
    }


@app.get("/pipeline/status/latest")
def pipeline_status_latest() -> dict[str, object]:
    if not PIPELINE_STATE_PATH.exists():
        raise HTTPException(status_code=404, detail="No pipeline state available")
    try:
        state = json.loads(PIPELINE_STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=500, detail="Failed to read latest pipeline state") from exc
    if not isinstance(state, dict):
        raise HTTPException(status_code=500, detail="Invalid latest pipeline state format")
    return _pipeline_progress_from_state(state)


@app.get("/pipeline/status/{run_id}")
def pipeline_status(run_id: str) -> dict[str, object]:
    state_file = PIPELINE_RUNS_DIR / f"{run_id}_state.json"
    if not state_file.exists():
        raise HTTPException(status_code=404, detail=f"Pipeline run not found: {run_id}")
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=500, detail="Failed to read pipeline state") from exc
    if not isinstance(state, dict):
        raise HTTPException(status_code=500, detail="Invalid pipeline state format")
    return _pipeline_progress_from_state(state)


@app.delete("/admin/cleanup-stream")
async def cleanup_stream() -> dict[str, object]:
    """Remove stale unlabeled images from curated/stream older than STREAM_STALE_HOURS."""
    cutoff = datetime.now(timezone.utc).timestamp() - STREAM_STALE_HOURS * 3600
    removed: list[str] = []
    if CURATED_STREAM_DIR.exists():
        for f in CURATED_STREAM_DIR.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
                removed.append(f.name)
    if STREAM_RAW_DIR.exists():
        for f in STREAM_RAW_DIR.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
                removed.append(f"raw/{f.name}")
    return {"removed": len(removed), "files": removed}


@app.get("/dataset/metadata")
def get_metadata() -> dict[str, object]:
    """Return the full dataset labeling index."""
    index = meta.load_index()
    return {"count": len(index), "entries": index}


# ---------------------------------------------------------------------------
# Dashboard decision-tool endpoint
# ---------------------------------------------------------------------------

_SEVERITY: dict[str, int] = {
    "pothole": 5,
    "crack": 4,
    "manhole": 3,
    "normal": 1,
    "uncertain": 0,
}


@app.get("/dashboard/stats")
def dashboard_stats() -> dict[str, object]:
    from collections import defaultdict
    from datetime import timedelta

    index = meta.load_index()

    # Split event types.
    label_events = [e for e in index if not e.get("type")]
    correction_events = [e for e in index if e.get("type") == "autolabel_correction"]

    total_labeled = len(label_events)
    by_class: dict[str, int] = defaultdict(int)
    conf_sum: dict[str, float] = defaultdict(float)
    conf_count: dict[str, int] = defaultdict(int)

    for event in label_events:
        lbl = str(event.get("label", "unknown"))
        by_class[lbl] += 1
        conf = event.get("confidence")
        if isinstance(conf, (int, float)):
            conf_sum[lbl] += float(conf)
            conf_count[lbl] += 1

    # Trends: last 30 days, labels by date.
    now = datetime.now(timezone.utc)
    trend_window = 30
    daily: dict[str, dict[str, int]] = {}
    for d in range(trend_window - 1, -1, -1):
        day_str = (now - timedelta(days=d)).strftime("%Y-%m-%d")
        daily[day_str] = {lbl: 0 for lbl in LABELS}

    for event in label_events:
        ts_raw = event.get("timestamp")
        if not isinstance(ts_raw, str):
            continue
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        except ValueError:
            continue
        day_str = ts.strftime("%Y-%m-%d")
        if day_str in daily:
            lbl = str(event.get("label", ""))
            if lbl in daily[day_str]:
                daily[day_str][lbl] += 1

    trends = [{"date": d, **counts} for d, counts in daily.items()]

    # Recent-window counts (last 7 days).
    recent_cutoff = now - timedelta(days=7)
    recent_by_class: dict[str, int] = defaultdict(int)
    for event in label_events:
        ts_raw = event.get("timestamp")
        if not isinstance(ts_raw, str):
            continue
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        except ValueError:
            continue
        if ts >= recent_cutoff:
            lbl = str(event.get("label", "unknown"))
            recent_by_class[lbl] += 1

    # Seven-day-prior window for trend direction.
    prior_cutoff = now - timedelta(days=14)
    prior_by_class: dict[str, int] = defaultdict(int)
    for event in label_events:
        ts_raw = event.get("timestamp")
        if not isinstance(ts_raw, str):
            continue
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        except ValueError:
            continue
        if prior_cutoff <= ts < recent_cutoff:
            lbl = str(event.get("label", "unknown"))
            prior_by_class[lbl] += 1

    # Severity ranking.
    severity_ranking = []
    for lbl in LABELS:
        count = int(by_class.get(lbl, 0))
        recent = int(recent_by_class.get(lbl, 0))
        prior = int(prior_by_class.get(lbl, 0))
        avg_conf = round(conf_sum[lbl] / conf_count[lbl], 4) if conf_count.get(lbl) else None
        sev = _SEVERITY.get(lbl, 0)

        if prior == 0 and recent > 0:
            trend_dir = "up"
        elif recent == 0 and prior > 0:
            trend_dir = "down"
        elif prior > 0:
            change = (recent - prior) / prior
            trend_dir = "up" if change >= 0.15 else "down" if change <= -0.15 else "flat"
        else:
            trend_dir = "flat"

        severity_ranking.append({
            "label": lbl,
            "severity": sev,
            "count": count,
            "recent_count": recent,
            "prior_count": prior,
            "avg_confidence": avg_conf,
            "trend": trend_dir,
        })

    severity_ranking.sort(key=lambda r: (r["severity"] * r["count"], r["recent_count"]), reverse=True)

    # Correction rate.
    correction_rate = round(len(correction_events) / total_labeled, 4) if total_labeled > 0 else 0.0

    # Model version.
    model_version = (
        MODEL_VERSION_PATH.read_text(encoding="utf-8").strip()
        if MODEL_VERSION_PATH.exists()
        else "unknown"
    )

    # Alerts.
    alerts: list[dict[str, str]] = []

    # Data gap alerts.
    for lbl in LABELS:
        if by_class.get(lbl, 0) == 0:
            alerts.append({
                "level": "warning",
                "code": "no_data",
                "label": lbl,
                "message": f"No labeled images for class '{lbl}'",
                "detail": f"The model has never seen any '{lbl}' examples. Add labeled images to improve accuracy.",
            })

    # Model quality alert.
    if model_version in ("unknown", "baseline", "v_baseline_random"):
        alerts.append({
            "level": "warning",
            "code": "baseline_model",
            "label": "",
            "message": "Running on baseline (untrained) model",
            "detail": "Predictions are random. Label images and run the training pipeline to deploy a real model.",
        })

    # High correction rate.
    if correction_rate > 0.15 and total_labeled >= 10:
        alerts.append({
            "level": "warning",
            "code": "high_correction_rate",
            "label": "",
            "message": f"High auto-label correction rate: {round(correction_rate * 100, 1)}%",
            "detail": "More than 15% of auto-labeled images were corrected manually, indicating low model confidence or errors.",
        })

    # Severity spike alerts.
    for entry in severity_ranking:
        if entry["severity"] >= 4 and entry["trend"] == "up" and entry["recent_count"] >= 3:
            prior_c = entry["prior_count"]
            recent_c = entry["recent_count"]
            pct = "" if prior_c == 0 else f" (+{round((recent_c - prior_c) / prior_c * 100)}% vs prior week)"
            alerts.append({
                "level": "critical",
                "code": "severity_spike",
                "label": str(entry["label"]),
                "message": f"Spike in '{entry['label']}' detections{pct}",
                "detail": f"{recent_c} instances labeled in the last 7 days — prioritize field inspection.",
            })

    # Dominant hazard alert (>50% of recent labeled).
    recent_total = sum(recent_by_class.values())
    for lbl, cnt in recent_by_class.items():
        sev = _SEVERITY.get(lbl, 0)
        if sev >= 4 and recent_total > 0 and cnt / recent_total > 0.5:
            alerts.append({
                "level": "critical",
                "code": "dominant_hazard",
                "label": lbl,
                "message": f"'{lbl}' is {round(cnt / recent_total * 100)}% of recent detections",
                "detail": "Road quality is predominantly one hazard type. Dispatch repair crew.",
            })

    # Good news: high overall confidence.
    total_conf = sum(conf_sum.values())
    total_conf_count = sum(conf_count.values())
    if total_conf_count >= 10 and (total_conf / total_conf_count) >= 0.85:
        alerts.append({
            "level": "info",
            "code": "high_confidence",
            "label": "",
            "message": f"Model confidence is high ({round(total_conf / total_conf_count * 100, 1)}% avg)",
            "detail": "Predictions are reliable. Consider lowering the auto-label threshold to speed up labeling.",
        })

    return {
        "summary": {
            "total_labeled": total_labeled,
            "by_class": dict(by_class),
            "correction_rate": correction_rate,
            "model_version": model_version,
        },
        "severity_ranking": severity_ranking,
        "trends": trends,
        "alerts": alerts,
    }

from __future__ import annotations

import asyncio
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import (
    AUTO_TRAIN_THRESHOLD,
    CURATED_DIR,
    CURATED_STREAM_DIR,
    DATASET_DIR,
    LABELED_DIR,
    LABELS,
    RAW_DIR,
    STREAM_RAW_DIR,
    STREAM_STALE_HOURS,
    ensure_directories,
)
from app import metadata as meta
from app.inference import predict
from app.model_loader import load_model, reload_model

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


app = FastAPI(title="StreetPulse ML Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ensure_directories()
app.mount("/stream/image", StaticFiles(directory=str(CURATED_STREAM_DIR)), name="stream_image")

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
    candidate = (DATASET_DIR / relative_path).resolve()
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


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "StreetPulse ML Backend Running"}


@app.get("/health")
def health() -> dict[str, str]:
    try:
        load_model()
        return {"status": "ok", "model": "loaded"}
    except FileNotFoundError:
        return {"status": "degraded", "model": "missing"}


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


@app.post("/label")
async def label_image(payload: LabelRequest, background_tasks: BackgroundTasks) -> dict[str, str]:
    if payload.label not in LABELS:
        raise HTTPException(status_code=400, detail=f"label must be one of: {', '.join(LABELS)}")

    normalized_name = Path(payload.filename).name
    if not normalized_name:
        raise HTTPException(status_code=400, detail="filename is required")

    curated_root = CURATED_STREAM_DIR.resolve()
    source_file = (CURATED_STREAM_DIR / normalized_name).resolve()
    try:
        source_file.relative_to(curated_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not source_file.exists() or not source_file.is_file():
        raise HTTPException(status_code=404, detail="Source curated stream image not found")

    destination_dir = LABELED_DIR / payload.label
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / normalized_name

    if destination.exists():
        raise HTTPException(status_code=409, detail="Destination file already exists")

    try:
        shutil.move(str(source_file), str(destination))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to move file for labeling") from exc

    new_path = Path("dataset") / "labeled" / payload.label / normalized_name

    background_tasks.add_task(
        _post_label_tasks,
        filename=normalized_name,
        label=payload.label,
    )

    return {"status": "success", "new_path": str(new_path).replace("\\", "/")}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)) -> dict[str, float | str]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    try:
        return predict(file.file)
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
        prediction = await loop.run_in_executor(None, predict, BytesIO(image_bytes))

        CURATED_STREAM_DIR.mkdir(parents=True, exist_ok=True)
        dest = CURATED_STREAM_DIR / filename
        shutil.move(str(saved_path), str(dest))

        payload: dict[str, object] = {
            "type": "new_image",
            "image_url": f"/stream/image/{filename}",
            "prediction": {
                "label": prediction["label"],
                "confidence": prediction["confidence"],
            },
        }
        await stream_connections.broadcast(payload)
    except Exception:
        log.exception("Stream image processing failed for %s", filename)
        if saved_path.exists():
            saved_path.unlink(missing_ok=True)


async def _post_label_tasks(*, filename: str, label: str) -> None:
    """Broadcast labeled event, persist metadata, and trigger auto-train if threshold reached."""
    await stream_connections.broadcast({"type": "labeled", "filename": filename})

    meta.record_label(filename, label, source="stream")

    total = meta.count_labeled()
    if total > 0 and total % AUTO_TRAIN_THRESHOLD == 0:
        log.info("Auto-train threshold reached (%d labeled images). Starting background training.", total)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _run_auto_train)


def _run_auto_train() -> None:
    try:
        from training.train import train
        from training.export_onnx import export
        train()
        export()
        reload_model()
        log.info("Auto-train complete. Model reloaded.")
    except Exception:
        log.exception("Auto-train failed.")


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

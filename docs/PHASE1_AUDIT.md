# StreetPulse ML — Phase 1 Audit

> Generated from codebase review + real device session (rad.txt, 2026-05)

## Step 1 — Model Status

**Current model:** `models/latest_model.onnx`
- Size: 88 KB
- Version tag: `v_baseline_random`
- Status: **RANDOM BASELINE** — produces random predictions
- Do not use for production severity scoring

`models/model.pth` (44 MB) exists but has not been exported to ONNX in a trained state.

**Action:** Train on labeled dataset, export to ONNX, replace `latest_model.onnx`.

---

## Step 2 — Image-Only Pipeline Gap

The current pipeline requires an image for classification:
```
run_pipeline.py → INGEST → VALIDATE → INFER → POSTPROCESS → SEND
```

Every stage assumes an image file. IMU-only events (no camera) are not scored.

**Action:** Add `processing/severity.py` as a parallel scoring path for IMU events.
The module is calibrated from real rad.txt device data.

---

## Step 3 — Inference Log Analysis

`models/inference_log.jsonl` (2.2 MB) contains real inference records.

Key observations:
- Confidence distribution is flat (~25% per class) — confirms random model
- Active learning queue filling with low-confidence events (<0.75 threshold)
- Temperature calibration applied but has no effect on random weights

**Action:** Use the active learning queue as the labeling backlog. Priority labels:
1. Events where `shock_magnitude > 5.0` (likely true positives)
2. Events with `gps_valid: true` (usable for location-tagged training data)

---

## Step 4 — Real Device Data (rad.txt)

Session characteristics that affect the ML pipeline:

| Finding | Impact |
|---|---|
| Noise floor 1.41–1.44G | 98% of events are non-events — class imbalance |
| Duration in microseconds | Severity features computed from duration are 1000x wrong |
| GPS lock failure | Cannot associate events with map locations |
| 4 real high-G impacts | Positive training examples: 8.66G, 7.69G, 8.89G, 5.73G |

**Action:** Add `ground_truth_tier` labels to the field test dataset before training.
See `dataset/field_test/sample_telemetry.jsonl`.

---

## Step 5 — IMU Severity Module

New file: `processing/severity.py`

Provides `score_event(peak_g, duration_us, speed_mps)` returning:
- `tier`: noise / negligible / low / moderate / severe / critical
- `defect_type`: pothole / crack / rough_pavement / expansion_joint / normal
- `severity_score`: 0.0–10.0 (matches defects table scale)
- `should_ingest`: bool

Thresholds calibrated from rad.txt:
- Noise floor: 1.50G (above observed 1.44G maximum)
- Critical: > 7.0G
- Severe: > 5.0G
- Moderate: > 3.5G
- Low: > 2.5G

---

## Step 6 — Test Coverage

Current test coverage: **0%** — no test files in `app/` or `processing/`.

**Priority tests:**
1. `severity.py` — table-driven tests against rad.txt known values
2. `inference.py` — mock ONNX session, verify calibration applied
3. `pipeline/state_machine.py` — verify retry logic and JSON state persistence
4. `processing/clean.py` — verify size/brightness/sharpness filters reject bad images

---

## Step 7 — Active Learning Queue

Queue threshold: `confidence < 0.75`
With random model: every event is queued (all confidences ~0.25).

**Action items:**
1. Pause auto-queuing until model is trained — queue will contain noise
2. Once trained, set threshold to 0.75 and begin reviewing queue
3. Hard negative mining (already implemented in `app/main.py`) — activate after first model

---

## Step 8 — Pipeline State Machine

`pipeline/state_machine.py` — 5-stage JSON-persisted pipeline with retry.

**Issues found:**
1. No timeout per stage — a hung INFER stage blocks forever
2. State file path hardcoded — not configurable via env var
3. SEND stage has no idempotency check — may double-send on retry

**Action:** Add per-stage timeout (default 300s), make state path env-configurable,
add content-hash deduplication on SEND.

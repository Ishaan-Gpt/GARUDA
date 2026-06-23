# GARUDA — Backend Reference Guide
**For frontend developers and integration partners**
*(Pitch name: "Gridlock Guardian" — Flipkart Gridlock 3.0. Code/API/DB still use "GARUDA" internally.)*

---

## What is GARUDA?

GARUDA is an **edge-native, automated traffic violation detection system**. A camera feed or uploaded image/video is processed through a multi-stage ML pipeline that detects violations, reads license plates, and either auto-issues challans or escalates uncertain cases to patrol officers — directly addressing the Flipkart Gridlock problem statement "Automated Photo Identification and Classification for Traffic Violations Using Computer Vision" (`ps.txt`).

---

## Status as of 2026-06-22 (post-refactor)

- **Backend**: all endpoints below are real and working against a live SQLite DB — not mocked. 13 routers, 2 WebSocket endpoints, auth + RBAC, an audit trail, and a local LLM ops agent are all wired into `backend/main.py`.
- **ML models**: 7 trained weight files are loaded and used in the live pipeline (not placeholders) — see the model table below for exact accuracy/mAP numbers pulled from their metrics JSON files.
- **Live data path**: `POST /api/v1/jobs/upload` runs the real ML pipeline (preprocess → detect → classify → OCR) in a background task and writes violations straight to the DB. `python ml/demo_pipeline.py --input <image> --backend-url http://localhost:8000` is the CLI equivalent, useful for local debugging. `/debug/inject-violation` still exists for pure UI testing with fake data — don't confuse its output with real detections.
- **Not implemented**: cross-camera vehicle re-identification; federated learning is wired (`ml/federated/`) but doesn't retrain from real edge data yet; there is **no standalone evaluation harness** that reports Accuracy/Precision/Recall/F1/mAP end-to-end across the whole pipeline — only per-model training metrics exist (see "ps.txt Coverage" below).
- **2026-06-22 architectural refactor**:
  - Extracted `cameras`, `vehicles`, `analytics`, `stream`, and `debug` from the 707-line `_routers.py` god-file into their own standalone routers.
  - Created `backend/services/` layer with three reusable services: `MLRegistry` (shared model singleton), `CalibrationService` (camera calibration), `ChallanService` (violation packaging + tier routing).
  - Eliminated the duplicate-ML-singleton bug — both `jobs.py` and `stream.py`'s patrol WebSocket now share one `MLRegistry` instance loaded via `get_ml_registry()`.
  - Reorganised `ml/models/weights/` into `detection/`, `violations/`, `ocr/`, `metrics/` subdirectories.
  - All weight paths updated in `detector.py`, `ocr.py`, and `violation_classifier.py`.
- **2026-06-23 violation-logic audit** (found by testing against real video, not just unit tests — see `ml/pipeline/violation_classifier.py`):
  - `check_red_light()` required only that the vehicle be *anywhere* past the stop line in the recent window, not that it actually crossed — a vehicle already parked past the line before the light turned red was wrongly classified as running it. Now requires evidence the vehicle was on the legal side earlier in the same window.
  - `check_illegal_parking()`'s 5-minute timer used `time.monotonic()` (wall-clock processing time), which has nothing to do with video time for batch/offline jobs. Now anchored to a frame counter scaled by `self.fps`.
  - `check_wrong_side()` used a raw velocity dot-product threshold, which a vehicle merely turning or changing lanes could trip. Now uses a heading-angle threshold (`WRONG_SIDE_ANGLE_THRESHOLD_DEG = 100`) plus a 2-consecutive-frame persistence counter (`WRONG_SIDE_STRIKES_REQUIRED`) before confirming.
  - `check_stop_line()` had no confidence floor on the signal reading at all (unlike `check_red_light()`'s 0.60 floor) — a marginal ~0.4-0.5-confidence signal misdetection (e.g. a stray red taillight/sign) was enough to cite a stopped vehicle with no real traffic light nearby. Added the same 0.60 floor to both the tracked and static-fallback versions.
  - The signal-state debounce buffer (`_smooth_signal_state`) could be dominated by a single isolated false-positive detection (frames with no light detected never entered the buffer, so one stray reading was 100% of it) and never went stale, so two false detections minutes apart could "agree" with each other. Now requires ≥2 agreeing samples and discards the buffer after a detection gap longer than the smoothing window.
  - `AIHelmetViolationDetector.detect()`'s crop-based fallback derived the head region from the **motorcycle's own bbox**, which is frequently tight around just the bike + lower body and excludes an upright rider's head entirely — confirmed visually on real footage where the crop captured the rider's torso, not their head. Now prefers the bbox of the rider's already-associated person detection (`associate_riders_with_vehicles()`), which reliably includes the head. Also added a confidence floor on `helmet` (compliant) detections symmetric to the existing one on `head` (no-helmet) detections — a single 0.19-confidence "wearing helmet" read was previously enough to silently clear an actually bare-headed rider.
  - **Known residual limitation, not fixed by the above**: even with the corrected head crop, the trained helmet CNN (`helmet_cnn.pt`) can still confidently (>0.99) misclassify a bare head as helmeted on very small/blurry/distant crops — that's a model-accuracy limitation requiring retraining, not a logic bug. Likewise, ByteTrack's track-ID continuity is unreliable in heavily congested/occluded scenes (observed both as a parking-timer reset and as a wrong-side velocity sign flip on the same physical vehicle) — calibrating tight zones helps, but isn't a complete fix.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and fill env
cp .env.example .env

# 3. Start backend
uvicorn backend.main:app --reload --port 8000

# 4. Open Swagger docs
open http://localhost:8000/docs

# 5. Open frontend dashboard
open frontend/index.html

# 6. Feed it real ML detections via the job queue (recommended):
curl -F "name=test" -F "source_type=Image" -F "file=@test/sample.jpg" \
     http://localhost:8000/api/v1/jobs/upload

# 6b. Or via the standalone CLI pipeline:
python ml/demo_pipeline.py --input sample.jpg --backend-url http://localhost:8000
```

---

## System Architecture

```
Image / Video / Camera Feed
    ↓
[Preprocessor]   ml/pipeline/preprocessor.py — CLAHE + denoise + gamma correction
    ↓
[Detector]       ml/pipeline/detector.py — YOLOv8m (yolov8m.pt) → vehicles, persons, phones
    ↓
[Tracker]        ml/pipeline/tracker.py — ByteTrack (video only) → persistent track IDs, velocity, history
    ↓
[Violation Classifier]   ml/pipeline/violation_classifier.py — 9 violation types (see table below)
    │   ├─ Helmet:   AICity 9-class detector (helmet_best.pt) on full image, falls back to
    │   │            crop CNN (helmet_cnn.pt) or edge/colour heuristic. Reports the model's
    │   │            real per-box confidence (fixed 2026-06-21 — was a hardcoded 0.50 stub).
    │   │            Used ONLY for helmet — does not feed triple-riding. The crop-based
    │   │            fallback's head region now comes from the rider's own person bbox, not
    │   │            the motorcycle's bbox (fixed 2026-06-23 — a motorcycle's bbox frequently
    │   │            excludes an upright rider's head entirely, silently breaking the crop).
    │   ├─ Triple riding: fully independent of the helmet model. Person + motorbike
    │   │            detections from the main YOLOv8m detector are grouped by IoU/position
    │   │            via associate_riders_with_vehicles() (nearest-rider-first, capped at
    │   │            4/vehicle, each person assigned to at most one vehicle per frame) —
    │   │            ported from temp/AI_Traffic_Violation_Detection_triple_riding_detection's
    │   │            group_riders_with_vehicles()
    │   ├─ Seatbelt: windshield-ROI YOLOv11s classifier (seatbelt_classifier.pt), Hough-line
    │   │            fallback if weights missing
    │   ├─ Signal:   traffic_lights_yolov8x.pt, falls back to HSV colour heuristic.
    │   │            Readings are majority-vote smoothed over the last 5 frames before
    │   │            being trusted (fixed 2026-06-23 — a single isolated false-positive
    │   │            detection used to be enough to confirm a colour change outright)
    │   └─ Wrong-side / stop-line / red-light / illegal-parking: tracker-based when video
    │       tracking state exists; static-position fallback when only a single image
    │       exists. Stop-line and red-light fallbacks gate on signal-detection
    │       confidence (both now floored at 0.60 — fixed 2026-06-23, stop-line previously
    │       had no floor at all); wrong-side fallback is a position-only proxy (no heading
    │       signal in a still frame) against a per-camera `wrong_side_lane` setting,
    │       not a universal rule — calibrate it like `stop_line_y`. The tracked red-light
    │       check now requires an actual crossing (legal-side-then-violating-side), not
    │       mere presence past the line; the tracked wrong-side check now uses a heading-
    │       angle threshold + 2-frame persistence instead of a raw velocity dot product;
    │       illegal-parking's duration timer is anchored to a frame counter instead of
    │       wall-clock time (all fixed 2026-06-23 — see Status section above)
    ↓
[Driver State]   ml/pipeline/driver_state.py — MediaPipe FaceMesh → drowsiness, yawn, phone-in-hand
    ↓
[OCR]            ml/pipeline/ocr.py — 2-stage plate detection (Koushi → YasirFaiz) +
    │            text engine fallback chain: fast-plate-ocr → PaddleOCR → EasyOCR → Tesseract
    ↓
[Confidence Router]  ml/pipeline/confidence_router.py — 3-tier routing decision
    ↓
┌─────────────────────┐
│  TIER 1 (conf≥0.90) │ → AUTO_CHALLAN (saved to DB, no human needed)
│  TIER 2 (conf≥0.60) │ → HUMAN_REVIEW (WhatsApp alert to officer)
│  TIER 3 (conf<0.60) │ → LOG_WITH_PLATE / DISCARD
└─────────────────────┘
    ↓
[Evidence Packager]  ml/utils/evidence.py — annotated JPEG + JSON record
    ↓
[FastAPI Backend]    backend/main.py — REST API + WebSocket + SQLite
    ↓
[Frontend Dashboard]
```

Two entry points run this pipeline today:
1. **`backend/api/jobs.py`** — uses `get_ml_registry()` from `backend/services/ml_registry.py` (shared singleton), used by `POST /jobs` and `POST /jobs/upload`. This is the path the dashboard's upload flow hits.
2. **`ml/demo_pipeline.py`** — standalone CLI for local testing/debugging, optionally POSTs results to the backend via `--backend-url`.

The patrol WebSocket (`backend/api/stream.py → ws_patrol`) also shares the same `MLRegistry` singleton — no separate model instances are loaded.

---

## Backend Directory Structure (post-refactor)

```
backend/
├── main.py                       FastAPI app — CORS, lifespan, router registration
├── core/
│   ├── config.py                 Settings (DATABASE_URL, SECRET_KEY, SMTP creds)
│   ├── database.py               ORM models, async engine, CRUD helpers
│   ├── auth_utils.py             JWT creation, password hashing, RBAC
│   ├── email_service.py          SMTP email dispatch
│   └── agent_executor.py         Gemma-2 local LLM ops-agent
├── models/
│   └── schemas.py                Pydantic request/response schemas
├── services/                     ← NEW business-logic layer
│   ├── ml_registry.py            Shared ML singleton (preprocessor, detector, OCR, classifier)
│   ├── calibration_service.py    Per-camera calibration resolver (stop_line_y, zones, direction)
│   └── challan_service.py        Violation packaging, tier routing, DB persistence
└── api/
    ├── cameras.py                CRUD for camera registry + calibration config
    ├── vehicles.py               Vehicle lookup + repeat-offender list
    ├── analytics.py              Summary, trends, heatmap endpoints
    ├── stream.py                 /ws/feed (dashboard) + /ws/patrol (mobile)
    ├── debug.py                  Inject test violation, pipeline status, ML registry health
    ├── jobs.py                   Job queue — upload, process, status, results
    ├── violations.py             Ingest + list violations
    ├── reviews.py                Officer review workflow
    ├── evidence.py               Evidence image serving
    ├── auth.py                   Login, token refresh, registration
    ├── users.py                  User management
    ├── audit_logs.py             Audit trail
    └── agent.py                  AI ops-agent chat endpoint
```

## ML Directory Structure (post-refactor)

```
ml/
├── pipeline/                     Inference modules (unchanged)
│   ├── preprocessor.py           CLAHE + denoise + gamma
│   ├── detector.py               YOLOv8m vehicle/person/phone detector
│   ├── tracker.py                ByteTrack per-camera state
│   ├── violation_classifier.py   9 violation types — all sub-classifiers here
│   ├── confidence_router.py      3-tier routing thresholds
│   ├── driver_state.py           MediaPipe FaceMesh drowsiness/phone
│   └── ocr.py                    2-stage plate detection + OCR engine fallback chain
├── models/
│   ├── helmet_cnn.py             CNN architecture definition
│   └── weights/
│       ├── detection/            Primary detector weights
│       │   └── yolov8m.pt
│       ├── violations/           Per-violation-type weights
│       │   ├── helmet_best.pt
│       │   ├── helmet_cnn.pt
│       │   ├── seatbelt_classifier.pt
│       │   └── traffic_lights_yolov8x.pt
│       ├── ocr/                  Plate detector weights
│       │   ├── plate_koushi.pt
│       │   ├── plate_yasir.pt
│       │   └── plate_yolov8_moin.pt
│       └── metrics/              Training artefacts (not loaded at runtime)
│           ├── helmet_metrics.json
│           ├── plate_metrics.json
│           └── *.png             Training curves
├── federated/                    Federated learning client/server
├── training/                     Training scripts + data prep
└── utils/
    ├── evidence.py               Evidence packaging helper
    └── visualizer.py             Frame annotation renderer
```

---

## ML Model Inventory

### `ml/models/weights/detection/`

| File | Role | Loaded by |
|------|------|-----------|
| `yolov8m.pt` | Primary vehicle/person/phone detector | `ml/pipeline/detector.py` |

### `ml/models/weights/violations/`

| File | Role | Verified metrics | Loaded by |
|------|------|-------------------|-----------|
| `helmet_best.pt` | **Primary** helmet check — AICity Track-5 9-class detector (helmet / head / person), runs on full image. Not used by triple-riding | mAP@0.5 = 0.648 | `AIHelmetViolationDetector` in `violation_classifier.py` |
| `helmet_cnn.pt` | Fallback helmet classifier — binary CNN on head crop | accuracy=0.8744, precision=0.8675, recall=0.8182, **f1=0.8421** (n=215) | `HelmetClassifier` in `violation_classifier.py` |
| `seatbelt_classifier.pt` | Windshield-ROI seatbelt classifier (YOLOv11s) | — | `ViolationClassifier._load_seatbelt_model` |
| `traffic_lights_yolov8x.pt` | Traffic signal state detector (DTLD+LISA+BSTLD+HDTLR) | — | `MLSignalStateDetector` in `violation_classifier.py` |

### `ml/models/weights/ocr/`

| File | Role | Verified metrics | Loaded by |
|------|------|-------------------|-----------|
| `plate_koushi.pt` | Plate detector Stage-1 (best spatial coverage) | mAP50=0.8816, mAP50-95=0.5102 | `ml/pipeline/ocr.py` |
| `plate_yasir.pt` | Plate detector Stage-2 (confirms/refines Stage-1 crop) | — | `ml/pipeline/ocr.py` |
| `plate_yolov8_moin.pt` | Legacy fallback plate detector if Stage-1 missing | — | `ml/pipeline/ocr.py` (fallback chain) |

### `ml/models/weights/metrics/`
Training-time artefacts only — **not loaded at runtime**. Includes `helmet_metrics.json`, `plate_metrics.json`, and training curve PNGs.

OCR **text recognition** (separate from plate *detection*) tries engines in order: `fast-plate-ocr` → `PaddleOCR` → `EasyOCR` → `Tesseract`. See `ml/pipeline/ocr.py`.

---

## API Base URL

```
http://localhost:8000/api/v1
```

Interactive docs: `http://localhost:8000/docs`

---

## Authentication

JWT-based, implemented in `backend/api/auth.py`. Email verification is mandatory — `POST /auth/register` requires SMTP env vars to be set (`SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM_EMAIL`) or registration fails with 400. Endpoints requiring `Depends(get_current_user)` (most of `users`, `audit_logs`, `agent`) need a Bearer token.

| Method | Endpoint | Description |
|--------|----------|--------------|
| POST | `/auth/register` | Create account, sends verification email |
| GET | `/auth/verify` | Email verification link target (returns HTML) |
| POST | `/auth/login` | Returns JWT |
| GET | `/auth/me` | Current user info |

---

## REST Endpoints

### Violations (`backend/api/violations.py`, no extra prefix — paths below are literal)

| Method | Endpoint | Description |
|--------|----------|--------------|
| GET | `/violations` | List violations (paginated, filterable) |
| GET | `/violations/{id}` | Get single violation with full JSON |
| POST | `/violations/ingest` | Submit new violation from ML pipeline |
| POST | `/violations/{id}/confirm` | Officer confirms → auto-challan |
| POST | `/violations/{id}/reject` | Officer rejects → false positive |
| GET | `/violations/{id}/image` | Redirect to annotated evidence image |
| POST | `/violations/public-report` | Public-facing citizen report submission |

#### List Violations — Query Params
| Param | Type | Description |
|-------|------|--------------|
| `page` | int | Page number (default 1) |
| `page_size` | int | Items per page (max 100, default 20) |
| `tier` | int | Filter by tier 1/2/3 |
| `status` | str | `pending` / `auto_challan` / `confirmed` / `rejected` |
| `camera_id` | str | Filter by camera |
| `type` | str | Violation type string |
| `date_from` / `date_to` | str | ISO date `YYYY-MM-DD` |

#### Violation Status Values
| Status | Meaning |
|--------|---------|
| `pending` | Tier 2 — awaiting officer action |
| `auto_challan` | Tier 1 — auto-issued, no review needed |
| `confirmed` | Officer confirmed, challan issued |
| `rejected` | Officer rejected (false positive) |
| `discarded` | Tier 3, too low confidence |

---

### Cameras (`/cameras`)

| Method | Endpoint | Description |
|--------|----------|--------------|
| GET | `/cameras` | List all registered cameras |
| POST | `/cameras` | Register a new camera |
| GET | `/cameras/{id}` | Get single camera info |
| PUT | `/cameras/{id}/config` | Update stop line, description |
| DELETE | `/cameras/{id}` | Remove camera |

```json
// Register Camera — Body
{
  "id":          "BLR-CAM-MG-ROAD-001",
  "location":    "MG Road & Brigade Road Intersection",
  "lat":         12.9753,
  "lon":         77.6069,
  "stop_line_y": 380,
  "description": "4-lane junction, 30 km/h zone"
}
```

---

### Vehicles (`/vehicles`)

| Method | Endpoint | Description |
|--------|----------|--------------|
| GET | `/vehicles/{plate}` | Vehicle history by plate |
| GET | `/vehicles/repeat` | All repeat offenders |
| DELETE | `/vehicles/{plate}/clear` | Admin: reset vehicle record |

---

### Analytics (`/analytics`)

| Method | Endpoint | Description |
|--------|----------|--------------|
| GET | `/analytics/summary` | Today + week totals, type breakdown |
| GET | `/analytics/trends?days=30` | Daily violation counts over N days |
| GET | `/analytics/heatmap` | Per-camera counts with lat/lon for Leaflet |

This is the closest thing to ps.txt's "Analytics and Reporting" requirement — it covers statistics/trends but is not a model-evaluation report (see gap note above).

---

### Jobs (`/jobs`) — the real ML ingestion path

| Method | Endpoint | Description |
|--------|----------|--------------|
| GET | `/jobs` | List all processing jobs |
| POST | `/jobs` | Create a metadata-only job, runs a stub background pipeline |
| POST | `/jobs/upload` | **Upload image/video** — runs the actual ML pipeline (preprocess→detect→classify→OCR) as a background task, writes results to `violations` table |
| GET | `/jobs/{job_id}` | Job status/progress |
| GET | `/jobs/{job_id}/violations` | Violations produced by a specific job |

---

### Reviews (`/reviews`) — officer review audit trail

| Method | Endpoint | Description |
|--------|----------|--------------|
| GET | `/reviews` | List review actions (filtered audit log entries: approved/rejected/escalated) |
| POST | `/reviews` | Submit officer decision on a Tier-2 violation; updates violation status and writes an audit log row |

---

### Evidence (`/evidence`)

| Method | Endpoint | Description |
|--------|----------|--------------|
| GET | `/evidence/{id}` | Before/violation/after frame metadata for a violation (used by the evidence-timeline UI) |
| GET | `/evidence/test-gallery/list` | List sample images in `test/` for demo purposes |
| GET | `/evidence/test-gallery/image/{filename}` | Serve a sample image (path-traversal guarded) |

---

### Users (`/users`) — requires auth

| Method | Endpoint | Description |
|--------|----------|--------------|
| GET | `/users` | List all platform users |
| PUT | `/users/{user_id}/role` | Change a user's role (Admin only) |

---

### Audit Logs (`/audit-logs`) — requires auth

| Method | Endpoint | Description |
|--------|----------|--------------|
| GET | `/audit-logs` | Full compliance/audit trail, newest first |

---

### Gemma AI Agent (`/agent`) — requires auth

| Method | Endpoint | Description |
|--------|----------|--------------|
| POST | `/agent/chat` | Natural-language ops queries against the DB via a local `gemma3:1b` model (Ollama) + guarded SQL tool calls |
| GET | `/agent/status` | Check whether Ollama and `gemma3:1b` are running/loaded |

---

### Debug Endpoints (`/debug`)

| Method | Endpoint | Description |
|--------|----------|--------------|
| POST | `/debug/inject-violation` | Create fake violation (for UI testing only — not real ML output) |
| GET | `/debug/pipeline-status` | Shows which ML modules are installed |

```json
// Inject Test — Body
{
  "violation_type": "helmet_non_compliance",
  "confidence": 0.75,
  "tier": 2,
  "plate": "KA-01-AB-1234",
  "camera_id": "BLR-CAM-DEMO-001",
  "location": "MG Road"
}
```

---

## WebSocket — Live Feed

```
ws://localhost:8000/ws/feed
```

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/feed');
ws.onmessage = (evt) => {
  const data = JSON.parse(evt.data);
  if (data.event === 'violation_detected') {
    // data.violation_id, data.violation_type, data.confidence,
    // data.tier, data.plate, data.camera_id, data.location,
    // data.timestamp, data.severity, data.annotated_image_url
  }
};
```

### Event Types
| Event | When | Fields |
|-------|------|--------|
| `connected` | On connect | `message` |
| `violation_detected` | New violation | See above |
| `system_stats` | Every 10s | `fps`, `active_cameras`, `violations_today`, `tier1`, `tier2` |
| `pong` | Reply to `ping` | — |

Send `"ping"` (string) to keep connection alive.

There is a second WebSocket, `ws://localhost:8000/ws/patrol`, for mobile patrol units: it accepts base64-encoded frames, runs them through the ML pipeline in real time, and returns annotated results + saves evidence on detection.

There is a third WebSocket, `ws://localhost:8000/ws/video-render`, that renders a full annotated + demo MP4 from an uploaded video using the exact same per-frame pipeline as the batch job path (`_render_frame_full()` in `backend/api/stream.py`). It computes plate OCR for every freshly-confirmed violation, but until 2026-06-23 only ever used that result for the DB record — the rendered video itself never drew the plate text on screen. Fixed: it now draws `PLATE: <text> (<conf>%)` under the vehicle's box via `FrameVisualizer.draw_plate_result()` whenever OCR actually returns non-empty text. Note this doesn't guarantee text appears often — plate OCR's hit rate on small/distant/motion-blurred plates (e.g. a two-wheeler's rear plate from an overhead traffic-cam angle) is genuinely low; an empty `formatted_text` is the OCR engine failing to read the plate, not the drawing code being broken.

---

## Static Files — Evidence Images

```
GET /evidence/annotated/{violation_id}.jpg
GET /evidence/raw/{violation_id}_raw.jpg
```

Mounted via `StaticFiles` in `backend/main.py`. A second static mount, `/test-images`, serves the `test/` directory (sample images for demo/gallery use).

---

## ML Pipeline — Violation Types

All 9 types are defined in `ml/pipeline/violation_classifier.py:39-73`. The first 7 map directly to ps.txt's violation list; the last 2 (`phone_use_while_driving`, `drowsy_driving`) are GARUDA additions beyond the problem statement's minimum scope.

| Type | Fine (₹) | Severity | Detection method |
|------|----------|----------|-------------------|
| `helmet_non_compliance` | 1,000 | High | `check_helmet()` — AICity 9-class detector on full image (primary), CNN crop or heuristic fallback. Confidence is the model's real per-box score (`AIHelmetViolationDetector.detect()`) — previously hardcoded to 0.50, fixed 2026-06-21. Crop-based fallback's head region now sourced from the rider's person bbox instead of the motorcycle's bbox, and a `head` (no-helmet) read needs ≥0.40 confidence symmetric to a `helmet` (compliant) read needing the same — previously a low-confidence "helmet" read could silently override an actual bare head (both fixed 2026-06-23). **Residual limitation**: the trained CNN itself can still confidently misclassify a bare head as helmeted on very small/blurry/distant crops — a model-accuracy issue, not something fixable in this layer |
| `seatbelt_non_compliance` | 1,000 | Medium | `check_seatbelt()` — YOLOv11s on windshield ROI, Hough-line fallback. Skips vehicles whose bbox exceeds 35% of frame area (added 2026-06-21 — oversized/frame-filling boxes like a close-up truck have no real windshield in that crop and were producing overconfident false positives) |
| `triple_riding` | 2,000 | High | `check_triple_riding()` — **independent of the helmet model** (decoupled 2026-06-21). Uses person + motorbike detections straight from the main YOLOv8m detector, grouped via `associate_riders_with_vehicles()`: IoU + position heuristic ported from `temp/AI_Traffic_Violation_Detection_triple_riding_detection`'s `group_riders_with_vehicles()` — nearest-rider-first, capped at 4/vehicle, persons assigned to at most one vehicle per frame. Computed once per frame in `check_all()` and shared across all two-wheelers |
| `wrong_side_driving` | 5,000 | Critical | Tracker: `check_wrong_side()` — heading-angle threshold (`WRONG_SIDE_ANGLE_THRESHOLD_DEG = 100`) against the calibrated `traffic_direction`, plus a 2-consecutive-frame persistence counter (`WRONG_SIDE_STRIKES_REQUIRED`) before confirming (fixed 2026-06-23 — previously a raw velocity dot-product threshold, which a vehicle merely turning or changing lanes could trip on a single noisy frame). Buses are exempt (`WRONG_SIDE_EXEMPT_CLASSES`, added 2026-06-23) — a legitimate bus-bay merge can momentarily point well "backward" relative to through-traffic and was false-positiving. Image-only: `_check_wrong_side_static()` — position-only proxy (no heading in a still frame) against the per-camera `wrong_side_lane` setting ("left"/"right", default "left"); only fires at moderate conf (0.60), treat as Tier-2 review material, not auto-challan |
| `stop_line_violation` | 500 | Medium | Tracker: `check_stop_line()` over N-frame history, now gated on `signal_conf >= 0.60` (fixed 2026-06-23 — previously had **no** confidence floor at all, so a ~0.4-0.5-confidence signal misdetection, e.g. a stray red taillight, was enough to cite a vehicle with no real traffic light nearby). Image-only: `_check_stop_line_static()` — position vs. stop-line-y, same `signal_conf >= 0.60` floor |
| `red_light_violation` | 1,000 | High | Tracker: `check_red_light()` — now requires an actual crossing (vehicle was on the legal side earlier in the same window, not just present past the line at some point), gated on `signal_conf >= 0.60` (crossing requirement added 2026-06-23 — previously "anywhere past the line in the last N frames" wrongly classified an already-parked vehicle as having run the light once the signal turned red). Image-only: `_check_red_light_static()` — position + signal state, gated on `signal_conf >= 0.65` |
| `illegal_parking` | 500 | Low | Tracker: `check_illegal_parking()`, 300s stationary threshold + zone check, duration now anchored to a frame counter (`self._frame_counter`) scaled by `self.fps` instead of `time.monotonic()` (fixed 2026-06-23 — wall-clock time has nothing to do with video time for batch/offline jobs, where frames can process faster or slower than real time). Image-only: `_check_illegal_parking_static()` zone-only (no timer) |
| `phone_use_while_driving` | 5,000 | High | `check_phone()` — COCO `cell_phone` class overlapping driver region |
| `drowsy_driving` | 2,000 | Critical | `ml/pipeline/driver_state.py` — MediaPipe FaceMesh eye/yawn analysis |

**Note**: wrong-side, stop-line, red-light, and illegal-parking checks were originally tracker-only (video). Image-only static fallbacks were added to `check_all()` so single-frame uploads through `/jobs/upload` still produce all 7 ps.txt-required violation types, not just helmet/seatbelt/triple-riding/phone.

**Known limitation (not yet fixed)**: ByteTrack's track-ID continuity is unreliable in heavily congested/occluded scenes — observed both as a single bad frame resetting `check_illegal_parking()`'s accumulated timer (an ID briefly reassigned to an unrelated vehicle) and as a sign flip in `check_wrong_side()`'s velocity for the same physical vehicle across an ID switch. Tight, carefully-bounded zone calibration reduces the blast radius but doesn't eliminate it.

**Fixed 2026-06-23**: `wrong_side_zone`/`parking_zones` are still axis-aligned rectangles only, and from an overhead camera at an oblique angle, two genuinely different lanes/paths (e.g. a bus's legitimate bus-bay merge vs. an adjacent scooter-only service lane) can sweep through the *same* rectangular screen region at different times — confirmed on a real clip where a `wrong_side_zone` drawn around the scooter lane also caught the bus's normal merge as a false "Wrong Way". Rather than chase a polygon-zone rewrite, `check_wrong_side()` now exempts buses outright (`WRONG_SIDE_EXEMPT_CLASSES`) — a bus pulling into/out of a marked stop routinely points well "backward" relative to through-traffic for several frames as a normal, legitimate maneuver, not a lane-discipline violation, for any camera angle, not just this one. Re-verified on the same clip: the bus no longer false-positives and the real wrong-way scooter rider is still correctly flagged.

---

## Confidence Router

```
TIER 1 (conf ≥ 0.90) → AUTO_CHALLAN
  • Evidence saved, challan auto-generated
  • No human intervention needed

TIER 2 (0.60 ≤ conf < 0.90) → HUMAN_REVIEW
  • WhatsApp + SMS alert to nearest officer
  • Officer has 10 min to CONFIRM or reject (FP)
  • Officer response → federated learning training data

TIER 3 (conf < 0.60) → LOG_WITH_PLATE / DISCARD
  • Stored for audit trail only
  • Cross-reference repeat offender DB

OVERRIDE → Repeat offender always escalates to TIER 2 (HIGH priority)
```

---

## Federated Learning

No raw video leaves the camera. Only **model weight deltas** are sent to the central server:

```bash
# On edge node (camera server):
python -m ml.federated.client --server-address central:8080 --camera-id BLR-CAM-MG-001

# On central server (run weekly):
python -m ml.federated.server --port 8080 --rounds 3 --min-cameras 3

# Local simulation (no hardware needed):
python -c "from ml.federated.server import simulate_training; simulate_training(5, 3)"
```

This is wired but does not yet retrain from real edge data — treat as a framework/demo, not a production loop.

---

## Running the Full Demo

```bash
# Test image (shows step-by-step logs)
python ml/demo_pipeline.py --input sample.jpg --verbose

# Test video
python ml/demo_pipeline.py --input traffic.mp4 --video

# Webcam + driver state + show window
python ml/demo_pipeline.py --webcam --driver-state --show

# Export YOLOv8m to TensorRT for Jetson
python -c "
from ml.pipeline.detector import VehicleDetector
d = VehicleDetector()
d.export_tensorrt(half=True)
"
```

---

## Database Schema (`backend/core/database.py`)

```sql
violations (
  id TEXT PRIMARY KEY,          -- VIO-BLR-YYYYMMDD-HHmmss-XXXXXX
  camera_id TEXT,                -- indexed
  location TEXT,
  timestamp TEXT,                -- indexed, ISO UTC
  violation_type TEXT,           -- indexed
  confidence REAL,
  severity TEXT,                 -- critical/high/medium/low
  tier INTEGER,                  -- 1/2/3
  action TEXT,                   -- AUTO_CHALLAN/HUMAN_REVIEW/...
  fine_amount INTEGER,
  plate_text TEXT,                -- indexed
  plate_conf REAL,
  vehicle_class TEXT,
  annotated_img TEXT,
  raw_img TEXT,
  json_record TEXT,               -- full evidence JSON
  status TEXT,                    -- indexed: pending/auto_challan/confirmed/rejected
  officer_id TEXT,
  created_at TEXT
)

cameras (
  id TEXT PRIMARY KEY, location TEXT, lat REAL, lon REAL,
  stop_line_y INTEGER DEFAULT 380, status TEXT DEFAULT 'active',
  last_seen TEXT, description TEXT
)

vehicles (
  plate TEXT PRIMARY KEY, violation_count INTEGER, is_repeat_offender BOOLEAN,
  first_seen TEXT, last_seen TEXT, violations_json TEXT, state_code TEXT
)

users (
  id TEXT PRIMARY KEY, name TEXT, role TEXT DEFAULT 'Operator', email TEXT,
  status TEXT DEFAULT 'Active', last_login TEXT, password_hash TEXT,
  is_verified BOOLEAN DEFAULT 0, verification_token TEXT  -- indexed
)

processing_jobs (
  id TEXT PRIMARY KEY, name TEXT, source_type TEXT DEFAULT 'Video',
  progress INTEGER DEFAULT 0, status TEXT DEFAULT 'Queued', duration INTEGER,
  frames_processed INTEGER, violations_found INTEGER, upload_time TEXT
)

audit_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, actor TEXT,
  action TEXT, target TEXT, details TEXT
)
```

---

## ps.txt Coverage Matrix

| ps.txt requirement | Status | Where |
|---------------------|--------|-------|
| Image preprocessing (low light, rain, shadows, blur) | ✅ Done | `ml/pipeline/preprocessor.py` — CLAHE, denoise, gamma |
| Vehicle/road-user detection + classification | ✅ Done | `ml/pipeline/detector.py` (YOLOv8m) |
| Helmet non-compliance | ✅ Done | `check_helmet()` |
| Seatbelt non-compliance | ✅ Done | `check_seatbelt()` |
| Triple riding | ✅ Done | `check_triple_riding()` |
| Wrong-side driving | ✅ Done (tracker + static fallback) | `check_wrong_side()` / `_check_wrong_side_static()` |
| Stop-line violation | ✅ Done (tracker + static fallback) | `check_stop_line()` / `_check_stop_line_static()` |
| Red-light violation | ✅ Done (tracker + static fallback) | `check_red_light()` / `_check_red_light_static()` |
| Illegal parking | ✅ Done (tracker + static fallback) | `check_illegal_parking()` / `_check_illegal_parking_static()` |
| Violation classification + confidence scores | ✅ Done | `ConfidenceRouter` 3-tier system |
| License plate detection + OCR | ✅ Done | 2-stage YOLO (Koushi+YasirFaiz) + OCR engine chain |
| Evidence generation (annotated images + metadata) | ✅ Done | `ml/utils/evidence.py`, `ml/utils/visualizer.py` |
| Analytics and reporting | ✅ Done (stats/trends/heatmap) | `/analytics/*` endpoints |
| **Performance evaluation (Accuracy/Precision/Recall/F1/mAP)** | ⚠️ **Partial** — per-model training metrics exist (`helmet_metrics.json`, `plate_metrics.json`) but there is **no end-to-end evaluation script** that scores the full pipeline (detection→classification→OCR) against a labeled test set | Not yet built |

---

## Frontend Migration Guide

The vanilla frontend (`frontend/`) is intentionally simple. To migrate to React/Vue:

1. **API calls**: Everything is in `frontend/js/api.js`. Import as-is or rewrite as `fetch`/`axios` calls.
2. **WebSocket**: Copy `GarudaWS` from `frontend/js/websocket.js` or use the same pattern with `useEffect`.
3. **Charts**: Replace `GarudaCharts` with Recharts/Victory — same data shapes.
4. **Map**: Replace `GarudaMap` with `react-leaflet` — same `API.getHeatmapData()` call.
5. **Base URL**: Change `BASE_URL` in `api.js` or set an env var.

---

## Environment Variables (Key Ones)

| Variable | Default | Description |
|----------|---------|--------------|
| `DATABASE_URL` | `sqlite+aiosqlite:///./garuda.db` | Switch to Postgres for prod |
| `DEVICE` | `cpu` | `cuda:0` for GPU |
| `STOP_LINE_Y` | `380` | Pixel Y of stop line (calibrate!) |
| `CONFIDENCE_TIER1` | `0.90` | Auto-challan threshold |
| `CONFIDENCE_TIER2` | `0.60` | Human review threshold |
| `ALERTS_ENABLED` | `false` | Set `true` + Twilio creds for real SMS |
| `FL_ENABLED` | `false` | Enable federated learning client |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASSWORD` / `SMTP_FROM_EMAIL` | — | Required for `/auth/register` email verification to work at all |

---

## Contact

- **ML & Backend**: You (the ML engineer)
- **Frontend & UX**: Your friend
- This file: `BACKEND_REFERENCE.md` — keep updated as the system evolves.

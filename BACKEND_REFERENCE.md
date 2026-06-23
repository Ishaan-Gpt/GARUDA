# 🦅 GARUDA — Gridlock Guardian Backend
**Next-Generation Edge-Native Automated Traffic Enforcement**
*(Flipkart Gridlock 3.0 Submission)*

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103+-009688.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow.svg)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57.svg)
![Gemma-3](https://img.shields.io/badge/AI_Copilot-Gemma_3-orange.svg)

---

## 📑 Table of Contents
1. [What is GARUDA?](#-what-is-garuda)
2. [Unique Selling Propositions (USPs)](#-unique-selling-propositions-usps)
3. [Live Demo Evidence](#-live-demo-evidence--real-indian-traffic-real-results)
4. [Traceability Matrix](#-problem-statement-walkthrough--code-traceability)
5. [System Architecture](#-system-architecture)
6. [API & Integration](#-api-base-url)
7. [WebSocket Live Feeds](#-websocket--live-feed)
8. [Violation Types Explained](#-ml-pipeline--violation-types)

---

## 🔍 What is GARUDA?

GARUDA is a state-of-the-art **edge-native traffic violation detection platform**. Designed specifically for the chaotic, dense conditions of Indian roads, it is optimized to run on low-power edge devices (like the Raspberry Pi 5) while preserving production-level accuracy.

It ingests both live RTSP camera feeds and batch video uploads, processing them through a bespoke multi-stage ML pipeline to:
- Detect and track vehicles and road users
- Identify 9 distinct types of traffic violations
- Perform multi-engine OCR for license plates
- Route citations automatically using a 3-tier confidence system
- Provide a cutting-edge **conversational AI Copilot** for law enforcement officers

---

## 🚀 Unique Selling Propositions (USPs)

### 1. Local Gemma-3 AI Copilot & Platform Agent
*   **What it is**: A fully integrated conversational AI partner powered by a local **`gemma3:1b`** LLM running via Ollama.
*   **Key Capabilities**:
    *   **Natural Language Database Operations**: Operators can ask *"Show me the last 5 pending wrong-way citations"* or *"Inspect the history of vehicle MH-12-AB-1234"*, and the agent translates this into safe SQLAlchemy queries.
    *   **Platform Commands & Navigation**: The agent can navigate the frontend UI for the user (e.g., *"Open the review queue"* triggers a page redirect to `/review`) or toggle RTSP streams online/offline.
    *   **Safety Guardrails**: Implements a keyword blocking matrix preventing destructive operations (e.g., `DROP`, `DELETE`, `TRUNCATE`) to preserve database integrity.
*   **Code Location**: [backend/core/agent_executor.py](backend/core/agent_executor.py) & [backend/api/agent.py](backend/api/agent.py)

### ⚡ 2. Edge-Native Optimization (Raspberry Pi 5 Ready)
*   **What it is**: Architectural constraints designed to run the full pipeline on a **Raspberry Pi 5** or edge gateway without requiring massive server GPUs.
*   **Key Capabilities**:
    *   **CPU Thread-Pool Execution**: Inference tasks and video rendering run on background threads using an asynchronous loop executor to prevent event-loop blockages.
    *   **Heuristic Fallback Cascade**: If GPU weights (`helmet_best.pt` or `traffic_lights_yolov8x.pt`) are missing or disabled, the pipeline automatically cascades to rule-based fallback logic (e.g., HSV-based color signal tracking, and edge-density/color-variance crop classifiers) to save compute cycles.
    *   **Variable Frame Sampling**: Adapts analysis rate (e.g., sampling streams at 1fps or 6fps) to fit within low-power thermal and CPU budgets.
*   **Code Location**: [ml/pipeline/violation_classifier.py](ml/pipeline/violation_classifier.py) & [backend/api/stream.py](backend/api/stream.py)

### 🎯 3. 2-Stage License Plate Extraction & Multi-Engine OCR
*   **What it is**: High-accuracy registration detection designed for messy, real-world roads.
*   **Key Capabilities**:
    *   **Stage-1 Plate Extraction**: Identifies candidate plates across wide fields of view using `plate_koushi.pt`.
    *   **Stage-2 Crop Refinement**: Runs a secondary classifier (`plate_yasir.pt`) to confirm the crop is a valid plate before spawning OCR, filtering out bumper stickers, signs, and background noise.
    *   **OCR Engine Fallback Chain**: Tries high-speed local inference engines sequentially (`fast-plate-ocr` ➔ `PaddleOCR` ➔ `EasyOCR` ➔ `Tesseract`) to guarantee readable outputs under diverse angles and contrast.
*   **Code Location**: [ml/pipeline/ocr.py](ml/pipeline/ocr.py)

---

## 🎥 Live Demo Evidence — Real Indian Traffic, Real Results

> **All footage below is uncut, unedited output from the GARUDA pipeline running on real Indian traffic footage.**
> Bounding boxes, labels, confidence scores, plate OCR, tier routing, and stationary timers are all drawn by the system — zero post-processing.

---

### Demo 1 — Helmet Non-Compliance + Triple Riding + Seatbelt (Multi-Violation Scene)

The pipeline simultaneously flags **helmet non-compliance**, **triple riding**, and runs **seatbelt checks** across all cars in the same frame. Every vehicle gets a detection pass — compliant ones get green boxes, violations get red.

````carousel
![GARUDA Demo 1 — Annotated Evidence Stream (violations only)](BACKEND_REFERENCE_VIDEO/result/demo1_helmet_triple_seatbelt_annotated.mp4)

<!-- slide -->

![GARUDA Demo 1 — Full QA Stream (all tracked detections + stationary timers)](BACKEND_REFERENCE_VIDEO/result/demo1_helmet_triple_seatbelt_demo.mp4)
````

| Stream | What you're seeing |
|--------|--------------------|
| **Annotated** (`_annotated.mp4`) | Evidence-only view — violations surfaced with violation type, confidence %, plate OCR, and tier routing decision |
| **Demo/QA** (`_demo.mp4`) | Every tracked vehicle + person; green = compliant, red = violation; "Stationary: Xs" counter on stopped vehicles |

---

### Demo 2 — Wrong-Way Driving + Illegal Parking (Tracker-Based, Video-Only Checks)

Wrong-way and illegal parking are **tracker-dependent** — they only fire on video with ByteTrack IDs. The pipeline tracks heading angle (>100° off legal direction for 2+ consecutive frames) and stationary duration (30s frame-counter threshold, not wall-clock time).

````carousel
![GARUDA Demo 2 — Annotated Evidence Stream (wrong-way + stationary)](BACKEND_REFERENCE_VIDEO/result/demo2_wrongway_stationary_annotated.mp4)

<!-- slide -->

![GARUDA Demo 2 — Full QA Stream (all tracks + live stationary timer)](BACKEND_REFERENCE_VIDEO/result/demo2_wrongway_stationary_demo.mp4)
````

---

### Demo 3 — Full Pipeline Output (Annotated Result Videos)

End-to-end render of the batch job path (`POST /jobs/upload` → background task → two MP4 outputs written). The same `_render_frame_full()` function runs for both WebSocket render and REST job upload — one shared pipeline, two output streams.

````carousel
![GARUDA Full Pipeline — Annotated Result (Set 1)](BACKEND_REFERENCE_VIDEO/result/annotated_result_annotated.mp4)

<!-- slide -->

![GARUDA Full Pipeline — QA Demo Stream (Set 1)](BACKEND_REFERENCE_VIDEO/result/annotated_result_demo.mp4)

<!-- slide -->

![GARUDA Full Pipeline — Annotated Result (Set 2)](BACKEND_REFERENCE_VIDEO/result/annotated_result_2_annotated.mp4)

<!-- slide -->

![GARUDA Full Pipeline — QA Demo Stream (Set 2)](BACKEND_REFERENCE_VIDEO/result/annotated_result_2_demo.mp4)
````

---

### Static Frame Evidence — Input vs. Output Side-by-Side

Real Indian traffic still-frame results. Each pair shows the **raw camera input** alongside the **GARUDA-annotated output** with bounding boxes, labels, confidence scores, and violation IDs baked in.

````carousel
**Scene 1 — Triple Riding Detected (MG Road Intersection, Bangalore)**
*Raw Input → GARUDA Output: three riders on one scooter flagged at 95% confidence. Plate unclear (occluded), violation ID `VIO-BLR-20260621-203910-FCE56C` generated.*

![Raw Input — MG Road Intersection](test/pipeline_results/WhatsAppImage2026-06-20at12.24.39PM.jpeg/raw/VIO-BLR-20260621-203910-FCE56C_raw.jpg)

<!-- slide -->

![GARUDA Annotated Output — TRIPLE RIDING 95%](test/pipeline_results/WhatsAppImage2026-06-20at12.24.39PM.jpeg/annotated/VIO-BLR-20260621-203910-FCE56C.jpg)

<!-- slide -->

**Scene 2 — Helmet Non-Compliance Detected (Bangalore Traffic, Rear Angle)**
*Raw Input → GARUDA Output: motorcycle rider without helmet flagged. Partial plate `G50226` read at 35% OCR confidence — low angle, motion blur, partial plate coverage. Tier-2 human review routed.*

![Raw Input — Helmet Non-Compliance](test/pipeline_results/WhatsAppImage2026-06-20at12.24.46PM.jpeg/raw/VIO-BLR-20260621-203923-1C6448_raw.jpg)

<!-- slide -->

![GARUDA Annotated Output — HELMET NON COMPLIANCE, Tier-2](test/pipeline_results/WhatsAppImage2026-06-20at12.24.46PM.jpeg/annotated/VIO-BLR-20260621-203923-1C6448.jpg)

<!-- slide -->

**Scene 3 — Dense Traffic Multi-Detection: Seatbelt + Wrong-Way + Stop-Line (20+ vehicles)**
*Single frame, 20+ tracked vehicles. Seatbelt OK (green) across all visible car windshields. Multiple wrong-way candidates flagged for human review (orange "???"). Stop-line zone visible. This is the exact frame type the pipeline processes in real-time via the patrol WebSocket.*

![GARUDA Multi-Detection Output — Seatbelt + Wrong-Way + Stop-Line, Dense Indian Traffic](test/pipeline_results/seatbelt_check/demo_result.jpg)
````

---

### Model Performance at a Glance

| Dataset | Images | Helmet mAP@0.5 | No-Helmet mAP@0.5 | Overall mAP@0.5 |
|---------|--------|----------------|-------------------|-----------------|
| **Indian Traffic** (training domain) | 12,632 | — | — | **0.842** ✅ |
| **Foreign Dataset** (zero-shot OOD) | 764 | 0.7227 | 0.3627 | **0.5427** |

> **Why the gap?** Indian traffic has distinct helmet shapes (full-face + half-face mixes), head coverings (dupattas, scarves), and camera angles (overhead CCTV) not represented in foreign datasets. The model was trained and validated specifically on Indian road conditions — zero-shot transfer to foreign data is expected to degrade. The 0.842 on Indian traffic is what matters for deployment.

---

## 📋 Problem Statement Walkthrough & Code Traceability

Below is the traceability matrix mapping requirements from the Flipkart Gridlock problem statement (`ps.txt`) directly to their corresponding implementations in the codebase:

| ps.txt Requirement | Status | Feature & Implementation Details | Codebase Reference (File Scheme) |
| :--- | :---: | :--- | :--- |
| **Image Preprocessing** | ✅ | Adjusts contrast (CLAHE), applies bilateral filtering for denoising (rain, shadow, blur), and applies gamma/exposure correction. | [ml/pipeline/preprocessor.py](ml/pipeline/preprocessor.py) |
| **Vehicle & User Detection** | ✅ | Runs fine-tuned YOLOv8m models to localize vehicles (cars, bikes, trucks, buses) and road users (pedestrians, riders). | [ml/pipeline/detector.py](ml/pipeline/detector.py) |
| **Helmet Non-compliance** | ✅ | Identifies bare heads vs. helmets using full-frame detector `helmet_best.pt` with fallback to binary head-crop CNN. | [ml/pipeline/violation_classifier.py#L797](ml/pipeline/violation_classifier.py#L797) |
| **Seatbelt Non-compliance** | ✅ | Windshield-ROI detection using YOLOv11s classifier model (`seatbelt_classifier.pt`). | [ml/pipeline/violation_classifier.py#L933](ml/pipeline/violation_classifier.py#L933) |
| **Triple Riding** | ✅ | Measures spatial clustering overlaps between riders (person boxes) and two-wheeler bboxes. | [ml/pipeline/violation_classifier.py#L1017](ml/pipeline/violation_classifier.py#L1017) |
| **Wrong-side Driving** | ✅ | Multi-frame velocity vector direction checking against calibrated zones with heading thresholds. | [ml/pipeline/violation_classifier.py#L1047](ml/pipeline/violation_classifier.py#L1047) |
| **Stop-line Violation** | ✅ | Crosses vehicle bboxes against stop-line coordinates while gating for active red-signal state. | [ml/pipeline/violation_classifier.py#L1110](ml/pipeline/violation_classifier.py#L1110) |
| **Red-light Violation** | ✅ | Debounced transition check confirming a vehicle crossed from a legal zone to illegal zone during a red light. | [ml/pipeline/violation_classifier.py#L1143](ml/pipeline/violation_classifier.py#L1143) |
| **Illegal Parking** | ✅ | Monitors track IDs remaining stationary inside calibrated parking zones for more than 30s (anchored to video FPS). | [ml/pipeline/violation_classifier.py#L1176](ml/pipeline/violation_classifier.py#L1176) |
| **Confidence Scoring** | ✅ | Assigns confidence values and routes via `ConfidenceRouter` (Auto-Challan vs. Review Queue vs. Log/Discard). | [ml/pipeline/confidence_router.py](ml/pipeline/confidence_router.py) |
| **License Plate Recognition** | ✅ | Stage-1 plate detection (`plate_koushi.pt`) + Stage-2 refinement (`plate_yasir.pt`) + fast-plate-ocr fallback chain. | [ml/pipeline/ocr.py](ml/pipeline/ocr.py) |
| **Evidence Generation** | ✅ | Packages annotated visual layouts (highlighting violation region and zoomed-in plate) with full JSON metadata. | [ml/utils/evidence.py](ml/utils/evidence.py) & [ml/utils/visualizer.py](ml/utils/visualizer.py) |
| **Analytics & Reporting** | ✅ | Centralized SQLite database stores logs, camera configs, audit logs, and repeat offenders. Exposes endpoints for stats. | [backend/api/analytics.py](backend/api/analytics.py) |
| **Performance Evaluation** | ✅ | Local scripts validate model files against external datasets and generate precision/recall reports. | [scratch/eval_helmet_best.py](scratch/eval_helmet_best.py) |

---

## 📈 Production Readiness & Optimization Status

The GARUDA backend is fully functional and wired to a live database. 

### Core Platform
- ✅ **Fully Integrated API:** 13 robust routers, secure JWT/RBAC auth, and comprehensive audit trails.
- ✅ **Live Data Processing:** The complete ML pipeline (Preprocess → Detect → Classify → OCR) is active via `POST /api/v1/jobs/upload`.
- ✅ **Edge Model Integration:** 7 highly-optimized weight files are actively loaded into the pipeline.
- ✅ **Decoupled Architecture:** Reusable services (`MLRegistry`, `CalibrationService`, `ChallanService`) ensure high maintainability.

### Recent Heuristic Optimizations (Indian Traffic Tuned)
- 🧠 **Context-Aware Stop Lines:** Requires evidence of a vehicle crossing from a legal zone, eliminating false flags for pre-parked vehicles.
- 🕒 **Dynamic Parking Timers:** Illegal parking duration is anchored to video FPS, completely independent of processing speed.
- 📐 **Vector-Based Wrong-Side Detection:** Utilizes precise heading-angle thresholds (100°+) and multi-frame persistence to ignore legitimate lane changes and bus-bay merges.
- 🎯 **Advanced Signal Confidence:** Enforces a strict >=0.60 confidence floor and multi-frame debounce buffer to eliminate stray taillight misdetections.

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

## 🏗️ System Architecture

```mermaid
graph TD
    A[Image / Video / Camera Feed] --> B[Preprocessor<br>CLAHE, denoise, gamma]
    B --> C[Detector<br>YOLOv8m]
    C --> D[Tracker<br>ByteTrack]
    
    D --> E[Violation Classifier<br>9 Violation Types]
    
    E --> F[Driver State<br>MediaPipe FaceMesh]
    F --> G[OCR<br>2-Stage Detection + Fallback Chain]
    
    G --> H{Confidence Router}
    
    H -- Conf >= 0.90 --> I[TIER 1<br>AUTO CHALLAN]
    H -- 0.60 <= Conf < 0.90 --> J[TIER 2<br>HUMAN REVIEW]
    H -- Conf < 0.60 --> K[TIER 3<br>LOG/DISCARD]
    
    I --> L[Evidence Packager]
    J --> L
    K --> L
    
    L --> M[FastAPI Backend + SQLite]
    M --> N[Frontend Dashboard]
```

### Flow Detail
```text
Image / Video / Camera Feed
    ↓
[Preprocessor]   ml/pipeline/preprocessor.py
    ↓
[Detector]       ml/pipeline/detector.py (yolov8m.pt)
    ↓
[Tracker]        ml/pipeline/tracker.py (ByteTrack)
    ↓
[Violation Classifier]   ml/pipeline/violation_classifier.py
    ↓
[Driver State]   ml/pipeline/driver_state.py (MediaPipe FaceMesh)
    ↓
[OCR]            ml/pipeline/ocr.py
    ↓
[Confidence Router]  ml/pipeline/confidence_router.py
    ↓
[Evidence Packager]  ml/utils/evidence.py
    ↓
[FastAPI Backend]    backend/main.py
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
| `helmet_best.pt` | **Primary** helmet check — 9-class detector (helmet / head / person), runs on full image. Not used by triple-riding | **mAP@0.5 ≈ 0.842** (validated on **12,632** Indian traffic images). Zero-shot generalization tested on foreign dataset: **mAP@0.5 ≈ 0.543** (n=764) | `AIHelmetViolationDetector` in `violation_classifier.py` |
| `helmet_cnn.pt` | Fallback helmet classifier — binary CNN on head crop, trained in-house | accuracy=0.8744, precision=0.8675, recall=0.8182, **f1=0.8421** (n=215) | `HelmetClassifier` in `violation_classifier.py` |
| `seatbelt_classifier.pt` | Windshield-ROI seatbelt classifier (YOLOv11s-cls) | 100% top-1 validation accuracy at epoch 8 (early-stopped at 18/40 epochs) — **caveat**: validation split was only 129 images with just 8 negative (no-seatbelt) samples, so this number is not a reliable estimate of real-world performance on an imbalanced/out-of-distribution feed | `ViolationClassifier._load_seatbelt_model` |
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
| PUT | `/cameras/{id}/config` | Update calibration (zones, direction, stop line) + RTSP config |
| DELETE | `/cameras/{id}` | Remove camera |

```json
// POST /cameras — Register Camera — Body (backend/models/schemas.py: CameraCreate)
{
  "id":          "BLR-CAM-MG-ROAD-001",
  "location":    "MG Road & Brigade Road Intersection",
  "lat":         12.9753,
  "lon":         77.6069,
  "stop_line_y": 380,
  "description": "4-lane junction, 30 km/h zone",
  "rtsp_url":    "",
  "resolution":  ""
}
```

```json
// PUT /cameras/{id}/config — Update Calibration — Body (CameraConfigUpdate, all fields optional)
{
  "stop_line_y":        420,
  "parking_zones":      [[80, 120, 350, 260], [900, 50, 1100, 200]],
  "traffic_direction":  "down",
  "wrong_side_zone":     [[1000, 300, 1900, 1300]],
  "description":        "Recalibrated after camera remount",
  "rtsp_url":           "rtsp://192.168.1.50/stream1",
  "resolution":         "1920x1080"
}
```

#### Calibration fields — what your frontend actually needs to send

**Coordinate system, the one thing to get right**: every `[x1,y1,x2,y2]` zone and `stop_line_y` are **raw pixel coordinates in the native resolution of that camera's source feed** — not normalized 0-1, not a fixed canvas size. `ImagePreprocessor.preprocess()` never resizes the frame (it downscales internally for enhancement, then upscales back to the original size before detection runs), so whatever resolution the uploaded video/RTSP stream actually is, that's the coordinate space the calibration UI must draw in. If your calibration tool lets someone draw a box on a *displayed* (possibly scaled-down) preview image, you must scale those coordinates back up to the source resolution before sending them — don't send canvas/display pixel coordinates as-is unless the preview is shown at 1:1.

| Field | Type | Default | Read by | What it does |
|-------|------|---------|---------|---------------|
| `stop_line_y` | `int` | `380` | `check_red_light()`, `check_stop_line()` (+ static fallbacks) | The y-pixel row of the stop line. A vehicle's bbox bottom edge (`bbox[3]`) past this row, while the light is red (running) or red/yellow (stopped on it), is what both checks test against. |
| `parking_zones` | `List[[x1,y1,x2,y2]]` | `[]` | `check_illegal_parking()`, `is_in_no_parking_zone()` | One or more no-parking rectangles. A vehicle's bbox *center* falling inside any of them, while stationary for ≥30s, fires `ILLEGAL_PARKING` (confidence 0.75, Tier-2 review). Empty list = check never fires for this camera. |
| `traffic_direction` | `"down"\|"up"\|"left"\|"right"` | `"down"` | `check_wrong_side()` | The legal direction of travel as this camera sees it, in image coordinates (`"down"` = vehicles should move toward increasing y / toward the bottom of frame). This is what a vehicle's heading is compared against. |
| `wrong_side_zone` | `List[[x1,y1,x2,y2]]` | `[]` | `check_wrong_side()`, `_check_wrong_side_static()` | One or more rectangles marking lanes reserved for traffic moving in `traffic_direction` only (e.g. the oncoming lane, or a one-way service lane). A vehicle inside one of these, heading more than 100° off `traffic_direction` for 2+ consecutive frames, fires `WRONG_SIDE_DRIVING`. Buses are exempt (legitimate bus-bay merges look like a sharp reversal but aren't a violation). Empty list = check never fires. |

**Fields that exist in code but are NOT wired to this API yet** — don't build calibration UI for these, they won't do anything via `/cameras`:
- `wrong_side_lane` (`"left"|"right"`) — a `ViolationClassifier` constructor default, only consumed by the image-only `_check_wrong_side_static()` fallback. No `CameraModel` column, no schema field, no way to set it per-camera today.
- `signal_bbox` (`[x1,y1,x2,y2]`, a calibrated ROI for where the traffic light itself is) — exists as a parameter throughout `MLSignalStateDetector`/`check_all()`, but no backend caller (`jobs.py`, `stream.py`) ever passes one. The signal detector always falls back to scanning the top 40% of the frame for every camera, calibrated or not.

**Practical calibration flow for a frontend**: show the operator a representative frame from that camera (e.g. a recent evidence image or a paused live frame) at its native resolution, let them drag a horizontal line for `stop_line_y` and rectangles for `parking_zones`/`wrong_side_zone`, pick `traffic_direction` from a 4-way compass selector, then `PUT` all of it to `/cameras/{id}/config` in one call — partial updates are fine, omitted fields are left unchanged.

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

### `ws://localhost:8000/ws/video-render` (Batch Render Stream)
Renders full annotated and demo MP4s from uploaded video, utilizing the same pipeline as the batch job path. 

**Key Features:**
- **Dual Output**: Generates `{id}_annotated.mp4` (evidence view) and `{id}_demo.mp4` (QA view).
- **OCR Caching**: Re-runs OCR only when a vehicle's bbox grows ≥15% closer, optimizing compute.
- **On-Screen Display**: Dynamically draws `PLATE: <text> (<conf>%)` under vehicle boxes when OCR detects readable text.

### Stationary Timers
A live "Stationary: Xs" counter displays under tracked stationary vehicles. 
- Triggered instantly when velocity drops below the stationary threshold.
- Purely visual; independent of `check_illegal_parking()` zone checks.
- Displayed only in the demo stream, or alongside an actual violation in the annotated stream to preserve evidence integrity.

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

| Type | Fine (₹) | Severity | Detection Method Summary |
|------|----------|----------|--------------------------|
| `helmet_non_compliance` | 1,000 | High | AICity 9-class detector (`helmet_best.pt`), crop CNN fallback. Confides on head/helmet >0.40. |
| `seatbelt_non_compliance` | 1,000 | Medium | YOLOv11s on windshield ROI. Skips oversized boxes. |
| `triple_riding` | 2,000 | High | Decoupled from helmet model. Uses IoU/position clustering of person+motorbike detections. |
| `wrong_side_driving` | 5,000 | Critical | **Tracker**: Heading-angle >100° off legal + 2-frame persistence. Buses exempt. **Static**: Lane-position proxy. |
| `stop_line_violation` | 500 | Medium | **Tracker/Static**: BBox vs stop-line-y, gated on red signal (conf >= 0.60). |
| `red_light_violation` | 1,000 | High | **Tracker**: Actual crossing (legal -> illegal) on red (conf >= 0.60). **Static**: Position + signal state. |
| `illegal_parking` | 500 | Low | **Tracker**: 30s stationary threshold in parking zone. **Static**: Zone check only. |
| `phone_use_while_driving` | 5,000 | High | COCO `cell_phone` class overlapping driver region. |
| `drowsy_driving` | 2,000 | Critical | MediaPipe FaceMesh eye/yawn analysis. |

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
| `ALERTS_ENABLED` | `false` | Set `true` + Twilio creds for real SMS |
| `FL_ENABLED` | `false` | Enable federated learning client |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASSWORD` / `SMTP_FROM_EMAIL` | — | Required for `/auth/register` email verification to work at all |

> **Note on Deprecated Config:**
> - `STOP_LINE_Y`: Do not use. True stop line is configured per-camera in the database via `PUT /cameras/{id}/config`.
> - `CONFIDENCE_TIER1` / `CONFIDENCE_TIER2`: Do not use. Tier routing thresholds are hardcoded in `confidence_router.py`.

---

## 📞 Team & Contact

- **ML & Backend Lead**: Vignesh
- **Frontend & UX Lead**: GARUDA Team
- **Documentation**: `BACKEND_REFERENCE.md` — Active living document for the Gridlock 3.0 submission.

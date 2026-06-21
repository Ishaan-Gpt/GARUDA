# 🚦 GARUDA — "Gridlock Guardian"
### Automated Traffic Violation Detection — Flipkart Gridlock 3.0

An edge-native, ML-driven pipeline that detects traffic violations from camera images, reads license plates, routes uncertain cases to human officers, and surfaces everything on a live dashboard.

This README documents what's actually built and measured — not the pitch deck version. For the full ML architecture notes and an honest status checklist, see [`ml_plan.txt`](ml_plan.txt). For the backend API contract, see [`BACKEND_REFERENCE.md`](BACKEND_REFERENCE.md).

---

## Architecture

```
Camera image
    │
    ▼
Preprocessing (CLAHE, denoise, gamma correction)        ml/pipeline/preprocessor.py
    │
    ▼
Detection — YOLOv8m (vehicles, persons)                 ml/pipeline/detector.py
    │
    ▼
Tracking — ByteTrack (within-camera)                    ml/pipeline/tracker.py
    │
    ▼
Violation Classifier — 9 violation types                ml/pipeline/violation_classifier.py
  (helmet: YOLOv8n, traffic lights: ML 8-class model)
    │
    ▼
License Plate — YOLOv8m detector + OCR chain            ml/pipeline/ocr.py
    │
    ▼
Confidence Router — Tier 1/2/3 + repeat-offender rule   ml/pipeline/confidence_router.py
    │
    ▼
Evidence Packager — annotated JPEG + JSON record         ml/utils/evidence.py
    │
    ▼
FastAPI Backend (REST + WebSocket, async SQLAlchemy DB)  backend/
    │
    ▼
Next.js Dashboard (dashboard, violations, analytics,     src/app/
  cameras, review queue, patrol, search, settings, login)
```

`ml/demo_pipeline.py --backend-url http://localhost:8000` is what actually connects the ML side to the live backend — it POSTs real detections to `/api/v1/violations/ingest`, so the dashboard reflects genuine model output, not just `/debug/inject-violation` test data.

---

## ML — Models and Performance

Five model checkpoints are live in `ml/models/weights/` and auto-load in `demo_pipeline.py`. The pipeline degrades gracefully to rule-based fallbacks if any weight file is missing — it never crashes.

### 1. Vehicle & person detector

| | |
|---|---|
| **Model** | `yolov8m.pt` (auto-downloaded, ~52 MB) |
| **Architecture** | YOLOv8m — 43M parameters |
| **Training data** | COCO 2017 — 118,000 training images, 80 classes |
| **Classes used** | person, bicycle, car, motorcycle, bus, truck (6 of 80) |
| **Detection threshold** | conf ≥ 0.25, IoU ≤ 0.45 (NMS) |
| **Tracking** | ByteTrack (built-in Ultralytics) — persistent track IDs across frames |

### 2. Helmet violation detector (primary)  `helmet_violation.pt`

| | |
|---|---|
| **Architecture** | YOLOv8n — lightweight, optimised for rider-crop inference |
| **Classes** | `With Helmet`, `Without Helmet` (2 classes) |
| **mAP@0.5** | **0.881** |
| **Training data** | Large-scale traffic surveillance dataset with helmet/no-helmet annotations |
| **Input** | Cropped 2-wheeler bounding box (full vehicle crop, 640 px inference) |
| **Fallback** | MobileNetV3-Small CNN (see §3) when this file is missing |

### 3. Helmet compliance CNN (fallback)  `helmet_cnn.pt`

| | |
|---|---|
| **Architecture** | MobileNetV3-Small backbone (ImageNet-pretrained) + custom head — `ml/models/helmet_cnn.py` |
| **Parameters** | 1,075,234 (~1.1M — edge-deployable) |
| **Training data** | 764 traffic surveillance images, Pascal VOC bounding boxes around riders' heads, classes "With Helmet" / "Without Helmet". Multiple riders per image — actual crop count exceeds 764. |
| **Split** | 70/15/15 train/val/test (split by source image, no leakage) |
| **Test accuracy** | **87.44%** (n=215 held-out crops) |
| **Precision / Recall / F1** | 86.75% / 81.82% / 84.21% |
| **Confusion matrix** | TP=72, TN=116, FP=11, FN=16 |

Training curves: `ml/models/weights/helmet_training_report.png`. Raw metrics: `ml/models/weights/helmet_metrics.json`.

### 4. License plate detector — primary  `open-image-models` (YOLOv9, ONNX)

| | |
|---|---|
| **Architecture** | `yolo-v9-s-608-license-plate-end2end` from the `open-image-models` package — ONNX, auto-downloaded and cached on first use |
| **Why this and not a local `.pt`** | Benchmarked against `plate_yolov8_moin.pt` and `plate_yolo.pt` (below) on real evidence frames: this detector found plates on 7-9/11 vehicles per frame vs. 2/11 for the local YOLOv8m model, which only fired on motorcycles and missed every car. ONNX also avoids the arbitrary-code-execution risk of loading a pickled `.pt` from an unverified source. |
| **Role in the pipeline** | Localises the plate inside each vehicle crop *before* OCR runs — this is the actual fix for "OCR finds nothing": feeding a general OCR model a precisely-cropped plate instead of guessing a region of the vehicle bbox is what makes recognition work at all. |
| **Reading** | EasyOCR on the detector's tight crop, with a plate-charset allowlist, multi-line (top-to-bottom) joining for 2-row Indian plates, and template-based O/0, I/1, S/5, Z/2, G/6, B/8 correction before validating against the state-code regex. |

### 5. License plate detector — local weights (present but currently unused) `plate_yolov8_moin.pt` / `plate_yolo.pt`

Two locally-trained YOLO plate detectors exist in `ml/models/weights/` but are **not** wired into the running pipeline:

| | |
|---|---|
| `plate_yolov8_moin.pt` | YOLOv8m, 25.85M params, 52 MB. Documented as "primary" but was never actually loaded by `jobs.py` (which only checked for `plate_yolo.pt`), and underperforms `open-image-models` in direct comparison — see above. |
| `plate_yolo.pt` | YOLO11n, fine-tuned via `ml/training/train_plate_yolo.py`. 433 images, mAP@0.5 = **88.16%** on its own val/test split, precision/recall 84.35%/83.67%. Kept as the fallback path in `PlateOCR._load_plate_detector` if `open-image-models` is ever unavailable. |

PR curves, F1 curve, confusion matrix for `plate_yolo.pt`: `ml/models/weights/*.png`. Raw metrics: `ml/models/weights/plate_metrics.json`.

### 6. Traffic light state detector  `traffic_lights_yolov8x.pt`

| | |
|---|---|
| **Architecture** | YOLOv8, ~49.6 MB |
| **Training data** | Multi-dataset traffic light corpus covering diverse intersection types and lighting conditions |
| **Classes** | 8 signal states — `GreenCircular`, `GreenLeft`, `GreenRight`, `GreenStraight`, `RedCircular`, `RedLeft`, `RedRight`, `RedStraight` |
| **Integration** | Scans top 40% of frame (where signals appear); maps class names to `red`/`green`/`yellow` state; falls back to HSV colour detection if no detection above conf=0.35 |

### How the custom models were trained (reproducible)

The CNN fallback and plate-YOLO fallback were trained on free Google Colab GPU (T4). Full pipeline:

1. `ml/training/voc_utils.py` — parses Pascal-VOC exports
2. `ml/training/prepare_helmet_data.py` — crops head/helmet bounding boxes into 64×64 classification images
3. `ml/training/prepare_plate_data.py` — converts plate bounding boxes into YOLO label format
4. `ml/training/train_helmet.py` — PyTorch training loop, outputs accuracy/precision/recall/F1/confusion matrix
5. `ml/training/train_plate_yolo.py` — wraps `ultralytics` training + validation
6. `ml/training/GARUDA_Train_Colab.ipynb` — self-contained notebook bundling all of the above; upload to Colab, run top to bottom, download the resulting weights+metrics zip

### What's ML-based vs. rule-based

Not every violation needs a dedicated trained model — geometry/temporal logic on top of the YOLOv8m detector + ByteTrack is the correct design for several checks:

| Violation | Method |
|---|---|
| Helmet non-compliance | **YOLOv8n model** (`helmet_violation.pt`, mAP@0.5=0.881) → CNN fallback (`helmet_cnn.pt`, 87.44%) |
| License plate | **YOLOv9 detector** (`open-image-models`, ONNX) locates the plate in the vehicle crop → **EasyOCR** reads the tight crop (charset-restricted, multi-line join, O/0·I/1·S/5 template correction). Local `plate_yolo.pt`/`plate_yolov8_moin.pt` kept as fallback only — see "Known limitations" |
| Traffic light violation | **ML detector** (`traffic_lights_yolov8x.pt`, 8 signal classes) + HSV colour fallback |
| Phone use while driving | YOLOv8m COCO class 67 ("cell phone") — no extra training needed |
| Triple riding | AI helmet detector rider count + geometry fallback (person-boxes inside 2-wheeler box) |
| Wrong-side driving | Tracker velocity vector vs. expected traffic direction |
| Stop-line violation | Bbox position vs. calibrated stop-line `y` coordinate |
| Illegal parking | Stationary-duration timer in a calibrated no-parking zone |
| Seatbelt | Hough-line heuristic (diagonal belt line) — intentionally capped at low confidence, always sent to human review |
| Drowsy driving | MediaPipe FaceMesh, Eye Aspect Ratio < 0.25 sustained |

### Known limitations (stated plainly, not buried)

- **Edge export untested**: `detector.py` has real `export_tensorrt()`/`export_tflite()` calls, but no Jetson/Raspberry Pi hardware was available to actually run them — no `.engine`/`.tflite` file has ever been produced, and FPS numbers anywhere in `ml_plan.txt` predating this are unverified targets, not measurements.
- **Cross-camera vehicle re-identification** (matching the same vehicle across different camera feeds) is not implemented — no OSNet/torchreid code exists in this repo.
- **Federated learning** (`ml/federated/`) has real Flower/FedAvg wiring, but the local training step is currently a no-op stub — officer corrections are collected but not yet used to retrain.
- License plate **OCR accuracy** (character-level) hasn't been separately benchmarked — only plate *localization* has a measured mAP for `plate_yolo.pt`. On spot-checked real CCTV frames, EasyOCR recovers recognizable-but-imperfect text (e.g. `GJ15CL0986` read as `6J15CL092`) — digit/letter confusion on small, low-contrast plates remains the main error source even with a correctly localized crop. Provide a labeled plate-text test set to get a real number.
- `plate_yolov8_moin.pt` is committed but unused — it underperformed `open-image-models` head-to-head (see "ML — Models" above) and was never wired into `jobs.py` to begin with. Left in place in case it's useful for a different camera angle/dataset, but don't assume it's running.

---

## Tech Stack

```
ML:        YOLOv8m (Ultralytics), MobileNetV3-Small, ByteTrack, MediaPipe FaceMesh,
           open-image-models (YOLOv9 plate detector) + EasyOCR, OpenCV, PyTorch, Albumentations
Backend:   FastAPI (async), SQLAlchemy 2.0 (async), SQLite (dev) / PostgreSQL (prod),
           Pydantic v2, WebSocket, Twilio (mock mode)
Frontend:  Next.js 16 (App Router), React 19, TypeScript
Federated: Flower (flwr) — wiring only, training stub
```

---

## Running It

### Backend + ML

The license-plate detector (`open-image-models`) requires **Python 3.10+** — newer than
the rest of the stack strictly needs, so it lives in its own venv rather than forcing a
system-wide Python upgrade.

```bash
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install "open-image-models[onnx]" easyocr
cp .env.example .env
.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000      # http://localhost:8000/docs

# Run the ML pipeline on an image and push real results into the backend:
.venv\Scripts\python.exe ml/demo_pipeline.py --input sample.jpg --backend-url http://localhost:8000 --verbose

# Or with driver-state (drowsiness/phone) analysis:
.venv\Scripts\python.exe ml/demo_pipeline.py --input sample.jpg --driver-state --backend-url http://localhost:8000
```

### Frontend dashboard

```bash
npm install
npm run dev      # http://localhost:3000
```

### Re-train the ML models

Open `ml/training/GARUDA_Train_Colab.ipynb` in Google Colab (free GPU), run all cells, drop the resulting `helmet_cnn.pt` / `plate_yolo.pt` into `ml/models/weights/`.

---

## Project Structure

```
ml/            ML pipeline, trained models, training scripts          → see ml_plan.txt
backend/       FastAPI REST + WebSocket API, async DB                 → see BACKEND_REFERENCE.md
src/           Next.js dashboard (App Router)
frontend/      Legacy vanilla HTML/JS dashboard (superseded by src/)
ps.txt         Original hackathon problem statement (unmodified)
```

---

*Team CodeKrafters | Flipkart Gridlock 3.0*

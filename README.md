# 🚦 GARUDA — "Gridlock Guardian"
### Automated Traffic Violation Detection — Flipkart Gridlock 3.0

GARUDA is an edge-native, machine learning-driven traffic violation detection pipeline. It ingests camera feeds or uploaded images/videos, preprocesses them to handle diverse environmental conditions, detects vehicles/pedestrians, classifies nine distinct types of traffic violations, performs license plate localization and OCR, and routes events dynamically based on model confidence. Detections are instantly streamed via WebSockets to a live Next.js officer dashboard or escalated for manual human review.

This README details the system architecture, file organization, installation/setup steps, and instructions for running the frontend, backend, and machine learning pipeline.

---

## 🏗️ System Architecture

```
                       Camera Image / Video Feed
                                   │
                                   ▼
             [Preprocessor] (ml/pipeline/preprocessor.py)
                    (CLAHE, Denoising, Gamma Correction)
                                   │
                                   ▼
             [Detector] (ml/pipeline/detector.py)
                    (YOLOv8m: Vehicle/Person/Phone)
                                   │
                                   ▼
             [Tracker] (ml/pipeline/tracker.py)
                    (ByteTrack: Persistent Track IDs)
                                   │
                                   ▼
      ┌────────────────────────────────────────────────────────┐
      │       [Violation Classifier] (violation_classifier.py)  │
      ├────────────────────────────────────────────────────────┤
      │ • Helmet non-compliance (helmet_best.pt / cnn fallback) │
      │ • Triple riding (Person-vehicle spatial clustering)     │
      │ • Seatbelt non-compliance (seatbelt_classifier.pt)     │
      │ • Traffic light violation (traffic_lights_yolov8x.pt)  │
      │ • Wrong-side driving (Tracker velocity vector / static)│
      │ • Stop-line violation (Position vs stop-line-y + light)│
      │ • Red-light violation (Crossing detection + light)     │
      │ • Illegal parking (Stationary duration timer)          │
      │ • Phone use while driving (YOLOv8m Cell phone overlaps) │
      │ • Drowsy driving (MediaPipe FaceMesh Eye Aspect Ratio) │
      └────────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
             [Plate Detector] (ml/pipeline/ocr.py)
               (Stage 1: plate_koushi.pt candidate bbox)
               (Stage 2: plate_yasir.pt confirmation)
                                   │
                                   ▼
             [Plate Text Reader OCR] (ml/pipeline/ocr.py)
           (fast-plate-ocr / PaddleOCR / EasyOCR / Tesseract)
                                   │
                                   ▼
             [Confidence Router] (ml/pipeline/confidence_router.py)
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │ Tier 1 (conf ≥ 0.90)    │ Tier 2 (0.60 ≤ conf < 0.90)│ Tier 3 (conf < 0.60)
         ▼                         ▼                         ▼
    AUTO_CHALLAN              HUMAN_REVIEW             LOG_WITH_PLATE
  (Instant DB Save)      (Officer Queue / WhatsApp)       / DISCARD
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   ▼
             [Evidence Packager] (ml/utils/evidence.py)
                    (Annotated JPEGs + JSON evidence)
                                   │
                                   ▼
             [FastAPI Backend] (backend/)
                    (REST API, SQLite/PostgreSQL, WebSockets)
                                   │
                                   ▼
             [Next.js Dashboard] (src/app/)
                    (Live feed, analytics, review queue)
```

---

## 🛠️ Tech Stack

*   **Machine Learning**: YOLOv8m (Ultralytics), YOLOv11s, Custom MobileNetV3-Small (PyTorch), ByteTrack, MediaPipe FaceMesh, EasyOCR / PaddleOCR / fast-plate-ocr, OpenCV, Albumentations
*   **Backend**: FastAPI, SQLAlchemy 2.0 (async), SQLite (development) / PostgreSQL (production), Pydantic v2, WebSockets, Twilio SMS/WhatsApp (optional), Ollama (Gemma-2 local LLM ops agent)
*   **Frontend**: Next.js 16 (App Router), React 19, TypeScript, WebSockets

---

## 📦 Project Folder Structure

```
GARUDA/
├── backend/                       # FastAPI backend codebase
│   ├── api/                       # API route handlers (cameras, violations, stream, etc.)
│   ├── core/                      # Configuration, async DB setup, auth, LLM agent executor
│   ├── models/                    # Pydantic validation schemas
│   ├── services/                  # Business logic (ML registry, calibration, challans)
│   └── main.py                    # Application entry point & router registration
├── ml/                            # Machine Learning pipeline & models
│   ├── federated/                 # Federated learning client/server stubs (Flower)
│   ├── models/                    # Architecture definitions & model weights
│   │   └── weights/
│   │       ├── detection/         # Vehicle/person/phone detector weight (yolov8m.pt)
│   │       ├── violations/        # Violation classifiers (helmet, seatbelt, traffic lights)
│   │       ├── ocr/               # License plate detectors (koushi, yasir, fallback moin)
│   │       └── metrics/           # Model training metrics, json reports, and curves
│   ├── pipeline/                  # Modular inference stages (detector, tracker, OCR, etc.)
│   ├── training/                  # Custom dataset preparation and training scripts
│   └── utils/                     # Annotation rendering and evidence packager
├── src/                           # Next.js frontend application (React 19)
├── public/                        # Static assets for the frontend
├── test/                          # Test suite and demo images/videos
└── requirements.txt               # Python package dependencies list
```

---

## 🚀 Installation & Setup

GARUDA consists of a Python FastAPI backend (which also runs the ML pipeline) and a Next.js React frontend dashboard.

### System Requirements
*   **OS**: Windows, macOS, or Linux
*   **Python**: Version `3.10` or `3.11` (Python 3.10+ is strictly required by `open-image-models` for license plate detection)
*   **Node.js**: Version `18.0` or higher
*   **Hardware (Optional)**: NVIDIA GPU with CUDA installed for fast inference

---

### Step 1: Backend & ML Pipeline Setup

1.  **Clone the Repository** and navigate to the project directory:
    ```bash
    cd GARUDA
    ```

2.  **Create a Python Virtual Environment**:
    *   **Windows**:
        ```bash
        py -3.11 -m venv .venv
        .venv\Scripts\activate
        ```
    *   **Linux/macOS**:
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```

3.  **Install Required Dependencies**:
    *   **For CPU-only execution**:
        ```bash
        pip install -r requirements.txt
        ```
    *   **For GPU acceleration** (highly recommended for live camera streams):
        ```bash
        pip install -r requirements.txt torch torchvision --index-url https://download.pytorch.org/whl/cu121
        ```

4.  **Configure Environment Variables**:
    Copy the sample environment file and modify it to suit your environment:
    ```bash
    cp .env.example .env
    ```
    *Open the `.env` file and verify the database configuration, model settings, and execution device (`cpu` or `cuda:0`).*

---

### Step 2: Frontend Dashboard Setup

1.  **Install Node.js dependencies** from the root folder:
    ```bash
    npm install
    ```

---

## 🏃 Running the Application

### Quick Start (Windows)
Double-click the `run_all.bat` file in the root directory, or execute it from the terminal:
```bash
run_all.bat
```
*This launches both the FastAPI backend and Next.js frontend in separate, titled console windows.*

---

### Manual Start

#### 1. Start the FastAPI Backend
With your virtual environment active, start the ASGI server:
```bash
uvicorn backend.main:app --reload --port 8000
```
*   **API Base URL**: `http://localhost:8000/api/v1`
*   **Interactive Swagger Documentation**: `http://localhost:8000/docs`
*   **Database**: Auto-initializes a local SQLite database file `garuda.db` inside the root directory upon startup.

#### 2. Start the Frontend Dashboard
In a separate terminal window, start the Next.js development server:
```bash
npm run dev
```
*   **Dashboard URL**: `http://localhost:3000` (or `http://localhost:3001` if port 3000 is occupied)
*   **Real-time updates**: Automatically connects to the backend WebSocket server (`ws://localhost:8000/ws/feed`) to stream live violations and system telemetry.

---

### 3. Running the ML Pipeline (Data Ingestion)

To inject real traffic data and test the ML pipeline, you can run files or streams directly.

#### Run the ML pipeline on a sample image:
```bash
python ml/demo_pipeline.py --input test/sample.jpg --backend-url http://localhost:8000 --verbose
```

#### Run the ML pipeline on a sample video:
```bash
python ml/demo_pipeline.py --input test/traffic.mp4 --video --backend-url http://localhost:8000
```

#### Run with driver state analysis (windshield check + drowsiness) on webcam:
```bash
python ml/demo_pipeline.py --webcam --driver-state --show --backend-url http://localhost:8000
```

#### Upload via API (cURL):
```bash
curl -F "name=camera_ingest" -F "source_type=Image" -F "file=@test/sample.jpg" \
     http://localhost:8000/api/v1/jobs/upload
```

---

## 🧠 ML Model Inventory & Weight Map

The ML models must be placed in their respective subdirectories within `ml/models/weights/`. If weights are missing, the pipeline degrades gracefully to rule-based fallback logic without crashing.

| Model File Path | Description / Target | Loaded by | Performance Metrics |
|---|---|---|---|
| **`ml/models/weights/detection/yolov8m.pt`** | Primary vehicle, person, and phone detector | `ml/pipeline/detector.py` | COCO 2017 (Auto-downloaded) |
| **`ml/models/weights/violations/helmet_best.pt`** | **Primary** helmet compliance detector (AICity 9-class model running on full frame) | `violation_classifier.py` | mAP@0.5 ≈ 0.842 (**12,632** Indian traffic images); mAP@0.5 ≈ 0.543 (764 foreign images) |
| **`ml/models/weights/violations/helmet_cnn.pt`** | Fallback binary helmet classifier (Runs on cropped head regions if primary model is unavailable) | `violation_classifier.py` | accuracy=0.8744, f1=0.8421 (n=215) |
| **`ml/models/weights/violations/seatbelt_classifier.pt`** | Windshield-region seatbelt classifier (YOLOv11s) | `violation_classifier.py` | Built-in fallback available |
| **`ml/models/weights/violations/traffic_lights_yolov8x.pt`** | Traffic light signal state detector (8-class model) | `violation_classifier.py` | HSV color tracking fallback |
| **`ml/models/weights/ocr/plate_koushi.pt`** | License Plate Detection Stage-1 (spatial extraction) | `ml/pipeline/ocr.py` | mAP@0.5 = 0.8816, mAP@0.5:0.95 = 0.5102 |
| **`ml/models/weights/ocr/plate_yasir.pt`** | License Plate Verification Stage-2 (crop refinement) | `ml/pipeline/ocr.py` | Stage-2 plate verify |
| **`ml/models/weights/ocr/plate_yolov8_moin.pt`** | Legacy license plate detector fallback | `ml/pipeline/ocr.py` | Legacy YOLOv8m |

---

## 🔍 Known Limitations

*   **Edge hardware optimizations**: Although `detector.py` includes methods for exporting YOLO models to TensorRT (`export_tensorrt()`) and TFLite, edge builds (`.engine` / `.tflite`) are untested.
*   **Cross-camera vehicle Re-ID**: Tracking vehicles across different disjoint cameras is not supported (re-identification models like OSNet are not integrated).
*   **Federated Learning Status**: The Flower client/server architecture is scaffolded in `ml/federated/`, but model retraining using edge officer feedback is currently a stub.
*   **OCR text recognition**: EasyOCR text recognition accuracy is highly dependent on contrast, lighting, and camera angle. Template-based regex checks correct common digit/letter confusions (e.g. `O` vs `0`), but plate recognition errors can still occur on highly distorted crops.

---

## 📖 Further Reading
For a complete documentation of REST endpoints, schemas, WebSocket parameters, database migrations, and detailed API responses, refer directly to [`BACKEND_REFERENCE.md`](BACKEND_REFERENCE.md).

---
*Team CodeKrafters | Flipkart Gridlock 3.0*

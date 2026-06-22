# GARUDA — Models & Workflow Architecture

This document summarizes the standardized machine learning models and end-to-end pipeline workflow deployed in the GARUDA Traffic Ingestion Network.

---

## 📊 Active Models Registry

| Model File | Target / Purpose | Mode / Usage |
|---|---|---|
| **`yolov8m.pt`** | Vehicle & Person Detection | Full frame (Auto-downloaded by Ultralytics) |
| **`helmet_best.pt`** | Helmet & Bare-Head Detection | Full frame detector |
| **`helmet_cnn.pt`** | Helmet Crop Classifier | Fallback crop classifier |
| **`plate_koushi.pt`** | License Plate Detection (Stage 1) | Candidate plate bounding-box extraction |
| **`plate_yasir.pt`** | License Plate Verification (Stage 2) | Candidate plate confirmation |
| **`plate_yolov8_moin.pt`**| License Plate Fallback | Fallback plate detector |
| **`traffic_lights_yolov8x.pt`**| Traffic Light Signal Detection | Signal state classification (Red/Yellow/Green) |

---

## 🔄 End-to-End Workflow Diagram

```mermaid
flowchart TD
    A[Raw Frame / Image Input] --> B[Preprocessor: CLAHE + Denoise + Exposure Correction]
    B --> C[Vehicle & Person Detector: YOLOv8m]
    
    C --> D[Helmet Compliance Check: helmet_best.pt]
    D -->|Bare-Head Detected| E[Flag Helmet Violation]
    D -->|No Helmet Detected| F[Track Normal Compliance]

    C --> G[License Plate Extraction Stage 1: plate_koushi.pt]
    G --> H[License Plate Verification Stage 2: plate_yasir.pt]
    H --> I[Plate OCR Read: fast-plate-ocr / EasyOCR]

    C --> J[Violation Classifier: Seatbelt, Triple Riding, Phone Use, Wrong-Way, Signal]
    E --> K[Confidence Router]
    I --> K
    J --> K

    K -->|Tier 1: Conf >= 90%| L[Auto Challan Generated]
    K -->|Tier 2: Conf >= 60%| M[Human Review Queue]
    K -->|Tier 3: Conf < 60%| N[Log & Discard]

    L --> O[Evidence Packager: Annotated Image + JSON Record]
    M --> O

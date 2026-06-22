from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from datetime import datetime
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import (
    AsyncSessionLocal,
    CameraModel,
    JobModel,
    VehicleModel,
    ViolationModel,
    get_db,
    save_violation,
    upsert_vehicle,
)
# Single source of truth for the auto-confirm confidence floor — same
# threshold ml/demo_pipeline.py's ConfidenceRouter uses for Tier 1.
from ml.pipeline.confidence_router import TIER1_AUTO_CHALLAN

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/jobs")

# ---------------------------------------------------------------------------
# ML pipeline — lazy init (shared with _routers.py globals)
# ---------------------------------------------------------------------------

_ml_initialized = False
ml_preprocessor = None
ml_detector = None
ml_ocr = None
ml_classifier = None
ml_driver_state = None
ml_visualizer = None
ml_available = False


def _ensure_ml():
    global _ml_initialized, ml_preprocessor, ml_detector, ml_ocr, ml_classifier, ml_driver_state, ml_visualizer, ml_available
    if _ml_initialized:
        return ml_available
    _ml_initialized = True
    try:
        from pathlib import Path
        from ml.pipeline.preprocessor import ImagePreprocessor
        from ml.pipeline.detector import VehicleDetector
        from ml.pipeline.ocr import PlateOCR
        from ml.pipeline.violation_classifier import ViolationClassifier
        from ml.pipeline.driver_state import DriverStateDetector
        from ml.utils.visualizer import FrameVisualizer

        WEIGHTS_DIR = Path(__file__).parent.parent.parent / "ml" / "models" / "weights"
        # 2-stage plate pipeline: Koushi (Stage-1) + YasirFaiz (Stage-2, auto-loaded by PlateOCR)
        PLATE_WEIGHTS = str(WEIGHTS_DIR / "plate_koushi.pt") if (WEIGHTS_DIR / "plate_koushi.pt").exists() else (
            str(WEIGHTS_DIR / "plate_yolov8_moin.pt") if (WEIGHTS_DIR / "plate_yolov8_moin.pt").exists() else None
        )
        # helmet_best.pt = JarvanLee yolov8m full-image detector (replaces non-working helmet_violation.pt)
        HELMET_WEIGHTS = str(WEIGHTS_DIR / "helmet_cnn.pt") if (WEIGHTS_DIR / "helmet_cnn.pt").exists() else None

        ml_preprocessor = ImagePreprocessor()
        ml_detector = VehicleDetector(model_path=None, device="cpu")
        ml_ocr = PlateOCR(plate_detector_weights=PLATE_WEIGHTS)
        ml_classifier = ViolationClassifier(stop_line_y=380, helmet_weights_path=HELMET_WEIGHTS)
        # enable_phone_detection=False: ml_detector already detects COCO class 67
        # (cell phone) as part of its normal traffic-class scan, so we pass those
        # detections in directly instead of loading a second redundant YOLO model.
        ml_driver_state = DriverStateDetector(enable_phone_detection=False)
        # Same FrameVisualizer used by ml/demo_pipeline.py — reused directly (not
        # reimplemented) so the live backend's "Demo" output is pixel-for-pixel
        # the same drawing code a local `python ml/demo_pipeline.py` run produces.
        ml_visualizer = FrameVisualizer()
        ml_available = True
        logger.info("Jobs ML pipeline initialized. OCR engine: %s", ml_ocr.engine_name)
    except Exception as e:
        logger.error("Failed to initialize ML pipeline in jobs.py: %s", e)
        ml_available = False
    return ml_available


_DEFAULT_CALIBRATION = {
    "stop_line_y": 380,
    "parking_zones": [],
    "traffic_direction": "down",
    "wrong_side_zone": [],
    "calibrated": False,
}


async def _resolve_calibration(camera_id: Optional[str]) -> dict:
    """
    Look up a registered camera's stop-line/parking-zone/traffic-direction
    calibration. Falls back to generic defaults (tagged calibrated=False)
    when no camera_id is given or it isn't registered — e.g. ad-hoc uploads
    with no associated camera.
    """
    if not camera_id:
        return dict(_DEFAULT_CALIBRATION)
    async with AsyncSessionLocal() as session:
        cam = (await session.execute(
            select(CameraModel).where(CameraModel.id == camera_id)
        )).scalar_one_or_none()
    if not cam:
        return dict(_DEFAULT_CALIBRATION)
    return {
        "stop_line_y": cam.stop_line_y,
        "parking_zones": json.loads(cam.parking_zones or "[]"),
        "traffic_direction": cam.traffic_direction or "down",
        "wrong_side_zone": json.loads(cam.wrong_side_zone or "[]"),
        "calibrated": True,
    }


def _apply_calibration(calibration: dict) -> None:
    """
    Point the shared ml_classifier singleton at this job's camera calibration.

    NOTE: ml_classifier is a module-level singleton (avoids reloading model
    weights per job). Setting these attributes is not concurrency-safe if
    multiple jobs for *different* cameras are processed in true parallel —
    acceptable for this single-process backend's current background-task
    model, but worth flagging if jobs ever move to a worker pool.
    """
    ml_classifier.stop_line_y = calibration["stop_line_y"]
    ml_classifier.parking_zones = calibration["parking_zones"]
    ml_classifier.traffic_direction = calibration["traffic_direction"]
    ml_classifier.wrong_side_zone = calibration["wrong_side_zone"]


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class JobCreate(BaseModel):
    name: str
    source_type: str  # "Image" or "Video"
    camera_id: Optional[str] = None  # registered camera to inherit calibration from


class JobResponse(BaseModel):
    id: str
    name: str
    source_type: str
    progress: int
    status: str
    duration: int
    frames_processed: int
    violations_found: int
    upload_time: str
    camera_id: Optional[str] = None

    class Config:
        from_attributes = True


class JobResultResponse(BaseModel):
    """GET /jobs/{job_id}/result — the full, uncollapsed real-pipeline
    breakdown for every image/frame processed in this job, clubbed under
    one job_id. Powers the Evidence page's step-by-step view."""
    job: JobResponse
    records: List[dict]


class ViolationInJobResponse(BaseModel):
    id: str
    timestamp: str
    violation_type: str
    confidence: float
    severity: str
    plate_text: str
    plate_conf: float
    vehicle_class: str
    camera_id: str
    location: str
    annotated_img: str
    raw_img: str
    json_record: str
    status: str

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Shared per-frame classification + evidence packaging
# (used by both the untracked image path and the tracked video path)
# ---------------------------------------------------------------------------

V_TYPE_DISPLAY = {
    "helmet_non_compliance": "No Helmet",
    "seatbelt_non_compliance": "Seatbelt",
    "triple_riding": "Triple Riding",
    "wrong_side_driving": "Wrong Way",
    "stop_line_violation": "Stop Line",
    "red_light_violation": "Red Light",
    "illegal_parking": "Illegal Parking",
    "phone_use_while_driving": "Phone Use",
    "drowsy_driving": "Drowsy",
}

_NOISE_PLATE_TEXTS = ("UNCLEAR", "PLATE-UNREAD", "")


def _draw_annotated_evidence(
    frame: np.ndarray,
    violations: List[dict],
    plate_text: str,
    plate_conf: float,
    location_label: str,
    vid: str,
) -> np.ndarray:
    """
    The official "Annotated Evidence" image — final violations only, no
    detection noise. Visual style mirrors ml/utils/evidence.py's
    EvidencePackager._annotate (header/footer bars, corner-accented violation
    boxes) so the live backend's evidence output matches the one a local
    `python ml/demo_pipeline.py` run would file. Unlike the old per-violation
    drawing, this draws every violation found in the image in one pass, and
    skips the plate line entirely when nothing was actually read — an
    "UNCLEAR (0%)" tag is noise, not evidence.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    ts = datetime.utcnow()

    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (12, 12, 22), -1)
    cv2.addWeighted(overlay, 0.80, annotated, 0.20, 0, annotated)
    cv2.putText(annotated, "GARUDA — AUTOMATED TRAFFIC ENFORCEMENT", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 212, 255), 2)
    cv2.putText(annotated, f"{location_label}  |  {ts.strftime('%Y-%m-%d %H:%M:%S')} UTC",
                (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1)

    severity_colors = {"critical": (0, 0, 255), "high": (0, 50, 255), "medium": (0, 140, 255), "low": (0, 210, 255)}
    if violations:
        for v in violations:
            bbox = v.get("bbox") or []
            if len(bbox) != 4:
                continue
            color = severity_colors.get(v.get("severity", "medium"), (0, 140, 255))
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            cl = 18
            for cx, cy, dx, dy in [(x1, y1, cl, cl), (x2, y1, -cl, cl), (x1, y2, cl, -cl), (x2, y2, -cl, -cl)]:
                cv2.line(annotated, (cx, cy), (cx + dx, cy), color, 4)
                cv2.line(annotated, (cx, cy), (cx, cy + dy), color, 4)
            vtype = v.get("type", "violation").replace("_", " ").upper()
            conf = v.get("confidence", 0)
            plate_txt = v.get("plate_text")
            label = f"{vtype}  {conf * 100:.0f}%"
            if plate_txt and plate_txt != "UNCLEAR":
                label += f" [{plate_txt}]"
            lw = len(label) * 8 + 10
            label_y1 = max(0, y1 - 30)
            cv2.rectangle(annotated, (x1, label_y1), (x1 + lw, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 4, max(16, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
    else:
        cv2.putText(annotated, "COMPLIANT — No Violation Detected", (12, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 200, 80), 2)

    overlay2 = annotated.copy()
    cv2.rectangle(overlay2, (0, h - 56), (w, h), (10, 10, 18), -1)
    cv2.addWeighted(overlay2, 0.80, annotated, 0.20, 0, annotated)

    if plate_text and plate_text not in _NOISE_PLATE_TEXTS:
        cv2.putText(annotated, f"PLATE: {plate_text}   ({plate_conf * 100:.0f}% OCR confidence)",
                    (10, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 240, 100), 2)
    cv2.putText(annotated, f"GARUDA v1.0  |  {vid}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (130, 130, 130), 1)

    return annotated


def _draw_demo_visualization(
    frame: np.ndarray,
    detections: list,
    violations: List[dict],
    driver_alerts: list,
    tier: int,
    action: str,
    plate_text: str,
    plate_conf: float,
    plate_valid: bool,
    stop_line_y: int,
) -> np.ndarray:
    """
    The "Demo" debug visualization — identical draw order to
    ml/demo_pipeline.py's run_image(): all raw detections, all violation
    boxes, the calibrated stop line, the strongest driver alert, the routing
    tier badge, then the plate result. Reuses the same FrameVisualizer
    instance as the local script, so this is exactly what running
    `python ml/demo_pipeline.py --input <this image>` would produce — not a
    re-derived approximation.
    """
    demo = frame.copy()
    demo = ml_visualizer.draw_detections(demo, [d.to_dict() for d in detections])
    if violations:
        demo = ml_visualizer.draw_violations(demo, violations)
    ml_visualizer.draw_stop_line(demo, stop_line_y)
    if driver_alerts:
        ml_visualizer.draw_driver_alert(demo, driver_alerts[0].alert_type)
    ml_visualizer.draw_tier_badge(demo, tier, action, (10, 90))
    ml_visualizer.draw_plate_result(demo, plate_text or "UNCLEAR", plate_conf, plate_valid)
    return demo


def _classify_and_package(
    img: np.ndarray,
    processed: np.ndarray,
    job_id: str,
    source_name: str,
    t_start: float,
    detections: list,
    vehicles: list,
    persons: list,
    tracker_states: Optional[dict] = None,
    calibrated: bool = False,
) -> List[dict]:
    """
    Driver state, OCR, violation classification, and evidence packaging for
    one frame — processed exactly once and packaged as exactly ONE result:
    one record (with every violation found, if any, listed inside it), one
    raw image, one annotated image, one demo image. A frame with 3 helmet
    violations produces 1 record with 3 entries in `violations`, not 3
    separate records/images — splitting one frame into many was the bug.

    tracker_states (from a VehicleTracker) enables the velocity/duration-aware
    track-based checks for stop-line/red-light/wrong-side/illegal-parking;
    pass None for single-image jobs with no track history, which falls back
    to the single-frame static checks.
    """
    # Driver state — drowsiness/yawn via MediaPipe FaceLandmarker on the whole
    # frame. A single still image rarely accumulates enough consecutive
    # low-EAR/high-MAR frames to cross the alert threshold (that's designed
    # for video), so alerts will usually legitimately be empty for images —
    # that's a real result, not a placeholder.
    driver_alerts = ml_driver_state.analyze_frame(processed, track_id=0)

    # OCR on each vehicle — collect all visible plates. Uses
    # read_plate_from_vehicle which scans the full vehicle crop for text.
    all_plates = []
    for veh in vehicles:
        x1, y1, x2, y2 = map(int, veh.bbox)
        h_img, w_img = processed.shape[:2]
        veh_crop = processed[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]
        if veh_crop.size > 0:
            ocr_result = ml_ocr.read_plate_from_vehicle(veh_crop)
            veh.plate_text = ocr_result.formatted_text or "UNCLEAR"
            veh.plate_conf = ocr_result.confidence
            all_plates.append({
                "plate_text": ocr_result.formatted_text or "UNCLEAR",
                "confidence": round(ocr_result.confidence, 3),
                "vehicle_class": veh.class_name,
                "bbox": list(map(int, veh.bbox)),
                "ocr_engine": ocr_result.ocr_engine,
                "state": ocr_result.state_name,
                "is_valid": ocr_result.is_valid,
            })

    # Violation classification — runs once for the whole image (pass full
    # frame for signal + phone detections, and tracker_states so video jobs
    # get the track-based checks). All violations across all vehicles in
    # this one frame come back in a single list.
    phone_detections = [d for d in detections if d.class_name == "cell phone"]
    violations = ml_classifier.check_all(
        processed, vehicles, persons,
        signal_frame=processed,
        phone_detections=phone_detections,
        tracker_states=tracker_states,
    )

    driver_state_dict = {
        "alerts": [a.to_dict() for a in driver_alerts],
        "total_alerts": len(driver_alerts),
    }

    inference_ms = round((time.perf_counter() - t_start) * 1000, 1)
    timestamp_now = datetime.utcnow().isoformat() + "Z"

    # All evidence for this job is clubbed into one job-scoped directory per
    # output type — one image's outputs, or many images'/frames' outputs,
    # always land together under evidence/{raw,annotated,demo}/{job_id}/.
    os.makedirs(f"evidence/raw/{job_id}", exist_ok=True)
    os.makedirs(f"evidence/annotated/{job_id}", exist_ok=True)
    os.makedirs(f"evidence/demo/{job_id}", exist_ok=True)

    raw_path = f"evidence/raw/{job_id}/{source_name}.jpg"
    cv2.imwrite(raw_path, img)

    vid = f"VIO-JOB-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4].upper()}"

    # Best plate for this whole image — highest-confidence validly-read plate,
    # falling back to the highest-confidence reading of any kind.
    valid_plates = [p for p in all_plates if p.get("is_valid") and p["plate_text"] not in _NOISE_PLATE_TEXTS]
    if valid_plates:
        best_plate = max(valid_plates, key=lambda p: p["confidence"])
    elif all_plates:
        best_plate = max(all_plates, key=lambda p: p["confidence"])
    else:
        best_plate = {"plate_text": "UNCLEAR", "confidence": 0.0, "vehicle_class": "unknown", "is_valid": False, "state": "Unknown"}

    # Per-violation routing: a violation only needs a human look when it's
    # below TIER1_AUTO_CHALLAN confidence. Clear, high-confidence violations
    # are auto-confirmed straight into the Violation Center — only the
    # genuinely uncertain ones land in the human review queue.
    violation_dicts = []
    for v in violations:
        v_type_display = V_TYPE_DISPLAY.get(v.violation_type.value, v.violation_type.value)
        v_tier = 1 if v.confidence >= TIER1_AUTO_CHALLAN else 2
        # Match vehicle by bbox to assign its specific plate
        v_plate = "UNCLEAR"
        for p in all_plates:
            if p["bbox"] == list(map(int, v.bbox)):
                v_plate = p["plate_text"]
                break
        violation_dicts.append({
            "type": v_type_display,
            "confidence": v.confidence,
            "severity": v.severity,
            "fine_amount_inr": v.fine_amount if hasattr(v, "fine_amount") else 1000,
            "bbox": list(map(int, v.bbox)),
            "metadata": v.metadata if hasattr(v, "metadata") else {},
            "tier": v_tier,
            "review_status": "auto_confirmed" if v_tier == 1 else "pending",
            "plate_text": v_plate,
        })

    # Record-level tier/action reflects the WORST case across all violations
    # in this one image — if even one violation needs a human look, the
    # whole image goes to the review queue; only when every violation in it
    # is independently clear does the record skip review entirely.
    if not violations:
        tier, action = 1, "PASSED"
    elif all(vd["tier"] == 1 for vd in violation_dicts):
        tier, action = 1, "AUTO_CHALLAN"
    else:
        tier, action = 2, "HUMAN_REVIEW"

    annotated_img = _draw_annotated_evidence(
        img, violation_dicts, best_plate["plate_text"], best_plate["confidence"],
        source_name, vid,
    )
    annotated_path = f"evidence/annotated/{job_id}/{vid}.jpg"
    cv2.imwrite(annotated_path, annotated_img)

    demo_img = _draw_demo_visualization(
        img, detections, violation_dicts, driver_alerts, tier, action,
        best_plate["plate_text"], best_plate["confidence"], best_plate.get("is_valid", False),
        ml_classifier.stop_line_y,
    )
    demo_path = f"evidence/demo/{job_id}/{vid}.jpg"
    cv2.imwrite(demo_path, demo_img)

    record = {
        "violation_id": vid,
        "tier": tier,
        "action": action,
        "timestamp": timestamp_now,
        "camera": {"id": job_id, "location": source_name, "coordinates": {}},
        "vehicle": {
            "vehicle_class": best_plate.get("vehicle_class", "unknown"),
            "license_plate": best_plate["plate_text"],
            "plate_confidence": best_plate["confidence"],
            "plate_valid": best_plate.get("is_valid", False),
            "plate_state": best_plate.get("state", "Unknown"),
        },
        "violations": violation_dicts,
        "all_plates_detected": all_plates,
        "processing": {
            "inference_device": "CPU",
            "inference_time_ms": inference_ms,
            "model": ml_detector.model_name,
            "ocr_engine": ml_ocr.engine_name,
            "vehicles_detected": len(vehicles),
            "persons_detected": len(persons),
            "camera_calibrated": calibrated,
        },
        "driver_state": driver_state_dict,
        "evidence": {
            "annotated_image": f"/{annotated_path}",
            "raw_frame": f"/{raw_path}",
            "demo_image": f"/{demo_path}",
        },
    }

    return [{
        "record": record,
        "plate_text": best_plate["plate_text"],
        "vehicle_class": best_plate.get("vehicle_class", "unknown"),
        "tier": tier,
        "status": "passed" if not violations else "pending",
    }]


# ---------------------------------------------------------------------------
# Real ML inference on a single image file (untracked — no track history)
# ---------------------------------------------------------------------------

def _run_ml_on_image(
    img: np.ndarray,
    job_id: str,
    source_name: str,
    is_video: bool = False,
    calibration: Optional[dict] = None,
) -> List[dict]:
    """Run the full GARUDA ML pipeline on a single image. Returns list of violation records."""
    if not _ensure_ml() or not ml_available:
        logger.warning("ML pipeline not available — returning no detections for job %s", job_id)
        return []

    calibration = calibration or _DEFAULT_CALIBRATION
    _apply_calibration(calibration)

    t_start = time.perf_counter()
    try:
        # Preprocess (optimized with brightness-based bypass and downscaling)
        processed = ml_preprocessor.preprocess(img, is_video=is_video)

        # Detect vehicles + persons + phones — no tracking for a single image,
        # so stop-line/red-light/wrong-side/illegal-parking fall back to the
        # single-frame static checks inside ml_classifier.check_all().
        detections = ml_detector.detect(processed)
        vehicles = ml_detector.get_vehicles(detections)
        persons = ml_detector.get_persons(detections)
        phones = ml_detector.get_phones(detections)

        logger.info(
            "Job %s | Image %s | %d vehicles, %d persons, %d phones detected",
            job_id, source_name, len(vehicles), len(persons), len(phones)
        )

        return _classify_and_package(
            img, processed, job_id, source_name, t_start,
            detections, vehicles, persons,
            tracker_states=None,
            calibrated=calibration["calibrated"],
        )
    except Exception as e:
        logger.error("ML inference error in job %s: %s", job_id, e, exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Real ML inference on a batch of independent image files
# ---------------------------------------------------------------------------

def _run_ml_on_batch(
    files: List[tuple],  # [(file_bytes, filename), ...]
    job_id: str,
    calibration: Optional[dict] = None,
) -> List[dict]:
    """
    Run the full ML pipeline independently on each uploaded image, clubbing
    every result under one job_id/evidence folder. Unlike video frames,
    batch images are unrelated photos — no tracking across them — and every
    image's result is kept (no all-compliant collapsing), since each was a
    distinct, deliberate upload rather than a redundant sampled frame.
    """
    if not _ensure_ml() or not ml_available:
        logger.warning("ML pipeline not available — skipping batch job %s", job_id)
        return []

    calibration = calibration or _DEFAULT_CALIBRATION
    _apply_calibration(calibration)

    all_results: List[dict] = []
    for file_bytes, filename in files:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Job %s: could not decode batch image %s", job_id, filename)
            continue

        t_start = time.perf_counter()
        try:
            processed = ml_preprocessor.preprocess(img, is_video=False)
            detections = ml_detector.detect(processed)
            vehicles = ml_detector.get_vehicles(detections)
            persons = ml_detector.get_persons(detections)
            phones = ml_detector.get_phones(detections)

            logger.info(
                "Job %s | Batch image %s | %d vehicles, %d persons, %d phones detected",
                job_id, filename, len(vehicles), len(persons), len(phones)
            )

            source_name = os.path.splitext(os.path.basename(filename))[0]
            all_results.extend(_classify_and_package(
                img, processed, job_id, source_name, t_start,
                detections, vehicles, persons,
                tracker_states=None,
                calibrated=calibration["calibrated"],
            ))
        except Exception as e:
            logger.error("ML inference error in batch job %s (file %s): %s", job_id, filename, e, exc_info=True)

    return all_results


# ---------------------------------------------------------------------------
# Real ML inference on a video file (frame sampling + persistent tracking)
# ---------------------------------------------------------------------------

# Sample roughly 1 frame/sec of footage rather than every frame — at ~1s of
# ML inference per frame (see _run_ml_on_image benchmark), processing every
# frame of even a short clip would take minutes. MAX_VIDEO_FRAMES bounds total
# processing time on CPU for a single upload to under a minute.
VIDEO_SAMPLE_FPS = 1.0
MAX_VIDEO_FRAMES = 40


def _run_ml_on_video(
    file_bytes: bytes,
    job_id: str,
    filename: str,
    calibration: Optional[dict] = None,
):
    """
    Decode a video file, sample frames, and run the full ML pipeline on each
    — with a persistent VehicleTracker across sampled frames so stop-line/
    red-light/wrong-side/illegal-parking get real velocity/duration history
    instead of degrading to the single-frame static fallback on every frame.

    Returns (violation records, frames actually processed).
    """
    if not _ensure_ml() or not ml_available:
        logger.warning("ML pipeline not available — skipping video job %s", job_id)
        return [], 0

    from ml.pipeline.tracker import VehicleTracker

    calibration = calibration or _DEFAULT_CALIBRATION
    _apply_calibration(calibration)

    suffix = os.path.splitext(filename)[1] or ".mp4"
    base_name = os.path.splitext(os.path.basename(filename))[0]
    tmp_path = None
    violation_results: List[dict] = []
    passed_results: List[dict] = []
    frames_processed = 0
    tracker = VehicleTracker(stop_line_y=calibration["stop_line_y"])
    first_sampled_frame = True

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            logger.error("Job %s: could not open video file %s", job_id, filename)
            return [], 0

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        sample_interval = max(1, round(fps / VIDEO_SAMPLE_FPS))

        frame_idx = 0
        while frames_processed < MAX_VIDEO_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_interval == 0:
                t_start = time.perf_counter()
                processed = ml_preprocessor.preprocess(frame, is_video=True)

                # persist=False on the first sampled frame resets ByteTrack
                # state so IDs don't leak in from a previous, unrelated job
                # that shares the same VehicleDetector singleton.
                detections = ml_detector.detect_with_tracking(processed, persist=not first_sampled_frame)
                first_sampled_frame = False
                tracker.update(detections, frame_idx)

                vehicles = ml_detector.get_vehicles(detections)
                persons = ml_detector.get_persons(detections)
                tracker_states = {s.track_id: s for s in tracker.active_tracks()}

                frame_results = _classify_and_package(
                    frame, processed, job_id, f"{base_name}_f{frame_idx}", t_start,
                    detections, vehicles, persons,
                    tracker_states=tracker_states,
                    calibrated=calibration["calibrated"],
                )
                for r in frame_results:
                    (passed_results if r["status"] == "passed" else violation_results).append(r)
                frames_processed += 1
            frame_idx += 1

        cap.release()
    except Exception as e:
        logger.error("ML inference error in video job %s: %s", job_id, e, exc_info=True)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    logger.info(
        "Job %s | Video %s | sampled %d frames, %d violations found",
        job_id, filename, frames_processed, len(violation_results)
    )

    # Keep every real violation found across the whole video; collapse an
    # all-compliant video down to one representative "passed" record instead
    # of one per sampled frame (avoids flooding the DB/evidence store).
    final_results = violation_results if violation_results else passed_results[:1]
    return final_results, frames_processed


# ---------------------------------------------------------------------------
# Background job runner
# ---------------------------------------------------------------------------

async def run_job_pipeline(
    job_id: str,
    file_bytes: Optional[bytes] = None,
    filename: str = "upload",
    source_type: str = "Image",
    camera_id: Optional[str] = None,
    batch_files: Optional[List[tuple]] = None,  # [(bytes, filename), ...] for source_type="Batch"
):
    """Real ML inference pipeline for batch upload jobs."""
    start = datetime.utcnow()

    async def _update_job(**kwargs):
        async with AsyncSessionLocal() as session:
            job = (await session.execute(
                select(JobModel).where(JobModel.id == job_id)
            )).scalar_one_or_none()
            if job:
                for k, v in kwargs.items():
                    setattr(job, k, v)
                await session.commit()

    await _update_job(status="Processing", progress=10)

    # Resolve this job's camera calibration once, here in the async context
    # (DB access), then hand the plain dict to the sync ML functions running
    # in the thread pool below — keeps DB I/O off the executor thread.
    calibration = await _resolve_calibration(camera_id)

    violations_created = []
    frames_processed = 0

    if batch_files or file_bytes is not None:
        loop = asyncio.get_event_loop()

        if source_type.lower() == "batch":
            await _update_job(progress=20)
            violations_created = await loop.run_in_executor(
                None, _run_ml_on_batch, batch_files, job_id, calibration
            )
            frames_processed = len(batch_files)
        elif source_type.lower() == "video":
            await _update_job(progress=20)
            violations_created, frames_processed = await loop.run_in_executor(
                None, _run_ml_on_video, file_bytes, job_id, filename, calibration
            )
        else:
            # Decode image
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                logger.error("Job %s: Failed to decode image bytes", job_id)
                await _update_job(status="Failed", progress=0)
                return

            await _update_job(progress=30)

            # Run ML inference in thread pool to avoid blocking event loop
            violations_created = await loop.run_in_executor(
                None, _run_ml_on_image, img, job_id, filename, False, calibration
            )
            frames_processed = 1

        await _update_job(progress=70)

        # Save only genuine violations to DB — compliant images contribute
        # nothing to the violation registry. One image's record carries every
        # violation found in it (e.g. helmet + triple riding on the same
        # frame) and becomes ONE clubbed citation row — not one row per
        # violation — since an officer reviews the whole image at once, not
        # each bounding box in isolation. save_violation() rolls the list up
        # into one summary (joined types, total fine, worst severity).
        violation_count = 0
        for v_data in violations_created:
            record = v_data["record"]
            record_violations = record.get("violations", [])
            if not record_violations:
                continue
            try:
                async with AsyncSessionLocal() as session:
                    await save_violation(session, record)
                    if v_data["plate_text"] and v_data["plate_text"] != "UNCLEAR":
                        for v in record_violations:
                            await upsert_vehicle(session, v_data["plate_text"], v["type"])
                    violation_count += len(record_violations)
            except Exception as e:
                logger.error("Failed to save violation record: %s", e)

        # Full, uncollapsed real-pipeline breakdown for every item processed —
        # this is what the Evidence page reads via GET /jobs/{job_id}/result,
        # independent of the (intentionally lossier) violations table above.
        result_summary = json.dumps(
            {"records": [v_data["record"] for v_data in violations_created]},
            default=str,
        )

        elapsed = max(1, int((datetime.utcnow() - start).total_seconds()))
        await _update_job(
            status="Completed",
            progress=100,
            duration=elapsed,
            frames_processed=frames_processed,
            violations_found=violation_count,
            result_summary=result_summary,
        )
        logger.info(
            "Job %s completed: %d violations found in %s (%d frames processed)",
            job_id, violation_count, filename, frames_processed,
        )
    else:
        # No file uploaded — simulate basic progress (no real inference)
        await asyncio.sleep(2)
        await _update_job(progress=50, duration=2, frames_processed=0)
        await asyncio.sleep(2)
        await _update_job(status="Completed", progress=100, duration=4, frames_processed=0, violations_found=0)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("", response_model=List[JobResponse])
async def list_jobs(db: AsyncSession = Depends(get_db)):
    rows = (
        await db.execute(select(JobModel).order_by(JobModel.upload_time.desc()))
    ).scalars().all()
    return [JobResponse.model_validate(r) for r in rows]


@router.post("", response_model=JobResponse, status_code=201)
async def create_job(
    body: JobCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Create a job without a file (metadata only). Use /upload for file-based jobs."""
    job_id = f"JOB-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4].upper()}"
    job = JobModel(
        id=job_id,
        name=body.name,
        source_type=body.source_type,
        progress=0,
        status="Queued",
        duration=0,
        frames_processed=0,
        violations_found=0,
        upload_time=datetime.utcnow().isoformat() + "Z",
        camera_id=body.camera_id,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    background_tasks.add_task(run_job_pipeline, job_id, None, body.name, body.source_type, body.camera_id)
    return JobResponse.model_validate(job)


@router.post("/upload", response_model=JobResponse, status_code=201)
async def upload_job(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    name: str = Form(...),
    source_type: str = Form("Image"),
    camera_id: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    """Upload a single image/video file and run ML inference pipeline in the background.

    camera_id is optional — when it matches a registered camera, stop-line/
    red-light/wrong-side/illegal-parking detection use that camera's
    calibration instead of generic defaults.
    """
    job_id = f"JOB-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4].upper()}"
    job = JobModel(
        id=job_id,
        name=name,
        source_type=source_type,
        progress=0,
        status="Queued",
        duration=0,
        frames_processed=0,
        violations_found=0,
        upload_time=datetime.utcnow().isoformat() + "Z",
        camera_id=camera_id,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Read file contents now (before request context ends)
    file_bytes = await file.read()
    filename = file.filename or name

    background_tasks.add_task(run_job_pipeline, job_id, file_bytes, filename, source_type, camera_id)
    logger.info("Job %s queued for ML inference on file: %s (%d bytes)", job_id, filename, len(file_bytes))
    return JobResponse.model_validate(job)


@router.post("/upload-batch", response_model=JobResponse, status_code=201)
async def upload_batch_job(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    name: str = Form(...),
    camera_id: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
):
    """Upload multiple independent images as one batch job.

    All images are processed independently (no tracking between them, since
    they're unrelated photos, not video frames) and clubbed under one
    job_id — one result, one evidence folder, regardless of how many images
    were submitted. Mirrors /upload's camera_id calibration behavior.
    """
    if not files:
        raise HTTPException(400, "At least one file is required")

    job_id = f"JOB-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4].upper()}"
    job = JobModel(
        id=job_id,
        name=name,
        source_type="Batch",
        progress=0,
        status="Queued",
        duration=0,
        frames_processed=0,
        violations_found=0,
        upload_time=datetime.utcnow().isoformat() + "Z",
        camera_id=camera_id,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Read all file contents now (before request context ends)
    batch_files = [(await f.read(), f.filename or f"image_{i}") for i, f in enumerate(files)]

    background_tasks.add_task(
        run_job_pipeline, job_id, None, name, "Batch", camera_id, batch_files
    )
    logger.info("Job %s queued for batch ML inference on %d files", job_id, len(batch_files))
    return JobResponse.model_validate(job)


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, db: AsyncSession = Depends(get_db)):
    job = (
        await db.execute(select(JobModel).where(JobModel.id == job_id))
    ).scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobResponse.model_validate(job)


@router.get("/{job_id}/result", response_model=JobResultResponse)
async def get_job_result(job_id: str, db: AsyncSession = Depends(get_db)):
    """
    Full, uncollapsed real-pipeline breakdown for this job — every image's
    or sampled frame's record (violation or compliant), clubbed under this
    one job_id. This is what the Evidence page renders; unlike
    /jobs/{job_id}/violations, nothing here is filtered out or collapsed.
    """
    job = (
        await db.execute(select(JobModel).where(JobModel.id == job_id))
    ).scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    try:
        records = json.loads(job.result_summary or "{}").get("records", [])
    except (json.JSONDecodeError, AttributeError):
        records = []

    return JobResultResponse(job=JobResponse.model_validate(job), records=records)


@router.get("/{job_id}/violations", response_model=List[ViolationInJobResponse])
async def get_job_violations(job_id: str, db: AsyncSession = Depends(get_db)):
    """Return all violation records associated with a specific job (camera_id prefix match)."""
    rows = (
        await db.execute(
            select(ViolationModel)
            .where(ViolationModel.camera_id.like(f"{job_id}%"))
            .order_by(ViolationModel.created_at.desc())
        )
    ).scalars().all()
    return [ViolationInJobResponse.model_validate(r) for r in rows]


def _remove_job_evidence(job_id: str) -> None:
    """Delete this job's evidence files. Citation rows store
    annotated/raw/demo paths scoped under evidence/{type}/{job_id}/, so
    removing those three directories removes everything that job produced.
    """
    for kind in ("raw", "annotated", "demo"):
        shutil.rmtree(f"evidence/{kind}/{job_id}", ignore_errors=True)


@router.delete("/{job_id}", status_code=204)
async def delete_job(job_id: str, db: AsyncSession = Depends(get_db)):
    """
    Delete one job: its DB row, every violation citation it produced
    (camera_id prefix match — covers the "-1"/"-2" per-violation suffixes
    from a multi-violation image), and its evidence images on disk.
    """
    job = (await db.execute(select(JobModel).where(JobModel.id == job_id))).scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    await db.execute(delete(ViolationModel).where(ViolationModel.camera_id.like(f"{job_id}%")))
    await db.execute(delete(JobModel).where(JobModel.id == job_id))
    await db.commit()

    _remove_job_evidence(job_id)
    logger.info("Deleted job %s (DB rows + evidence files)", job_id)
    return None


@router.delete("", status_code=204)
async def clear_all_jobs(db: AsyncSession = Depends(get_db)):
    """
    Batch-clear every job, every violation citation, and every repeat-offender
    vehicle record, plus all evidence images on disk. A full reset of
    everything the ML pipeline has produced — registered cameras and users
    are untouched.
    """
    await db.execute(delete(ViolationModel))
    await db.execute(delete(JobModel))
    await db.execute(delete(VehicleModel))
    await db.commit()

    for kind in ("raw", "annotated", "demo"):
        shutil.rmtree(f"evidence/{kind}", ignore_errors=True)
        os.makedirs(f"evidence/{kind}", exist_ok=True)

    logger.warning("Cleared all jobs, violations, vehicles, and evidence files")
    return None

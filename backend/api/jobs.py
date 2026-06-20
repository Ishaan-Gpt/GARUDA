from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import (
    AsyncSessionLocal,
    JobModel,
    ViolationModel,
    get_db,
    save_violation,
    upsert_vehicle,
)

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
ml_available = False


def _ensure_ml():
    global _ml_initialized, ml_preprocessor, ml_detector, ml_ocr, ml_classifier, ml_available
    if _ml_initialized:
        return ml_available
    _ml_initialized = True
    try:
        from pathlib import Path
        from ml.pipeline.preprocessor import ImagePreprocessor
        from ml.pipeline.detector import VehicleDetector
        from ml.pipeline.ocr import PlateOCR
        from ml.pipeline.violation_classifier import ViolationClassifier

        WEIGHTS_DIR = Path(__file__).parent.parent.parent / "ml" / "models" / "weights"
        PLATE_WEIGHTS = str(WEIGHTS_DIR / "plate_yolo.pt") if (WEIGHTS_DIR / "plate_yolo.pt").exists() else None
        HELMET_WEIGHTS = str(WEIGHTS_DIR / "helmet_cnn.pt") if (WEIGHTS_DIR / "helmet_cnn.pt").exists() else None

        ml_preprocessor = ImagePreprocessor()
        ml_detector = VehicleDetector(model_path=None, device="cpu")
        ml_ocr = PlateOCR(plate_detector_weights=PLATE_WEIGHTS)
        ml_classifier = ViolationClassifier(stop_line_y=380, helmet_weights_path=HELMET_WEIGHTS)
        ml_available = True
        logger.info("Jobs ML pipeline initialized. OCR engine: %s", ml_ocr.engine_name)
    except Exception as e:
        logger.error("Failed to initialize ML pipeline in jobs.py: %s", e)
        ml_available = False
    return ml_available


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class JobCreate(BaseModel):
    name: str
    source_type: str  # "Image" or "Video"


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

    class Config:
        from_attributes = True


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
# Real ML inference on an image file
# ---------------------------------------------------------------------------

def _run_ml_on_image(
    img: np.ndarray,
    job_id: str,
    source_name: str,
) -> List[dict]:
    """Run the full GARUDA ML pipeline on a single image. Returns list of violation records."""
    results = []
    if not _ensure_ml() or not ml_available:
        logger.warning("ML pipeline not available — returning no detections for job %s", job_id)
        return results

    try:
        # Stage 1: Preprocess
        processed = ml_preprocessor._enhance_low_light(img)
        processed = ml_preprocessor._normalize_exposure(processed)
        h, w = processed.shape[:2]

        # Stage 2: Detect vehicles + persons
        detections = ml_detector.detect(processed)
        vehicles = ml_detector.get_vehicles(detections)
        persons = ml_detector.get_persons(detections)

        logger.info(
            "Job %s | Image %s | %d vehicles, %d persons detected",
            job_id, source_name, len(vehicles), len(persons)
        )

        # Stage 3: OCR on each vehicle — collect all visible plates
        # Use read_plate_from_vehicle which scans the full vehicle crop for text
        all_plates = []
        for veh in vehicles:
            x1, y1, x2, y2 = map(int, veh.bbox)
            h_img, w_img = processed.shape[:2]
            veh_crop = processed[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]
            if veh_crop.size > 0:
                ocr_result = ml_ocr.read_plate_from_vehicle(veh_crop)
                all_plates.append({
                    "plate_text": ocr_result.formatted_text or "UNCLEAR",
                    "confidence": round(ocr_result.confidence, 3),
                    "vehicle_class": veh.class_name,
                    "bbox": list(map(int, veh.bbox)),
                    "ocr_engine": ocr_result.ocr_engine,
                    "state": ocr_result.state_name,
                    "is_valid": ocr_result.is_valid,
                })

        # Stage 4: Violation classification
        violations = ml_classifier.check_all(processed, vehicles, persons)

        timestamp_now = datetime.utcnow().isoformat() + "Z"
        os.makedirs("evidence/annotated", exist_ok=True)
        os.makedirs("evidence/raw", exist_ok=True)

        # Save raw original frame
        raw_path = f"evidence/raw/{job_id}_{source_name}.jpg"
        cv2.imwrite(raw_path, img)

        if not violations:
            # No violations — save one "passed" record for this image
            vid = f"VIO-JOB-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4].upper()}"
            annotated_path = f"evidence/annotated/{vid}.jpg"

            # Draw green border + all plate labels
            annotated = processed.copy()
            cv2.rectangle(annotated, (4, 4), (w - 4, h - 4), (0, 200, 80), 3)
            cv2.putText(annotated, "COMPLIANT — No Violation", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 80), 2)
            for i, p in enumerate(all_plates):
                x1, y1, x2, y2 = p["bbox"]
                color = (0, 255, 80)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{p['plate_text']} ({int(p['confidence']*100)}%)"
                cv2.putText(annotated, label, (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.imwrite(annotated_path, annotated)

            best_plate = all_plates[0] if all_plates else {"plate_text": "UNCLEAR", "confidence": 0.0, "vehicle_class": "unknown"}
            record = {
                "violation_id": vid,
                "tier": 1,
                "action": "PASSED",
                "timestamp": timestamp_now,
                "camera": {"id": f"JOB-{job_id}", "location": source_name, "coordinates": {}},
                "vehicle": {
                    "vehicle_class": best_plate.get("vehicle_class", "unknown"),
                    "license_plate": best_plate["plate_text"],
                    "plate_confidence": best_plate["confidence"],
                    "plate_valid": True,
                },
                "violations": [],
                "all_plates_detected": all_plates,
                "processing": {
                    "inference_device": "CPU",
                    "model": "yolo11n",
                    "ocr_engine": ml_ocr.engine_name,
                    "vehicles_detected": len(vehicles),
                    "persons_detected": len(persons),
                },
                "driver_state": {"alerts": [], "total_alerts": 0},
                "evidence": {
                    "annotated_image": f"/evidence/annotated/{vid}.jpg",
                    "raw_frame": f"/{raw_path}",
                },
            }
            results.append({
                "record": record,
                "plate_text": best_plate["plate_text"],
                "plate_conf": best_plate["confidence"],
                "vehicle_class": best_plate.get("vehicle_class", "unknown"),
                "violation_type": "Passed / No Violation",
                "confidence": 1.0,
                "severity": "none",
                "tier": 1,
                "annotated_img": f"/evidence/annotated/{vid}.jpg",
                "raw_img": f"/{raw_path}",
                "status": "passed",
            })
        else:
            for v in violations:
                vid = f"VIO-JOB-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4].upper()}"
                annotated_path = f"evidence/annotated/{vid}.jpg"

                # Annotate the violation on the frame
                annotated = processed.copy()
                vx1, vy1, vx2, vy2 = map(int, v.bbox)
                cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (0, 0, 255), 3)

                v_type_raw = v.violation_type.value
                v_type_display = {
                    "helmet_non_compliance": "No Helmet",
                    "seatbelt_non_compliance": "Seatbelt",
                    "triple_riding": "Triple Riding",
                    "wrong_side_driving": "Wrong Way",
                    "stop_line_violation": "Stop Line",
                    "red_light_violation": "Red Light",
                    "illegal_parking": "Illegal Parking",
                    "phone_use_while_driving": "Phone Use",
                    "drowsy_driving": "Drowsy",
                }.get(v_type_raw, v_type_raw)

                cv2.putText(annotated, f"VIOLATION: {v_type_display}",
                            (vx1, max(vy1 - 12, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Find best plate for this violated vehicle
                best_plate = {"plate_text": "UNCLEAR", "confidence": 0.0, "vehicle_class": "unknown"}
                for p in all_plates:
                    px1, py1, px2, py2 = p["bbox"]
                    # Pick plate closest to the violation bbox (by IoU overlap)
                    if (px1 < vx2 and px2 > vx1 and py1 < vy2 and py2 > vy1):
                        if p["confidence"] > best_plate["confidence"]:
                            best_plate = p

                # Draw all plates (green if no violation, red if violated)
                for p in all_plates:
                    x1, y1, x2, y2 = p["bbox"]
                    has_viol = (x1 < vx2 and x2 > vx1 and y1 < vy2 and y2 > vy1)
                    color = (0, 0, 255) if has_viol else (0, 255, 80)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"{p['plate_text']} ({int(p['confidence']*100)}%)"
                    cv2.putText(annotated, label, (x1, max(y1 - 5, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                # Red banner at top
                cv2.rectangle(annotated, (0, 0), (w, 44), (0, 0, 200), -1)
                cv2.putText(annotated, f"GARUDA | VIOLATION: {v_type_display.upper()} | PLATE: {best_plate['plate_text']}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imwrite(annotated_path, annotated)

                record = {
                    "violation_id": vid,
                    "tier": v.tier if hasattr(v, "tier") else 2,
                    "action": "HUMAN_REVIEW",
                    "timestamp": timestamp_now,
                    "camera": {"id": f"JOB-{job_id}", "location": source_name, "coordinates": {}},
                    "vehicle": {
                        "vehicle_class": best_plate.get("vehicle_class", "unknown"),
                        "license_plate": best_plate["plate_text"],
                        "plate_confidence": best_plate["confidence"],
                        "plate_valid": best_plate.get("is_valid", False),
                        "plate_state": best_plate.get("state", "Unknown"),
                    },
                    "violations": [{
                        "type": v_type_display,
                        "confidence": v.confidence,
                        "severity": v.severity,
                        "fine_amount_inr": v.fine_amount if hasattr(v, "fine_amount") else 1000,
                        "bbox": list(map(int, v.bbox)),
                    }],
                    "all_plates_detected": all_plates,
                    "processing": {
                        "inference_device": "CPU",
                        "model": "yolo11n",
                        "ocr_engine": ml_ocr.engine_name,
                        "vehicles_detected": len(vehicles),
                        "persons_detected": len(persons),
                    },
                    "driver_state": {"alerts": [], "total_alerts": 0},
                    "evidence": {
                        "annotated_image": f"/evidence/annotated/{vid}.jpg",
                        "raw_frame": f"/{raw_path}",
                    },
                }
                results.append({
                    "record": record,
                    "plate_text": best_plate["plate_text"],
                    "plate_conf": best_plate["confidence"],
                    "vehicle_class": best_plate.get("vehicle_class", "unknown"),
                    "violation_type": v_type_display,
                    "confidence": v.confidence,
                    "severity": v.severity,
                    "tier": record["tier"],
                    "annotated_img": f"/evidence/annotated/{vid}.jpg",
                    "raw_img": f"/{raw_path}",
                    "status": "pending",
                })

    except Exception as e:
        logger.error("ML inference error in job %s: %s", job_id, e, exc_info=True)

    return results


# ---------------------------------------------------------------------------
# Background job runner
# ---------------------------------------------------------------------------

async def run_job_pipeline(job_id: str, file_bytes: Optional[bytes] = None, filename: str = "upload"):
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

    violations_created = []

    if file_bytes is not None:
        # Decode image
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("Job %s: Failed to decode image bytes", job_id)
            await _update_job(status="Failed", progress=0)
            return

        await _update_job(progress=30)

        # Run ML inference in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        violations_created = await loop.run_in_executor(
            None, _run_ml_on_image, img, job_id, filename
        )

        await _update_job(progress=70)

        # Save all detected violations to DB
        violation_count = 0
        for v_data in violations_created:
            try:
                async with AsyncSessionLocal() as session:
                    await save_violation(session, v_data["record"])
                    if v_data["plate_text"] and v_data["plate_text"] != "UNCLEAR" and v_data["violation_type"] != "Passed / No Violation":
                        await upsert_vehicle(session, v_data["plate_text"], v_data["violation_type"])
                    if v_data["violation_type"] != "Passed / No Violation":
                        violation_count += 1
            except Exception as e:
                logger.error("Failed to save violation record: %s", e)

        elapsed = max(1, int((datetime.utcnow() - start).total_seconds()))
        await _update_job(
            status="Completed",
            progress=100,
            duration=elapsed,
            frames_processed=1,
            violations_found=violation_count,
        )
        logger.info(
            "Job %s completed: %d violations found in %s", job_id, violation_count, filename
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
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    background_tasks.add_task(run_job_pipeline, job_id, None, body.name)
    return JobResponse.model_validate(job)


@router.post("/upload", response_model=JobResponse, status_code=201)
async def upload_job(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    name: str = Form(...),
    source_type: str = Form("Image"),
    file: UploadFile = File(...),
):
    """Upload an image/video file and run ML inference pipeline in the background."""
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
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Read file contents now (before request context ends)
    file_bytes = await file.read()
    filename = file.filename or name

    background_tasks.add_task(run_job_pipeline, job_id, file_bytes, filename)
    logger.info("Job %s queued for ML inference on file: %s (%d bytes)", job_id, filename, len(file_bytes))
    return JobResponse.model_validate(job)


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, db: AsyncSession = Depends(get_db)):
    job = (
        await db.execute(select(JobModel).where(JobModel.id == job_id))
    ).scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobResponse.model_validate(job)


@router.get("/{job_id}/violations", response_model=List[ViolationInJobResponse])
async def get_job_violations(job_id: str, db: AsyncSession = Depends(get_db)):
    """Return all violation records associated with a specific job (camera_id prefix match)."""
    rows = (
        await db.execute(
            select(ViolationModel)
            .where(ViolationModel.camera_id.like(f"JOB-{job_id}%"))
            .order_by(ViolationModel.created_at.desc())
        )
    ).scalars().all()
    return [ViolationInJobResponse.model_validate(r) for r in rows]

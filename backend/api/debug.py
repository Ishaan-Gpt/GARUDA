"""GARUDA API — Debug router.

Endpoints:
  POST /debug/inject-violation   Inject a fake violation (for testing)
  GET  /debug/pipeline-status    Report which ML modules are importable
  GET  /debug/ml-registry        Report current MLRegistry state (loaded/failed)
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import get_db, save_violation
from ..models.schemas import DebugInjectRequest
from ..services.ml_registry import get_ml_registry, WEIGHTS
from .stream import broadcast_violation

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Inject fake violation — for frontend / WebSocket testing
# ---------------------------------------------------------------------------

@router.post("/inject-violation")
async def inject_test_violation(
    body: DebugInjectRequest,
    db: AsyncSession = Depends(get_db),
):
    """Inject a fake violation record and broadcast it to all WS feed clients."""
    vid = f"VIO-TEST-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:4].upper()}"
    record = {
        "violation_id": vid,
        "tier":         body.tier,
        "action":       "HUMAN_REVIEW" if body.tier == 2 else "AUTO_CHALLAN",
        "timestamp":    datetime.utcnow().isoformat() + "Z",
        "camera":  {"id": body.camera_id, "location": body.location, "coordinates": {}},
        "vehicle": {
            "class": "motorcycle", "color": "red",
            "license_plate": body.plate, "plate_confidence": 0.85,
            "plate_valid": True, "plate_state": "Karnataka",
            "repeat_offender": False, "prior_violations": 0,
        },
        "violations": [{
            "type":           body.violation_type,
            "confidence":     body.confidence,
            "severity":       "high",
            "fine_amount_inr": 1000,
            "bbox":           [100, 100, 300, 300],
            "metadata":       {"test": True},
        }],
        "driver_state": {"alerts": [], "total_alerts": 0},
        "evidence":     {"annotated_image": "", "raw_frame": ""},
    }

    await save_violation(db, record)
    await broadcast_violation({
        "event":               "violation_detected",
        "violation_id":        vid,
        "violation_type":      body.violation_type,
        "confidence":          body.confidence,
        "tier":                body.tier,
        "plate":               body.plate,
        "camera_id":           body.camera_id,
        "location":            body.location,
        "timestamp":           record["timestamp"],
        "severity":            "high",
        "annotated_image_url": "",
        "is_test":             True,
    })

    return {"violation_id": vid, "status": "injected", "message": "Test violation created and broadcast"}


# ---------------------------------------------------------------------------
# Pipeline module availability check
# ---------------------------------------------------------------------------

@router.get("/pipeline-status")
async def pipeline_status():
    """Report which ML dependency packages are importable."""
    modules = [
        ("ultralytics",    "YOLO detection"),
        ("mediapipe",      "Driver state (FaceMesh)"),
        ("paddleocr",      "License plate OCR (primary)"),
        ("easyocr",        "License plate OCR (fallback)"),
        ("flwr",           "Federated learning"),
        ("albumentations", "Training augmentation"),
        ("torch",          "PyTorch"),
        ("cv2",            "OpenCV"),
    ]
    status = {}
    for mod, label in modules:
        try:
            __import__(mod)
            status[mod] = {"available": True, "label": label}
        except ImportError:
            status[mod] = {"available": False, "label": label}

    return {"pipeline_modules": status, "timestamp": datetime.utcnow().isoformat()}


# ---------------------------------------------------------------------------
# ML Registry health
# ---------------------------------------------------------------------------

@router.get("/ml-registry")
async def ml_registry_status():
    """Report whether the shared ML singleton is loaded and which weights are found."""
    reg = get_ml_registry()
    weight_status = {key: str(path) + (" ✓" if path.exists() else " ✗ MISSING") for key, path in WEIGHTS.items()}
    return {
        "available": reg.available,
        "error":     reg.error if not reg.available else None,
        "components": {
            "preprocessor":  reg.preprocessor is not None,
            "detector":      reg.detector is not None,
            "ocr":           reg.ocr is not None,
            "classifier":    reg.classifier is not None,
            "driver_state":  reg.driver_state is not None,
        },
        "weights": weight_status,
        "timestamp": datetime.utcnow().isoformat(),
    }

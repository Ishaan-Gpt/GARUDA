"""GARUDA API — Cameras, Vehicles, Analytics, Stream, Debug routers"""
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Set

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import CameraModel, VehicleModel, ViolationModel, get_db

logger = logging.getLogger(__name__)

# Initialize ML pipeline components globally in routers
try:
    import cv2
    import numpy as np
    import random
    import base64
    from ml.pipeline.preprocessor import ImagePreprocessor
    from ml.pipeline.detector import VehicleDetector
    from ml.pipeline.ocr import PlateOCR
    from ml.pipeline.violation_classifier import ViolationClassifier
    from pathlib import Path

    WEIGHTS_DIR = Path(__file__).parent.parent.parent / "ml" / "models" / "weights"
    HELMET_WEIGHTS = str(WEIGHTS_DIR / "helmet_cnn.pt") if (WEIGHTS_DIR / "helmet_cnn.pt").exists() else None
    PLATE_WEIGHTS = str(WEIGHTS_DIR / "plate_yolo.pt") if (WEIGHTS_DIR / "plate_yolo.pt").exists() else None

    logger.info("Initializing real ML pipeline models: helmet_cnn=%s, plate_yolo=%s", HELMET_WEIGHTS, PLATE_WEIGHTS)
    
    ml_preprocessor = ImagePreprocessor()
    ml_detector = VehicleDetector(model_path=None, device="cpu")
    ml_ocr = PlateOCR(plate_detector_weights=PLATE_WEIGHTS)
    ml_classifier = ViolationClassifier(stop_line_y=380, helmet_weights_path=HELMET_WEIGHTS)
    ml_available = True
    logger.info("Real ML pipeline components initialized successfully in _routers.py!")
except Exception as ml_err:
    logger.error("Failed to initialize ML models in routers: %s", ml_err)
    ml_available = False
from ..models.schemas import (
    AnalyticsSummary, CameraConfigUpdate, CameraCreate, CameraResponse,
    HeatmapPoint, HeatmapResponse, TrendPoint, TrendResponse,
    VehicleResponse, ViolationTypeStat, DebugInjectRequest,
)

# ===========================================================================
# CAMERAS ROUTER
# ===========================================================================
router = APIRouter()           # shared import point — each section uses sub-routers


cameras_router = APIRouter(prefix="/cameras")

@cameras_router.get("", response_model=List[CameraResponse])
async def list_cameras(db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(select(CameraModel))).scalars().all()
    return [CameraResponse.model_validate(r) for r in rows]


@cameras_router.post("", response_model=CameraResponse, status_code=201)
async def register_camera(body: CameraCreate, db: AsyncSession = Depends(get_db)):
    existing = (await db.execute(
        select(CameraModel).where(CameraModel.id == body.id)
    )).scalar_one_or_none()

    if existing:
        raise HTTPException(status_code=409, detail=f"Camera {body.id} already exists")

    cam = CameraModel(
        id=body.id, location=body.location,
        lat=body.lat, lon=body.lon,
        stop_line_y=body.stop_line_y,
        status="active",
        last_seen=datetime.utcnow().isoformat(),
        description=body.description,
    )
    db.add(cam)
    await db.commit()
    await db.refresh(cam)
    logger.info("Camera registered: %s @ %s", body.id, body.location)
    return CameraResponse.model_validate(cam)


@cameras_router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(camera_id: str, db: AsyncSession = Depends(get_db)):
    cam = (await db.execute(
        select(CameraModel).where(CameraModel.id == camera_id)
    )).scalar_one_or_none()
    if not cam:
        raise HTTPException(404, f"Camera {camera_id} not found")
    return CameraResponse.model_validate(cam)


@cameras_router.put("/{camera_id}/config", response_model=CameraResponse)
async def update_camera_config(
    camera_id: str, body: CameraConfigUpdate, db: AsyncSession = Depends(get_db)
):
    cam = (await db.execute(
        select(CameraModel).where(CameraModel.id == camera_id)
    )).scalar_one_or_none()
    if not cam:
        raise HTTPException(404, f"Camera {camera_id} not found")

    if body.stop_line_y is not None:
        cam.stop_line_y = body.stop_line_y
    if body.description is not None:
        cam.description = body.description

    await db.commit()
    await db.refresh(cam)
    return CameraResponse.model_validate(cam)


@cameras_router.delete("/{camera_id}", status_code=204)
async def delete_camera(camera_id: str, db: AsyncSession = Depends(get_db)):
    cam = (await db.execute(
        select(CameraModel).where(CameraModel.id == camera_id)
    )).scalar_one_or_none()
    if not cam:
        raise HTTPException(404, f"Camera {camera_id} not found")
    await db.delete(cam)
    await db.commit()


# ===========================================================================
# VEHICLES ROUTER
# ===========================================================================

vehicles_router = APIRouter(prefix="/vehicles")

@vehicles_router.get("/repeat", response_model=List[VehicleResponse])
async def list_repeat_offenders(
    limit: int = Query(50, le=200),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(
        select(VehicleModel)
        .where(VehicleModel.is_repeat_offender == True)
        .order_by(VehicleModel.violation_count.desc())
        .limit(limit)
    )).scalars().all()

    results = []
    for r in rows:
        v = VehicleResponse.model_validate(r)
        v.violations = json.loads(r.violations_json or "[]")
        results.append(v)
    return results


@vehicles_router.get("/{plate}", response_model=VehicleResponse)
async def get_vehicle(plate: str, db: AsyncSession = Depends(get_db)):
    plate_upper = plate.upper().strip()
    row = (await db.execute(
        select(VehicleModel).where(VehicleModel.plate == plate_upper)
    )).scalar_one_or_none()

    if not row:
        raise HTTPException(404, f"No record for plate: {plate_upper}")

    resp = VehicleResponse.model_validate(row)
    resp.violations = json.loads(row.violations_json or "[]")
    return resp


@vehicles_router.delete("/{plate}/clear", status_code=204)
async def clear_vehicle_record(plate: str, db: AsyncSession = Depends(get_db)):
    """Admin endpoint — reset a vehicle's violation history"""
    plate_upper = plate.upper().strip()
    row = (await db.execute(
        select(VehicleModel).where(VehicleModel.plate == plate_upper)
    )).scalar_one_or_none()
    if row:
        await db.delete(row)
        await db.commit()


# ===========================================================================
# ANALYTICS ROUTER
# ===========================================================================

analytics_router = APIRouter(prefix="/analytics")

@analytics_router.get("/summary", response_model=AnalyticsSummary)
async def analytics_summary(db: AsyncSession = Depends(get_db)):
    today = datetime.utcnow().date().isoformat()
    week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()

    total_today = (await db.execute(
        select(func.count()).select_from(ViolationModel)
        .where(ViolationModel.timestamp >= today)
    )).scalar_one()

    total_week = (await db.execute(
        select(func.count()).select_from(ViolationModel)
        .where(ViolationModel.timestamp >= week_ago)
    )).scalar_one()

    auto_count = (await db.execute(
        select(func.count()).select_from(ViolationModel)
        .where(ViolationModel.status == "auto_challan")
    )).scalar_one()

    review_count = (await db.execute(
        select(func.count()).select_from(ViolationModel)
        .where(ViolationModel.status == "pending")
    )).scalar_one()

    # Type breakdown
    type_rows = (await db.execute(
        select(ViolationModel.violation_type, func.count().label("cnt"))
        .group_by(ViolationModel.violation_type)
        .order_by(func.count().desc())
    )).all()

    total_all = sum(r.cnt for r in type_rows) or 1
    breakdown = [
        ViolationTypeStat(
            violation_type=r.violation_type,
            count=r.cnt,
            percentage=round(r.cnt / total_all * 100, 1),
        )
        for r in type_rows
    ]

    top_cam_row = (await db.execute(
        select(ViolationModel.camera_id, func.count().label("cnt"))
        .group_by(ViolationModel.camera_id)
        .order_by(func.count().desc())
        .limit(1)
    )).first()

    return AnalyticsSummary(
        total_today=total_today,
        total_this_week=total_week,
        auto_challan_count=auto_count,
        human_review_count=review_count,
        top_violation_type=breakdown[0].violation_type if breakdown else "N/A",
        top_camera=top_cam_row.camera_id if top_cam_row else "N/A",
        violation_type_breakdown=breakdown,
    )


@analytics_router.get("/trends", response_model=TrendResponse)
async def violation_trends(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
):
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()
    rows = (await db.execute(
        select(
            func.substr(ViolationModel.timestamp, 1, 10).label("date"),
            func.count().label("cnt"),
        )
        .where(ViolationModel.timestamp >= since)
        .group_by(func.substr(ViolationModel.timestamp, 1, 10))
        .order_by(func.substr(ViolationModel.timestamp, 1, 10))
    )).all()

    return TrendResponse(
        period=f"last_{days}_days",
        data_points=[TrendPoint(date=r.date, count=r.cnt) for r in rows],
    )


@analytics_router.get("/heatmap", response_model=HeatmapResponse)
async def violation_heatmap(db: AsyncSession = Depends(get_db)):
    """Return per-camera violation counts with coordinates for Leaflet heatmap"""
    cam_rows = (await db.execute(select(CameraModel))).scalars().all()
    cam_map = {c.id: c for c in cam_rows}

    count_rows = (await db.execute(
        select(ViolationModel.camera_id, func.count().label("cnt"))
        .group_by(ViolationModel.camera_id)
    )).all()

    points = []
    for row in count_rows:
        cam = cam_map.get(row.camera_id)
        if cam and (cam.lat or cam.lon):
            points.append(HeatmapPoint(
                lat=cam.lat, lon=cam.lon,
                intensity=row.cnt,
                camera_id=cam.id,
                location=cam.location,
            ))

    return HeatmapResponse(points=points)


# ===========================================================================
# WEBSOCKET STREAM ROUTER
# ===========================================================================

stream_router = APIRouter()
_ws_connections: Set[WebSocket] = set()


async def broadcast_violation(data: dict) -> None:
    """Called by violations.ingest to push events to all connected clients"""
    dead = set()
    for ws in _ws_connections.copy():
        try:
            await ws.send_json(data)
        except Exception:
            dead.add(ws)
    _ws_connections.difference_update(dead)


@stream_router.websocket("/ws/feed")
async def ws_feed(websocket: WebSocket):
    """
    WebSocket: real-time violation event stream.
    Connect from frontend with:
        new WebSocket("ws://localhost:8000/ws/feed")

    Events emitted:
      - violation_detected  (on every new violation)
      - system_stats        (every 10 seconds)
      - ping                (every 30 seconds keepalive)
    """
    await websocket.accept()
    _ws_connections.add(websocket)
    logger.info("WS client connected | total=%d", len(_ws_connections))

    try:
        await websocket.send_json({"event": "connected", "message": "GARUDA feed live"})
        while True:
            # Wait for client messages (keepalive or ping)
            try:
                msg = await websocket.receive_text()
                if msg == "ping":
                    await websocket.send_json({"event": "pong"})
            except WebSocketDisconnect:
                break
    except Exception as e:
        logger.debug("WS error: %s", e)
    finally:
        _ws_connections.discard(websocket)
        logger.info("WS client disconnected | total=%d", len(_ws_connections))


@stream_router.websocket("/ws/patrol")
async def ws_patrol(websocket: WebSocket):
    """
    WebSocket: real-time police patrol mobile webcam stream.
    Receives base64 frames, decodes and processes them,
    returns annotated overlays and saves violations dynamically.
    """
    await websocket.accept()
    logger.info("Patrol WebSocket client connected")
    
    import random
    import base64
    import cv2
    import numpy as np
    import uuid
    from datetime import datetime
    
    # We yield a new DB session dynamically when saving
    from ..core.database import AsyncSessionLocal, save_violation
    
    try:
        while True:
            # We receive text messages containing JSON
            data = await websocket.receive_json()
            frame_b64 = data.get("frame", "")
            camera_id = data.get("camera_id", "PATROL-EDGE-01")
            location = data.get("location", "Mobile Patrol (Sector 4)")
            
            if not frame_b64:
                continue
                
            # Strip data URI prefix if present
            if "," in frame_b64:
                frame_b64 = frame_b64.split(",", 1)[1]
            
            try:
                img_data = base64.b64decode(frame_b64)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                logger.error("Error decoding patrol frame: %s", e)
                continue
                
            if img is None:
                continue
                
            h, w, c = img.shape
            
            is_violation = False
            violation_info = None
            
            is_simulator = "SIM" in camera_id or "sim" in camera_id or "radar" in location.lower()
            
            if ml_available and not is_simulator:
                try:
                    # Apply CLAHE and Adaptive Gamma correction (low-light enhancement) in real-time
                    processed = ml_preprocessor._enhance_low_light(img)
                    processed = ml_preprocessor._normalize_exposure(processed)
                    
                    detections = ml_detector.detect(processed)
                    vehicles = ml_detector.get_vehicles(detections)
                    persons = ml_detector.get_persons(detections)
                    
                    # Draw normal detections on the enhanced processed frame
                    for det in detections:
                        x1, y1, x2, y2 = map(int, det.bbox)
                        color = (0, 255, 0)
                        label = f"{det.class_name} ({det.confidence*100:.1f}%)"
                        cv2.rectangle(processed, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(processed, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Run checks
                    violations = ml_classifier.check_all(processed, vehicles, persons)
                    if violations:
                        v = violations[0]
                        is_violation = True
                        v_type_raw = v.violation_type.value
                        v_type = {
                            "helmet_non_compliance": "No Helmet",
                            "seatbelt_non_compliance": "Seatbelt",
                            "triple_riding": "Triple Riding",
                            "wrong_side_driving": "Wrong Way",
                            "stop_line_violation": "Stop Line",
                            "red_light_violation": "Red Light",
                            "illegal_parking": "Illegal Parking",
                            "phone_use_while_driving": "Phone Use",
                            "drowsy_driving": "Drowsy"
                        }.get(v_type_raw, v_type_raw)
                        
                        plate_text = "PLATE-UNREAD"
                        plate_conf = 0.0
                        for vehicle in vehicles:
                            vx1, vy1, vx2, vy2 = map(int, vehicle.bbox)
                            h_p, w_p = processed.shape[:2]
                            veh_crop = processed[max(0, vy1):min(h_p, vy2), max(0, vx1):min(w_p, vx2)]
                            if veh_crop.size > 0:
                                ocr_res = ml_ocr.read_plate_from_vehicle(veh_crop)
                                if ocr_res.confidence > plate_conf:
                                    plate_text = ocr_res.formatted_text or "PLATE-UNREAD"
                                    plate_conf = ocr_res.confidence
                        
                        # Draw Red overlay for violation on the enhanced processed frame
                        vx1, vy1, vx2, vy2 = map(int, v.bbox)
                        cv2.rectangle(processed, (vx1, vy1), (vx2, vy2), (0, 0, 255), 3)
                        cv2.putText(processed, f"VIOLATION: {v_type}", (vx1, vy1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        cv2.rectangle(processed, (10, 10), (w - 10, 50), (0, 0, 255), -1)
                        cv2.putText(processed, f"WARNING: {v_type.upper()} DETECTED", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        vid = f"VIO-PATROL-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4].upper()}"
                        
                        record = {
                            "violation_id": vid,
                            "tier": 2,
                            "action": "HUMAN_REVIEW",
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "camera": {"id": camera_id, "location": location, "coordinates": {}},
                            "vehicle": {
                                "vehicle_class": "motorcycle" if "helmet" in v_type_raw or "triple" in v_type_raw else "car",
                                "color": "white",
                                "license_plate": plate_text,
                                "plate_confidence": plate_conf,
                                "plate_valid": True,
                                "plate_state": "Karnataka",
                                "repeat_offender": False,
                                "prior_violations": 0,
                            },
                            "violations": [{
                                "type": v_type,
                                "confidence": v.confidence,
                                "severity": v.severity,
                                "fine_amount_inr": v.fine_amount,
                                "bbox": v.bbox,
                                "metadata": {"source": "patrol"},
                            }],
                            "driver_state": {"alerts": [], "total_alerts": 0},
                            "evidence": {
                                "annotated_image": f"/evidence/annotated/{vid}.jpg", 
                                "raw_frame": f"/evidence/raw/{vid}.jpg"
                            },
                        }
                        
                        import os
                        os.makedirs("evidence/annotated", exist_ok=True)
                        cv2.imwrite(f"evidence/annotated/{vid}.jpg", processed)
                        
                        async with AsyncSessionLocal() as session:
                            await save_violation(session, record)
                            from ..core.database import upsert_vehicle
                            await upsert_vehicle(session, plate_text, v_type)
                        
                        await broadcast_violation({
                            "event": "violation_detected",
                            "violation_id": vid,
                            "violation_type": v_type,
                            "confidence": v.confidence * 100.0,
                            "tier": 2,
                            "plate": plate_text,
                            "camera_id": camera_id,
                            "location": location,
                            "timestamp": record["timestamp"],
                            "severity": v.severity,
                            "annotated_image_url": f"/evidence/annotated/{vid}.jpg",
                        })
                        
                        violation_info = {
                            "violation_id": vid,
                            "type": v_type,
                            "plate": plate_text,
                            "confidence": round(v.confidence * 100.0, 1)
                        }
                    
                    # Point raw img to enhanced processed frame so it is encoded and returned to the phone screen
                    img = processed
                except Exception as run_err:
                    logger.error("Error executing real ML pipeline inside ws_patrol: %s", run_err)
                    is_violation = False
            
            if not is_violation and (is_simulator or not ml_available):
                # When ML is unavailable draw a simple "no ML" watermark but do NOT generate fake violations
                cv2.putText(img, "ML OFFLINE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
                vehicles = []
                persons = []
                detections = []
            
            # Encode annotated image back to base64
            _, buffer = cv2.imencode(".jpg", img)
            annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")
            
            # Send frame response back to patrol client
            await websocket.send_json({
                "frame": annotated_b64,
                "violation": violation_info,
                "detections": {
                    "vehicles": len(vehicles) if 'vehicles' in dir() else 0,
                    "persons": len(persons) if 'persons' in dir() else 0,
                    "total": len(detections) if 'detections' in dir() else 0,
                }
            })
            
    except WebSocketDisconnect:
        logger.info("Patrol WebSocket client disconnected")
    except Exception as e:
        logger.error("Error in patrol WebSocket stream: %s", e)


# ===========================================================================
# DEBUG ROUTER
# ===========================================================================

debug_router = APIRouter()

@debug_router.post("/inject-violation")
async def inject_test_violation(
    body: DebugInjectRequest,
    db: AsyncSession = Depends(get_db),
):
    """Inject a fake violation for frontend/WebSocket testing"""
    from ..core.database import save_violation

    vid = f"VIO-TEST-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:4].upper()}"
    record = {
        "violation_id": vid,
        "tier": body.tier,
        "action": "HUMAN_REVIEW" if body.tier == 2 else "AUTO_CHALLAN",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "camera": {"id": body.camera_id, "location": body.location, "coordinates": {}},
        "vehicle": {
            "class": "motorcycle", "color": "red",
            "license_plate": body.plate, "plate_confidence": 0.85,
            "plate_valid": True, "plate_state": "Karnataka",
            "repeat_offender": False, "prior_violations": 0,
        },
        "violations": [{
            "type": body.violation_type,
            "confidence": body.confidence,
            "severity": "high",
            "fine_amount_inr": 1000,
            "bbox": [100, 100, 300, 300],
            "metadata": {"test": True},
        }],
        "driver_state": {"alerts": [], "total_alerts": 0},
        "evidence": {"annotated_image": "", "raw_frame": ""},
    }

    await save_violation(db, record)
    await broadcast_violation({
        "event": "violation_detected",
        "violation_id": vid,
        "violation_type": body.violation_type,
        "confidence": body.confidence,
        "tier": body.tier,
        "plate": body.plate,
        "camera_id": body.camera_id,
        "location": body.location,
        "timestamp": record["timestamp"],
        "severity": "high",
        "annotated_image_url": "",
        "is_test": True,
    })

    return {"violation_id": vid, "status": "injected", "message": "Test violation created and broadcast"}


@debug_router.get("/pipeline-status")
async def pipeline_status():
    """Check which ML modules are available"""
    status = {}
    modules = [
        ("ultralytics",  "YOLO detection"),
        ("mediapipe",    "Driver state (FaceMesh)"),
        ("paddleocr",    "License plate OCR (primary)"),
        ("easyocr",      "License plate OCR (fallback)"),
        ("flwr",         "Federated learning"),
        ("albumentations","Training augmentation"),
        ("torch",        "PyTorch"),
        ("cv2",          "OpenCV"),
    ]
    for mod, label in modules:
        try:
            __import__(mod)
            status[mod] = {"available": True, "label": label}
        except ImportError:
            status[mod] = {"available": False, "label": label}

    return {"pipeline_modules": status, "timestamp": datetime.utcnow().isoformat()}


# ---------------------------------------------------------------------------
# Expose sub-routers for main.py import
# ---------------------------------------------------------------------------

cameras   = type("cameras",   (), {"router": cameras_router})()
vehicles  = type("vehicles",  (), {"router": vehicles_router})()
analytics = type("analytics", (), {"router": analytics_router})()
stream    = type("stream",    (), {"router": stream_router, "broadcast_violation": broadcast_violation})()
debug     = type("debug",     (), {"router": debug_router})()

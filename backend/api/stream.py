"""GARUDA API — WebSocket stream router.

Endpoints:
  WS /ws/feed    Real-time violation event broadcast to all dashboard clients.
  WS /ws/patrol  Police mobile patrol webcam: receives base64 frames, returns
                 annotated overlays and persists detected violations.

Internal helpers:
  broadcast_violation(data)  — called by other routers to push events.
"""
from __future__ import annotations

import base64
import json
import logging
import uuid
from datetime import datetime
from typing import Set

import cv2
import numpy as np
from fastapi import APIRouter
from fastapi.websockets import WebSocket, WebSocketDisconnect
from sqlalchemy import select

from ..core.database import AsyncSessionLocal, CameraModel, save_violation, upsert_vehicle
from ..services.calibration_service import CalibrationService
from ..services.challan_service import ChallanService, display_name
from ..services.ml_registry import get_ml_registry

logger = logging.getLogger(__name__)

router = APIRouter()

# Active WebSocket connections (dashboard feed)
_ws_connections: Set[WebSocket] = set()


# ---------------------------------------------------------------------------
# Broadcast helper — used by violations.py and debug.py
# ---------------------------------------------------------------------------

async def broadcast_violation(data: dict) -> None:
    """Push a violation event to all connected dashboard feed clients."""
    dead: Set[WebSocket] = set()
    for ws in _ws_connections.copy():
        try:
            await ws.send_json(data)
        except Exception:
            dead.add(ws)
    _ws_connections.difference_update(dead)


# ---------------------------------------------------------------------------
# /ws/feed — dashboard live feed
# ---------------------------------------------------------------------------

@router.websocket("/ws/feed")
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
            try:
                msg = await websocket.receive_text()
                if msg == "ping":
                    await websocket.send_json({"event": "pong"})
            except WebSocketDisconnect:
                break
    except Exception as e:
        logger.debug("WS feed error: %s", e)
    finally:
        _ws_connections.discard(websocket)
        logger.info("WS client disconnected | total=%d", len(_ws_connections))


# ---------------------------------------------------------------------------
# /ws/patrol — mobile patrol webcam stream
# ---------------------------------------------------------------------------

@router.websocket("/ws/patrol")
async def ws_patrol(websocket: WebSocket):
    """
    WebSocket: real-time police patrol mobile webcam stream.
    Receives base64 frames, decodes them, runs the full ML pipeline,
    returns annotated overlays and saves detected violations to DB.

    Message format (client → server):
        { "frame": "<base64-jpeg>", "camera_id": "...", "location": "..." }

    Message format (server → client):
        { "frame": "<base64-jpeg>", "violation": {...}|null, "detections": {...} }
    """
    await websocket.accept()
    logger.info("Patrol WS client connected")

    ml = get_ml_registry()

    try:
        while True:
            data      = await websocket.receive_json()
            frame_b64 = data.get("frame", "")
            camera_id = data.get("camera_id", "PATROL-EDGE-01")
            location  = data.get("location", "Mobile Patrol (Sector 4)")

            if not frame_b64:
                continue

            # Strip data URI prefix if present
            if "," in frame_b64:
                frame_b64 = frame_b64.split(",", 1)[1]

            try:
                img_data = base64.b64decode(frame_b64)
                nparr    = np.frombuffer(img_data, np.uint8)
                img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as dec_err:
                logger.error("Patrol: frame decode error — %s", dec_err)
                continue

            if img is None:
                continue

            h, w, _ = img.shape
            is_simulator = "SIM" in camera_id or "sim" in camera_id

            violation_info: dict | None = None
            vehicles   = []
            persons    = []
            detections = []

            # ----------------------------------------------------------------
            # Real ML inference path
            # ----------------------------------------------------------------
            if ml.available and not is_simulator:
                try:
                    async with AsyncSessionLocal() as cal_session:
                        calib_svc = CalibrationService(cal_session)
                        calibrated = await calib_svc.apply(camera_id, ml.classifier)

                    processed  = ml.preprocessor.preprocess(img, is_video=True)
                    detections = ml.detector.detect(processed)
                    vehicles   = ml.detector.get_vehicles(detections)
                    persons    = ml.detector.get_persons(detections)

                    # Draw all detections on processed frame
                    for det in detections:
                        x1, y1, x2, y2 = map(int, det.bbox)
                        cv2.rectangle(processed, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            processed,
                            f"{det.class_name} ({det.confidence*100:.1f}%)",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                        )

                    phone_dets = [d for d in detections if d.class_name == "cell phone"]
                    violations = ml.classifier.check_all(
                        processed, vehicles, persons,
                        signal_frame=processed,
                        phone_detections=phone_dets,
                    )

                    if violations:
                        v = violations[0]

                        # Run OCR on each vehicle crop; pick highest-confidence result
                        plate_result = None
                        violated_vehicle = None
                        best_plate_conf  = 0.0
                        for vehicle in vehicles:
                            vx1, vy1, vx2, vy2 = map(int, vehicle.bbox)
                            ph, pw = processed.shape[:2]
                            crop = processed[
                                max(0, vy1):min(ph, vy2),
                                max(0, vx1):min(pw, vx2),
                            ]
                            if crop.size > 0:
                                ocr_res = ml.ocr.read_plate_from_vehicle(crop)
                                if ocr_res.confidence > best_plate_conf:
                                    plate_result    = ocr_res
                                    best_plate_conf = ocr_res.confidence
                            if list(map(int, vehicle.bbox)) == list(map(int, v.bbox)):
                                violated_vehicle = vehicle

                        vid = (
                            f"VIO-PATROL-{datetime.now().strftime('%Y%m%d')}"
                            f"-{str(uuid.uuid4())[:4].upper()}"
                        )

                        # Save evidence image
                        import os
                        os.makedirs("evidence/annotated", exist_ok=True)

                        # Red violation overlay
                        vx1, vy1, vx2, vy2 = map(int, v.bbox)
                        cv2.rectangle(processed, (vx1, vy1), (vx2, vy2), (0, 0, 255), 3)
                        vtype_label = display_name(
                            v.violation_type.value
                            if hasattr(v.violation_type, "value")
                            else str(v.violation_type)
                        )
                        cv2.putText(
                            processed,
                            f"VIOLATION: {vtype_label}",
                            (vx1, vy1 - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                        )
                        cv2.rectangle(processed, (10, 10), (w - 10, 50), (0, 0, 255), -1)
                        cv2.putText(
                            processed,
                            f"WARNING: {vtype_label.upper()} DETECTED",
                            (20, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                        )
                        cv2.imwrite(f"evidence/annotated/{vid}.jpg", processed)

                        async with AsyncSessionLocal() as db_session:
                            svc    = ChallanService(db_session)
                            record = await svc.package_and_save(
                                violation_id=vid,
                                camera_id=camera_id,
                                location=location,
                                violations=violations,
                                vehicle=violated_vehicle,
                                plate_result=plate_result,
                                annotated_img_path=f"/evidence/annotated/{vid}.jpg",
                                raw_img_path=f"/evidence/raw/{vid}.jpg",
                                source="patrol",
                                calibrated=calibrated,
                            )

                        if record:
                            await broadcast_violation({
                                "event":              "violation_detected",
                                "violation_id":       vid,
                                "violation_type":     record["violations"][0]["type"] if record["violations"] else "",
                                "confidence":         v.confidence * 100.0,
                                "tier":               record["tier"],
                                "plate":              record["vehicle"]["license_plate"],
                                "camera_id":          camera_id,
                                "location":           location,
                                "timestamp":          record["timestamp"],
                                "severity":           record["violations"][0].get("severity", ""),
                                "annotated_image_url": f"/evidence/annotated/{vid}.jpg",
                            })
                            violation_info = {
                                "violation_id": vid,
                                "type":         record["violations"][0]["type"] if record["violations"] else "",
                                "plate":        record["vehicle"]["license_plate"],
                                "confidence":   round(v.confidence * 100.0, 1),
                            }

                    img = processed

                except Exception as run_err:
                    logger.error("Patrol ML inference error: %s", run_err, exc_info=True)

            # ----------------------------------------------------------------
            # Simulator / ML-offline path — no fake violations
            # ----------------------------------------------------------------
            else:
                cv2.putText(
                    img, "ML OFFLINE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1,
                )

            # Encode annotated frame back to base64 and return
            _, buf = cv2.imencode(".jpg", img)
            annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

            await websocket.send_json({
                "frame":      annotated_b64,
                "violation":  violation_info,
                "detections": {
                    "vehicles": len(vehicles),
                    "persons":  len(persons),
                    "total":    len(detections),
                },
            })

    except WebSocketDisconnect:
        logger.info("Patrol WS client disconnected")
    except Exception as e:
        logger.error("Patrol WS error: %s", e, exc_info=True)

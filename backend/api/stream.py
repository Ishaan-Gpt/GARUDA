"""GARUDA API — WebSocket stream router.

Endpoints:
  WS /ws/feed          Real-time violation event broadcast to all dashboard clients.
  WS /ws/patrol        Police mobile patrol webcam: receives base64 frames, returns
                       annotated overlays and persists detected violations.
  WS /ws/video-render  Pre-render the full accurate ML pipeline (same yolov8m +
                       helmet + seatbelt + signal + OCR stack as the batch job
                       pipeline, every check, every frame) onto a brand-new
                       output video with multi-color violation boxes and
                       persistent track IDs burned in, then hand back a URL to
                       the finished file for normal smooth playback. There is
                       no GPU/NPU on this box, so there's no such thing as free
                       real-time analysis of a full enforcement-grade pipeline —
                       this is the honest version of what most "live" detection
                       demos actually are: analyze once, then play back the
                       result.

Internal helpers:
  broadcast_violation(data)  — called by other routers to push events.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import imageio_ffmpeg
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
                    # No tracker_states here — each patrol frame is checked
                    # independently with no real motion history, so the
                    # motion/duration violations (wrong-side, stop-line,
                    # red-light, illegal-parking) are disabled rather than
                    # guessed from one isolated frame.
                    violations = ml.classifier.check_all(
                        processed, vehicles, persons,
                        signal_frame=processed,
                        phone_detections=phone_dets,
                        enable_motion_violations=False,
                    )

                    if violations:
                        v = violations[0]

                        # Run OCR on each vehicle crop; pick highest-confidence result
                        plate_result = None
                        violated_vehicle = None
                        best_plate_conf  = 0.0
                        best_plate_crop  = None
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
                                    # Detect plate crop region
                                    p_crop = ml.ocr.detect_plate_region(crop, [0, 0, crop.shape[1], crop.shape[0]])
                                    if p_crop is not None and p_crop.size > 0:
                                        best_plate_crop = p_crop
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
                        # Draw license plate crop zoom-in
                        if best_plate_crop is not None and ml.visualizer is not None:
                            processed = ml.visualizer.draw_plate_crop(processed, best_plate_crop, (vx1, vy1 - 70))
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


# ---------------------------------------------------------------------------
# /ws/video-render — pre-render the full accurate pipeline onto an output video
# ---------------------------------------------------------------------------

# Output framerate of the rendered video. Lower = faster total render time
# (fewer frames to run the full pipeline on) at the cost of choppier
# playback. 6fps is a reasonable floor for "looks like a video, not a slide
# show" while keeping total render time roughly 6x the clip's own length on
# this CPU (full pipeline ≈ 700-1000ms/frame measured).
RENDER_OUTPUT_FPS = 6.0

# Hard cap on how much source footage one render will process — without this,
# a long upload could take a very long time on CPU. Trims to the first N
# seconds and says so in the "done" event rather than hanging indefinitely.
MAX_RENDER_SOURCE_SECONDS = 120.0

# One color per violation type so multiple simultaneous violations on the
# same vehicle are visually distinguishable — drawn as concentric outlines,
# all at once, rather than overwriting each other.
VIOLATION_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "No Helmet":       (0, 140, 255),   # orange
    "Seatbelt":        (0, 215, 255),   # gold
    "Triple Riding":   (200, 70, 220),  # magenta
    "Wrong Way":        (0, 0, 255),     # red
    "Stop Line":       (255, 255, 0),   # cyan
    "Red Light":       (0, 0, 200),     # dark red
    "Illegal Parking": (180, 0, 255),   # pink
    "Phone Use":       (255, 140, 0),   # blue
    "Drowsy":          (19, 69, 139),   # brown
}
DEFAULT_BOX_COLOR_BGR = (0, 255, 0)  # green — no violation, just tracked


def _reencode_to_browser_h264(path: str) -> bool:
    """
    cv2.VideoWriter on this machine only reliably produces mp4v (MPEG-4 Part
    2) — the openh264 DLL needed for real H.264 encoding is broken/missing,
    so cv2.VideoWriter_fourcc(*'avc1'/'H264') silently writes a corrupt
    stream despite isOpened() returning True. Browsers can't decode mp4v
    inside an mp4 container, which is why the rendered video showed up
    blank/broken in the player. Re-encode with the static ffmpeg binary
    imageio-ffmpeg bundles (no system ffmpeg install required) to real
    H.264 + faststart so it plays back normally everywhere.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    tmp_out = path + ".h264.mp4"
    cmd = [
        ffmpeg_exe, "-y", "-i", path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
        "-movflags", "+faststart", tmp_out,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0 or not os.path.exists(tmp_out):
            logger.error("ffmpeg re-encode failed for %s: %s", path, result.stderr[-2000:])
            return False
        os.replace(tmp_out, path)
        return True
    except Exception as e:
        logger.error("ffmpeg re-encode exception for %s: %s", path, e)
        return False


def _draw_id_label(frame: np.ndarray, text: str, x1: int, y1: int, color: Tuple[int, int, int]) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ly = max(th + 4, y1)
    cv2.rectangle(frame, (x1, ly - th - 6), (x1 + tw + 8, ly), color, -1)
    text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
    cv2.putText(frame, text, (x1 + 4, ly - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)


def _render_frame_full(
    ml,
    frame: np.ndarray,
    tracker,
    frame_idx: int,
    persist: bool,
    track_seq: Dict[int, int],
    next_seq: List[int],
    reported: Dict[int, Set[str]],
) -> dict:
    """
    One frame of a video-render session, using the SAME full pipeline as the
    batch job path (yolov8m, every violation check, every frame — no
    lightweight swap, no throttling) so this is genuinely "your pipeline",
    not an approximation of it.

    Persistent ByteTrack IDs (track_seq) are remapped to small sequential
    numbers (#1, #2, ...) in order of first appearance, drawn on every
    tracked box. Each vehicle's violations (there can be more than one — a
    two-wheeler can be both un-helmeted AND triple-riding at once) are
    grouped onto that vehicle's box as separate colored concentric outlines,
    all drawn together, instead of one violation silently overwriting
    another's annotation.

    `reported` is the per-session, per-track_id set of violation types
    already cited — a vehicle that's been riding wrong-way for 50 frames
    gets ONE citation the first frame it's confirmed, not 50. The box stays
    visually flagged every frame the condition holds (so it's still obvious
    on screen), but only the first occurrence is persisted as a violation
    record.

    Runs in a thread-pool executor (CPU-bound). Returns the annotated frame
    (written into the output video by the caller) plus any newly-confirmed
    violations for the caller to persist to the DB exactly once.
    """
    processed = ml.preprocessor.preprocess(frame, is_video=True)
    detections = ml.detector.detect_with_tracking(processed, persist=persist)
    tracker.update(detections, frame_idx)

    vehicles = ml.detector.get_vehicles(detections)
    persons = ml.detector.get_persons(detections)
    tracker_states = {s.track_id: s for s in tracker.active_tracks()}

    phone_dets = [d for d in detections if d.class_name == "cell phone"]
    violations = ml.classifier.check_all(
        processed, vehicles, persons,
        signal_frame=processed,
        phone_detections=phone_dets,
        tracker_states=tracker_states,
        enable_motion_violations=True,  # real video, full pipeline — all violations checked
    )

    # Group this frame's violations by the offending vehicle's bbox — a
    # vehicle can have more than one violation type at once.
    by_bbox: Dict[Tuple[int, int, int, int], List[Any]] = defaultdict(list)
    for v in violations:
        by_bbox[tuple(map(int, v.bbox))].append(v)

    annotated = processed.copy()
    # Plain debug view — every tracked detection in one neutral color with
    # its ID/class/confidence, no violation tinting. Mirrors the image
    # pipeline's annotated-vs-demo split (jobs.py _draw_annotated_evidence
    # vs _draw_demo_visualization): "annotated" is the clean evidence view,
    # "demo" is the QA view showing everything the model saw regardless of
    # whether it became a citation.
    demo = processed.copy()
    new_violations: List[dict] = []

    for det in detections:
        if det.track_id is not None and det.track_id not in track_seq:
            track_seq[det.track_id] = next_seq[0]
            next_seq[0] += 1
        seq = track_seq.get(det.track_id, "?")

        x1, y1, x2, y2 = map(int, det.bbox)
        vlist = by_bbox.get((x1, y1, x2, y2), [])
        vtype_names = [
            display_name(v.violation_type.value if hasattr(v.violation_type, "value") else str(v.violation_type))
            for v in vlist
        ]

        if vtype_names:
            for i, vt in enumerate(vtype_names):
                color = VIOLATION_COLORS_BGR.get(vt, (0, 0, 255))
                pad = i * 4
                cv2.rectangle(annotated, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), color, 2)
            label = f"#{seq} " + " + ".join(vtype_names)
            label_color = VIOLATION_COLORS_BGR.get(vtype_names[0], (0, 0, 255))
        else:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), DEFAULT_BOX_COLOR_BGR, 2)
            label = f"#{seq} {det.class_name}"
            label_color = DEFAULT_BOX_COLOR_BGR
        _draw_id_label(annotated, label, x1, y1, label_color)

        demo_color = (255, 255, 0)  # cyan — neutral, same regardless of violation status
        cv2.rectangle(demo, (x1, y1), (x2, y2), demo_color, 1)
        _draw_id_label(demo, f"#{seq} {det.class_name} {det.confidence*100:.0f}%", x1, y1, demo_color)

        # Dedup: only the violation types not already cited for this track_id.
        if det.track_id is not None and vlist:
            already = reported[det.track_id]
            fresh = [v for v, name in zip(vlist, vtype_names) if name not in already]
            if fresh:
                ph, pw = annotated.shape[:2]
                crop = processed[max(0, y1):min(ph, y2), max(0, x1):min(pw, x2)]
                plate_result = ml.ocr.read_plate_from_vehicle(crop) if crop.size > 0 else None
                new_violations.append({
                    "track_id": det.track_id,
                    "seq": seq,
                    "vehicle": det,
                    "violations": fresh,
                    "plate_result": plate_result,
                })
                fresh_names = [
                    display_name(v.violation_type.value if hasattr(v.violation_type, "value") else str(v.violation_type))
                    for v in fresh
                ]
                already.update(fresh_names)

    cv2.putText(demo, f"DETECTIONS: {len(detections)}  VEHICLES: {len(vehicles)}  PERSONS: {len(persons)}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return {"frame": annotated, "demo_frame": demo, "new_violations": new_violations}


@router.websocket("/ws/video-render")
async def ws_video_render(websocket: WebSocket):
    """
    WebSocket: pre-render the full accurate pipeline onto an output video.

    Handshake (client → server):
      1. One text frame:   {"camera_id": "...", "location": "..."}
      2. One binary frame: the raw video file bytes.

    Streamed messages (server → client):
      {"event": "progress", "percent": 0-100, "frames_processed": N,
       "total_frames": N, "eta_seconds": N}
      {"event": "done", "video_url": "/evidence/video/{id}_annotated.mp4",
       "demo_video_url": "/evidence/video/{id}_demo.mp4",
       "violations": [...one entry per newly-confirmed citation...],
       "truncated": bool}
      {"event": "error", "message": "..."}

    Why this shape: there is no GPU/NPU here, and the full enforcement-grade
    pipeline (yolov8m + helmet + seatbelt + signal + OCR, every check, every
    frame) runs at roughly 1 frame/sec on CPU — too slow to overlay on a
    freely-playing video without the boxes drifting behind within seconds.
    Analyzing once and handing back a normal video file is the only way to
    get genuinely smooth, frame-accurate playback without sacrificing the
    real trained pipeline for a lighter approximation.
    """
    await websocket.accept()
    logger.info("Video-render WS client connected")

    ml = get_ml_registry()
    tmp_path: Optional[str] = None
    cap = None
    writer = None
    demo_writer = None

    try:
        header_raw = await websocket.receive_text()
        try:
            header = json.loads(header_raw)
        except json.JSONDecodeError:
            header = {}
        camera_id = header.get("camera_id", "VIDEO-RENDER-01")
        location = header.get("location", "Pre-rendered Video Upload")

        if not ml.available:
            await websocket.send_json({"event": "error", "message": "ML pipeline unavailable"})
            return

        # Receive the video as a stream of binary chunks (client-controlled
        # size, e.g. 512KB) terminated by a text {"event":"upload_complete"}
        # message, instead of one giant binary frame — a single >16MB
        # message hits the websocket protocol's default max message size
        # and gets dropped with a 1009 "message too big" close, which is
        # exactly what happened on a real ~57MB demo clip. Chunking removes
        # that ceiling entirely regardless of how large the source video is.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp_path = tmp.name
            while True:
                msg = await websocket.receive()
                if msg.get("bytes") is not None:
                    tmp.write(msg["bytes"])
                elif msg.get("text") is not None:
                    try:
                        ctrl = json.loads(msg["text"])
                    except json.JSONDecodeError:
                        ctrl = {}
                    if ctrl.get("event") == "upload_complete":
                        break
                elif msg.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect()

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            await websocket.send_json({"event": "error", "message": "Could not open video file"})
            return

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_fps = min(src_fps, RENDER_OUTPUT_FPS)
        sample_interval = max(1, round(src_fps / output_fps))
        max_src_frames = min(total_src_frames, int(MAX_RENDER_SOURCE_SECONDS * src_fps)) if total_src_frames else 0
        truncated = bool(total_src_frames and max_src_frames < total_src_frames)

        async with AsyncSessionLocal() as cal_session:
            calib_svc = CalibrationService(cal_session)
            calibrated = await calib_svc.apply(camera_id, ml.classifier)

        render_id = f"RENDER-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:6].upper()}"
        out_dir = "evidence/video"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/{render_id}_annotated.mp4"
        demo_out_path = f"{out_dir}/{render_id}_demo.mp4"
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
        demo_writer = cv2.VideoWriter(demo_out_path, cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))

        from ml.pipeline.tracker import VehicleTracker
        tracker = VehicleTracker(stop_line_y=ml.classifier.stop_line_y)
        track_seq: Dict[int, int] = {}
        next_seq = [1]
        reported: Dict[int, Set[str]] = defaultdict(set)
        violation_summaries: List[dict] = []

        loop = asyncio.get_event_loop()
        frame_idx = 0
        processed_count = 0
        first_sampled = True
        t_render_start = time.perf_counter()
        expected_total = (max_src_frames // sample_interval) if max_src_frames else 0

        while True:
            if max_src_frames and frame_idx >= max_src_frames:
                break
            ret, frame = await loop.run_in_executor(None, cap.read)
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                result = await loop.run_in_executor(
                    None, _render_frame_full, ml, frame, tracker, frame_idx, not first_sampled,
                    track_seq, next_seq, reported,
                )
                first_sampled = False
                processed_count += 1
                writer.write(result["frame"])
                demo_writer.write(result["demo_frame"])

                for nv in result["new_violations"]:
                    vid = f"VIO-RENDER-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4].upper()}"
                    async with AsyncSessionLocal() as db_session:
                        svc = ChallanService(db_session)
                        record = await svc.package_and_save(
                            violation_id=vid,
                            camera_id=camera_id,
                            location=location,
                            violations=nv["violations"],
                            vehicle=nv["vehicle"],
                            plate_result=nv["plate_result"],
                            annotated_img_path=f"/{out_path}",
                            raw_img_path=f"/{out_path}",
                            source="video-render",
                            calibrated=calibrated,
                        )
                    if record:
                        summary = {
                            "vehicle_id": nv["seq"],
                            "types": [vv["type"] for vv in record["violations"]],
                            "plate": record["vehicle"]["license_plate"],
                            "confidence": round(max((vv["confidence"] for vv in record["violations"]), default=0.0) * 100.0, 1),
                            "frame_idx": frame_idx,
                        }
                        violation_summaries.append(summary)
                        await broadcast_violation({
                            "event": "violation_detected",
                            "violation_id": vid,
                            "violation_type": record["violations"][0]["type"] if record["violations"] else "",
                            "confidence": summary["confidence"],
                            "tier": record["tier"],
                            "plate": record["vehicle"]["license_plate"],
                            "camera_id": camera_id,
                            "location": location,
                            "timestamp": record["timestamp"],
                            "severity": record["violations"][0].get("severity", "") if record["violations"] else "",
                            "annotated_image_url": f"/{out_path}",
                        })

                elapsed = time.perf_counter() - t_render_start
                rate = processed_count / elapsed if elapsed > 0 else 0
                remaining = max(0, expected_total - processed_count)
                eta = round(remaining / rate, 1) if rate > 0 else None
                percent = round(processed_count / expected_total * 100, 1) if expected_total else 0
                await websocket.send_json({
                    "event": "progress",
                    "percent": min(percent, 99.9),
                    "frames_processed": processed_count,
                    "total_frames": expected_total,
                    "eta_seconds": eta,
                })

            frame_idx += 1

        writer.release()
        writer = None
        demo_writer.release()
        demo_writer = None
        cap.release()
        cap = None

        await websocket.send_json({"event": "progress", "percent": 99.9, "frames_processed": processed_count, "total_frames": expected_total, "eta_seconds": None, "stage": "encoding"})
        reencoded = await loop.run_in_executor(None, _reencode_to_browser_h264, out_path)
        demo_reencoded = await loop.run_in_executor(None, _reencode_to_browser_h264, demo_out_path)
        if not reencoded or not demo_reencoded:
            logger.warning("Serving raw mp4v output for %s — browser playback may fail", render_id)

        await websocket.send_json({
            "event": "done",
            "video_url": f"/{out_path}",
            "demo_video_url": f"/{demo_out_path}",
            "violations": violation_summaries,
            "truncated": truncated,
        })

    except WebSocketDisconnect:
        logger.info("Video-render WS client disconnected")
    except Exception as e:
        logger.error("Video-render WS error: %s", e, exc_info=True)
        try:
            await websocket.send_json({"event": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if writer is not None:
            writer.release()
        if demo_writer is not None:
            demo_writer.release()
        if cap is not None:
            cap.release()
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

"""
GARUDA — End-to-End Demo Pipeline
====================================
Tests the full ML inference chain on a static image, video file, or webcam.

Usage:
    # Single image
    python ml/demo_pipeline.py --input sample.jpg

    # Video file
    python ml/demo_pipeline.py --input traffic.mp4 --video

    # Webcam
    python ml/demo_pipeline.py --webcam

    # With driver state analysis
    python ml/demo_pipeline.py --input sample.jpg --driver-state

    # Verbose output + save results
    python ml/demo_pipeline.py --input sample.jpg --verbose --output evidence/
"""
from __future__ import annotations

import argparse
import sys
import time
import json
import logging
from pathlib import Path

import cv2
import numpy as np

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.pipeline.preprocessor import ImagePreprocessor
from ml.pipeline.detector import VehicleDetector
from ml.pipeline.tracker import VehicleTracker
from ml.pipeline.ocr import PlateOCR
from ml.pipeline.violation_classifier import ViolationClassifier
from ml.pipeline.confidence_router import ConfidenceRouter, RepeatOffenderDB
from ml.pipeline.driver_state import DriverStateDetector
from ml.utils.evidence import EvidencePackager
from ml.utils.visualizer import FrameVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("garuda.demo")

# ---------------------------------------------------------------------------
# Trained weight auto-discovery (see ml/training/ + GARUDA_Train_Colab.ipynb)
# ---------------------------------------------------------------------------

WEIGHTS_DIR = Path(__file__).parent / "models" / "weights"
HELMET_WEIGHTS = WEIGHTS_DIR / "helmet_cnn.pt"
# Stage-1 plate detector: Koushi (6.2MB) — best spatial coverage
# Stage-2 plate detector: YasirFaiz (6.2MB) — confirms crop (auto-loaded by PlateOCR)
# Legacy fallback: 52MB moin model passed if Koushi missing
PLATE_WEIGHTS = WEIGHTS_DIR / "plate_koushi.pt"
PLATE_WEIGHTS_FALLBACK = WEIGHTS_DIR / "plate_yolov8_moin.pt"


def _resolve_weight(path: Path, cli_override: str | None, fallback: Path | None = None) -> str | None:
    if cli_override:
        return cli_override
    if path.exists():
        return str(path)
    if fallback and fallback.exists():
        logger.info("Primary weights not found, using fallback: %s", fallback)
        return str(fallback)
    return None


# ---------------------------------------------------------------------------
# Backend push (ML pipeline -> FastAPI -> dashboard)
# ---------------------------------------------------------------------------

def _build_ingest_payload(decision, package: dict, plate_info: dict) -> dict:
    """Map a routed decision + evidence package onto backend's ViolationIngestRequest schema."""
    record = package["record"]
    vehicle = dict(record["vehicle"])
    vehicle["vehicle_class"] = vehicle.pop("class", "unknown")
    vehicle.pop("plate_raw", None)

    return {
        "violation_id": decision.violation_id,
        "tier": decision.tier,
        "action": decision.action,
        "timestamp": decision.timestamp,
        "camera": record["camera"],
        "vehicle": vehicle,
        "violations": record["violations"],
        "driver_state": record["driver_state"],
        "evidence": record["evidence"],
        "processing": record["processing"],
        "plate": plate_info or None,
        "escalation_reason": decision.escalation_reason,
    }


def _push_to_backend(decision, package: dict, plate_info: dict, backend_url: str) -> None:
    """POST a violation to the live backend so the dashboard reflects real ML output."""
    try:
        import httpx
        payload = _build_ingest_payload(decision, package, plate_info)
        resp = httpx.post(f"{backend_url}/api/v1/violations/ingest", json=payload, timeout=3.0)
        resp.raise_for_status()
        logger.info("      Pushed to backend: %s", decision.violation_id)
    except Exception as e:
        logger.warning("      Could not push to backend (%s): %s", backend_url, e)

# ---------------------------------------------------------------------------
# Camera metadata for demo
# ---------------------------------------------------------------------------

DEMO_CAMERA = {
    "camera_id":   "BLR-CAM-DEMO-001",
    "location":    "MG Road & Brigade Road Intersection",
    "coordinates": {"lat": 12.9753, "lon": 77.6069},
}


# ---------------------------------------------------------------------------
# Image pipeline
# ---------------------------------------------------------------------------

def run_image(args) -> None:
    logger.info("=" * 64)
    logger.info("GARUDA — Image Pipeline | %s", args.input)
    logger.info("=" * 64)

    # --- Init modules ---
    helmet_weights = _resolve_weight(HELMET_WEIGHTS, args.helmet_weights)
    plate_weights  = _resolve_weight(PLATE_WEIGHTS, args.plate_weights, PLATE_WEIGHTS_FALLBACK)
    if helmet_weights:
        logger.info("Using trained helmet classifier: %s", helmet_weights)
    if plate_weights:
        logger.info("Using trained plate detector: %s", plate_weights)

    preprocessor = ImagePreprocessor()
    detector     = VehicleDetector(device="cpu")
    ocr          = PlateOCR(plate_detector_weights=plate_weights)
    classifier   = ViolationClassifier(stop_line_y=args.stop_line_y, helmet_weights_path=helmet_weights)
    repeat_db    = RepeatOffenderDB()
    router       = ConfidenceRouter(repeat_db)
    packager     = EvidencePackager(output_dir=args.output)
    visualizer   = FrameVisualizer()

    driver_det = DriverStateDetector() if args.driver_state else None

    # --- Load image ---
    frame = cv2.imread(args.input)
    if frame is None:
        logger.error("Cannot read image: %s", args.input)
        sys.exit(1)

    h, w = frame.shape[:2]
    logger.info("Image: %dx%d", w, h)

    t0 = time.perf_counter()

    # === STEP 1: Preprocess ===
    logger.info("[1/6] Preprocessing…")
    processed = preprocessor.preprocess(frame)

    # === STEP 2: Detect ===
    logger.info("[2/6] Detecting vehicles and persons…")
    detections = detector.detect(processed)
    vehicles   = detector.get_vehicles(detections)
    persons    = detector.get_persons(detections)
    phones     = detector.get_phones(detections)
    logger.info("      %d vehicles, %d persons, %d phones detected", len(vehicles), len(persons), len(phones))

    # === STEP 3: Driver state ===
    driver_alerts = []
    if driver_det:
        logger.info("[3/6] Analysing driver state…")
        driver_alerts = driver_det.analyze_frame(processed)
        if driver_alerts:
            for a in driver_alerts:
                logger.warning("      ⚠  %s (conf=%.2f)", a.alert_type, a.confidence)

    # === STEP 4: Violations ===
    logger.info("[4/6] Classifying violations…")
    all_violations = classifier.check_all(processed, vehicles, persons, phone_detections=phones)
    logger.info("      %d violations found", len(all_violations))
    for v in all_violations:
        logger.info("      → %s | conf=%.2f | severity=%s", v.violation_type.value, v.confidence, v.severity)

    # === STEP 5: OCR ===
    logger.info("[5/6] Reading license plates…")
    plate_info = {"formatted_text": "", "confidence": 0.0, "is_valid": False, "state": "Unknown"}
    for vehicle in vehicles:
        plate_region = ocr.detect_plate_region(processed, vehicle.bbox)
        if plate_region is not None and plate_region.size > 0:
            result = ocr.read_plate(plate_region)
            if result.confidence > plate_info.get("confidence", 0):
                plate_info = result.to_dict()
    logger.info(
        "      Plate: %s (conf=%.0f%%, valid=%s)",
        plate_info.get("formatted_text") or "UNCLEAR",
        plate_info.get("confidence", 0) * 100,
        plate_info.get("is_valid"),
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # === STEP 6: Route + Package ===
    logger.info("[6/6] Routing decisions…")
    decisions = router.route_batch(all_violations, plate_info, DEMO_CAMERA)

    results_summary = []
    for decision in decisions:
        tier_label = {1: "✅ AUTO-CHALLAN", 2: "⚠  HUMAN REVIEW", 3: "📝 LOGGED"}.get(decision.tier, "?")
        logger.info("      %s | %s | ID: %s", tier_label, decision.action, decision.violation_id)

        if decision.tier == 2 and args.verbose:
            print("\n" + router.build_whatsapp_alert(decision) + "\n")

        # Generate evidence (reuse the router's violation_id so DB + files agree)
        package = packager.create_package(
            frame=frame,
            violations=[decision.violation.to_dict()],
            plate_info=plate_info,
            camera_info=DEMO_CAMERA,
            driver_alerts=[a.to_dict() for a in driver_alerts],
            processing_info={"time_ms": round(elapsed_ms, 1), "model": "yolov8m"},
            violation_id=decision.violation_id,
        )
        logger.info("      Evidence: %s", package["annotated_image_path"])
        results_summary.append(package)

        if args.backend_url:
            _push_to_backend(decision, package, plate_info, args.backend_url)

    # --- Display ---
    display = frame.copy()
    display = visualizer.draw_detections(display, [d.to_dict() for d in detections])
    if all_violations:
        display = visualizer.draw_violations(display, [v.to_dict() for v in all_violations])
    visualizer.draw_stop_line(display, args.stop_line_y)

    if driver_alerts:
        visualizer.draw_driver_alert(display, driver_alerts[0].alert_type)

    if decisions:
        d = decisions[0]
        visualizer.draw_tier_badge(display, d.tier, d.action, (10, 90))

    plate_text = plate_info.get("formatted_text") or "UNCLEAR"
    visualizer.draw_plate_result(
        display, plate_text,
        plate_info.get("confidence", 0),
        plate_info.get("is_valid", False),
    )

    logger.info("=" * 64)
    logger.info("DONE in %.0f ms | Violations: %d | Plates read: %s",
                elapsed_ms, len(all_violations), plate_text)
    logger.info("=" * 64)

    if args.show:
        cv2.imshow("GARUDA — Detection Result", display)
        logger.info("Press any key to exit…")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save display image
    out_path = Path(args.output) / "demo_result.jpg"
    cv2.imwrite(str(out_path), display)
    logger.info("Result saved: %s", out_path)


# ---------------------------------------------------------------------------
# Video / webcam pipeline
# ---------------------------------------------------------------------------

def run_video(args) -> None:
    source = 0 if args.webcam else args.input
    logger.info("GARUDA — Video Pipeline | source=%s", "webcam" if args.webcam else args.input)

    helmet_weights = _resolve_weight(HELMET_WEIGHTS, args.helmet_weights)
    plate_weights  = _resolve_weight(PLATE_WEIGHTS, args.plate_weights, PLATE_WEIGHTS_FALLBACK)

    preprocessor  = ImagePreprocessor()
    detector      = VehicleDetector(device="cpu")
    tracker       = VehicleTracker(stop_line_y=args.stop_line_y)
    ocr           = PlateOCR(plate_detector_weights=plate_weights)
    classifier    = ViolationClassifier(stop_line_y=args.stop_line_y, helmet_weights_path=helmet_weights)
    repeat_db     = RepeatOffenderDB()
    router        = ConfidenceRouter(repeat_db)
    packager      = EvidencePackager(output_dir=args.output)
    visualizer    = FrameVisualizer()
    driver_det    = DriverStateDetector() if args.driver_state else None

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Cannot open video source: %s", source)
        sys.exit(1)

    fps_target     = int(cap.get(cv2.CAP_PROP_FPS) or 30)
    frame_idx      = 0
    violations_total = 0
    fps_timer      = time.time()
    fps_count      = 0
    display_fps    = 0.0
    tier1_count    = 0
    tier2_count    = 0

    logger.info("Processing video… press Q to stop")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        fps_count += 1

        if time.time() - fps_timer >= 1.0:
            display_fps = fps_count
            fps_count   = 0
            fps_timer   = time.time()

        # Every 2nd frame for performance on CPU
        if frame_idx % 2 != 0:
            continue

        # Detect + track
        processed  = preprocessor.preprocess(frame, enhance=False, is_video=True)
        detections = detector.detect_with_tracking(processed)
        vehicles   = detector.get_vehicles(detections)
        persons    = detector.get_persons(detections)
        phones     = detector.get_phones(detections)
        tracker.update(detections, frame_idx)

        # Violations (pass full frame so MLSignalStateDetector can scan top 40%)
        track_states = {s.track_id: s for s in tracker.active_tracks()}
        violations   = classifier.check_all(processed, vehicles, persons,
                                             signal_frame=processed,
                                             tracker_states=track_states,
                                             phone_detections=phones)

        # OCR — read plate of each vehicle, keep best result this frame
        plate_info = {"formatted_text": "", "confidence": 0.0, "is_valid": False}
        for vehicle in vehicles:
            plate_region = ocr.detect_plate_region(processed, vehicle.bbox)
            if plate_region is not None and plate_region.size > 0:
                result = ocr.read_plate(plate_region)
                if result.confidence > plate_info.get("confidence", 0):
                    plate_info = result.to_dict()

        for v in violations:
            violations_total += 1
            decisions = router.route_batch([v], plate_info, DEMO_CAMERA)
            for d in decisions:
                if d.tier == 1:
                    tier1_count += 1
                elif d.tier == 2:
                    tier2_count += 1

        # Driver state
        driver_alerts = []
        if driver_det:
            driver_alerts = driver_det.analyze_frame(processed)

        # Visualise
        display = frame.copy()
        display = visualizer.draw_detections(display, [d.to_dict() for d in detections])
        if violations:
            display = visualizer.draw_violations(display, [v.to_dict() for v in violations])
        for alert in driver_alerts:
            visualizer.draw_driver_alert(display, alert.alert_type)
        visualizer.draw_stop_line(display, args.stop_line_y)
        visualizer.draw_hud(display, {
            "fps":              display_fps,
            "active_tracks":   len(tracker.active_tracks()),
            "violations_today": violations_total,
            "tier1":           tier1_count,
            "tier2":           tier2_count,
        })

        cv2.imshow("GARUDA — Live Detection (Q to quit)", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info(
        "Video complete | frames=%d | violations=%d | tier1=%d | tier2=%d",
        frame_idx, violations_total, tier1_count, tier2_count,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GARUDA Traffic Violation Detection — Demo Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input",  "-i", help="Path to image or video file")
    src.add_argument("--webcam", "-w", action="store_true", help="Use webcam (device 0)")

    parser.add_argument("--video",        "-v", action="store_true", help="Treat --input as video")
    parser.add_argument("--output",       "-o", default="evidence",  help="Output directory")
    parser.add_argument("--show",         "-s", action="store_true", default=False,
                        help="Show result in OpenCV window")
    parser.add_argument("--stop-line-y",        type=int, default=380,
                        help="Y-coordinate of stop line in pixels (default: 380)")
    parser.add_argument("--driver-state",        action="store_true",
                        help="Enable driver drowsiness + phone detection")
    parser.add_argument("--verbose",             action="store_true",
                        help="Print WhatsApp alert previews to console")
    parser.add_argument("--helmet-weights",      default=None,
                        help=f"Path to trained helmet_cnn.pt (default: auto-detect {HELMET_WEIGHTS})")
    parser.add_argument("--plate-weights",       default=None,
                        help=f"Path to trained plate_yolo.pt (default: auto-detect {PLATE_WEIGHTS})")
    parser.add_argument("--backend-url",         default=None,
                        help="If set (e.g. http://localhost:8000), POST violations to the live backend "
                             "so the dashboard reflects real detections instead of /debug/inject-violation fakes")

    args = parser.parse_args()

    if args.webcam or args.video:
        run_video(args)
    else:
        run_image(args)


if __name__ == "__main__":
    main()

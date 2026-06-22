"""One-off script: run the full GARUDA pipeline on a single image and save
an annotated frame + cropped license plate (if any vehicle's plate region
is found) + a printed violation report. Not part of the permanent pipeline.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))

from ml.pipeline.preprocessor import ImagePreprocessor
from ml.pipeline.detector import VehicleDetector
from ml.pipeline.ocr import PlateOCR
from ml.pipeline.violation_classifier import ViolationClassifier
from ml.pipeline.confidence_router import ConfidenceRouter, RepeatOffenderDB
from ml.utils.visualizer import FrameVisualizer

WEIGHTS_DIR = Path("ml/models/weights")


def run(image_path: str, out_prefix: str, out_dir: str = "result") -> None:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    preprocessor = ImagePreprocessor()
    detector = VehicleDetector(device="cpu")
    ocr = PlateOCR(plate_detector_weights=str(WEIGHTS_DIR / "plate_koushi.pt"))
    classifier = ViolationClassifier(stop_line_y=380)
    router = ConfidenceRouter(RepeatOffenderDB())
    visualizer = FrameVisualizer()

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: cannot read {image_path}")
        return

    h, w = frame.shape[:2]
    processed = preprocessor.preprocess(frame)

    detections = detector.detect(processed)
    vehicles = detector.get_vehicles(detections)
    persons = detector.get_persons(detections)
    phones = detector.get_phones(detections)

    all_violations = classifier.check_all(processed, vehicles, persons, phone_detections=phones)

    # Best plate across all vehicles + save the crop of whichever vehicle gave it
    best_plate_info = {"formatted_text": "", "confidence": 0.0, "is_valid": False}
    best_plate_crop = None
    for vehicle in vehicles:
        x1, y1, x2, y2 = map(int, vehicle.bbox)
        h_img, w_img = processed.shape[:2]
        veh_crop = processed[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]
        if veh_crop.size > 0:
            result = ocr.read_plate_from_vehicle(veh_crop)
            if result.confidence > best_plate_info.get("confidence", 0):
                best_plate_info = result.to_dict()
                best_plate_crop = ocr.detect_plate_region(processed, vehicle.bbox)

    decisions = router.route_batch(all_violations, best_plate_info, {
        "camera_id": "BLR-CAM-DEMO-001",
        "location": "MG Road & Brigade Road Intersection",
        "coordinates": {"lat": 12.9753, "lon": 77.6069},
    })

    # Annotated combined frame
    display = frame.copy()
    display = visualizer.draw_detections(display, [d.to_dict() for d in detections])
    if all_violations:
        display = visualizer.draw_violations(display, [v.to_dict() for v in all_violations])
    visualizer.draw_stop_line(display, 380)
    plate_text = best_plate_info.get("formatted_text") or "UNCLEAR"
    visualizer.draw_plate_result(display, plate_text, best_plate_info.get("confidence", 0), best_plate_info.get("is_valid", False))
    cv2.imwrite(str(out / f"{out_prefix}_annotated.jpg"), display)

    if best_plate_crop is not None:
        cv2.imwrite(str(out / f"{out_prefix}_plate_crop.jpg"), best_plate_crop)

    # Structured report
    report = {
        "image": image_path,
        "size": f"{w}x{h}",
        "vehicles_detected": len(vehicles),
        "persons_detected": len(persons),
        "phones_detected": len(phones),
        "violations": [],
        "plate": best_plate_info,
        "plate_crop_saved": best_plate_crop is not None,
    }
    for v, d in zip(all_violations, decisions):
        report["violations"].append({
            "type": v.violation_type.value,
            "confidence": v.confidence,
            "severity": v.severity,
            "fine_inr": v.fine_amount,
            "tier": d.tier,
            "action": d.action,
            "violation_id": d.violation_id,
        })

    with open(out / f"{out_prefix}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])

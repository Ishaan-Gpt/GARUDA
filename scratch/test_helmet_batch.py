import sys
import json
import time
import uuid
from pathlib import Path
import cv2

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.pipeline.preprocessor import ImagePreprocessor
from ml.pipeline.detector import VehicleDetector
from ml.pipeline.ocr import PlateOCR
from ml.pipeline.violation_classifier import ViolationClassifier
from ml.pipeline.confidence_router import ConfidenceRouter, RepeatOffenderDB
from ml.utils.evidence import EvidencePackager
from ml.utils.visualizer import FrameVisualizer

DEMO_CAMERA = {
    "camera_id": "BLR-CAM-DEMO-001",
    "location": "MG Road & Brigade Road Intersection",
    "coordinates": {"lat": 12.9753, "lon": 77.6069},
}

def main():
    test_dir = Path("test/helmet")
    images = sorted(list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.jpeg")))
    
    if not images:
        print(f"No images found in {test_dir}")
        return

    print(f"Found {len(images)} images in {test_dir}. Running batch test with evidence packaging...")
    
    # Initialize pipeline
    preprocessor = ImagePreprocessor()
    detector = VehicleDetector(device="cpu")
    ocr = PlateOCR(plate_detector_weights="ml/models/weights/plate_koushi.pt")
    classifier = ViolationClassifier(stop_line_y=380)
    router = ConfidenceRouter(RepeatOffenderDB())
    packager = EvidencePackager(output_dir="evidence")
    visualizer = FrameVisualizer()

    results = []
    
    for img_path in images:
        print(f"\nProcessing {img_path.name}...")
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Failed to read {img_path}")
            continue

        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        processed = preprocessor.preprocess(frame)
        
        detections = detector.detect(processed)
        vehicles = detector.get_vehicles(detections)
        persons = detector.get_persons(detections)
        phones = detector.get_phones(detections)

        violations = classifier.check_all(processed, vehicles, persons, phone_detections=phones)
        
        # Best plate across all vehicles + OCR for each vehicle
        plate_info = {"formatted_text": "", "confidence": 0.0, "is_valid": False, "state": "Unknown"}
        all_plates = []
        for vehicle in vehicles:
            plate_region = ocr.detect_plate_region(processed, vehicle.bbox)
            if plate_region is not None and plate_region.size > 0:
                plate_result = ocr.read_plate(plate_region)
                vehicle.plate_text = plate_result.formatted_text or "UNCLEAR"
                vehicle.plate_conf = plate_result.confidence
                if plate_result.confidence > plate_info.get("confidence", 0):
                    plate_info = plate_result.to_dict()
                all_plates.append({
                    "plate_text": vehicle.plate_text,
                    "confidence": round(vehicle.plate_conf, 3),
                    "bbox": list(map(int, vehicle.bbox)),
                    "is_valid": plate_result.is_valid
                })

        elapsed_ms = (time.perf_counter() - t0) * 1000

        decisions = router.route_batch(violations, plate_info, DEMO_CAMERA)

        # Associate plate with each specific violation
        for v in violations:
            v_plate = "UNCLEAR"
            for vehicle in vehicles:
                if [round(x, 2) for x in vehicle.bbox] == [round(x, 2) for x in v.bbox]:
                    v_plate = getattr(vehicle, "plate_text", "UNCLEAR")
                    break
            v.plate_text = v_plate

        # Determine collapsed routing tier/action
        if not decisions:
            tier, action = 1, "PASSED"
        elif all(d.tier == 1 for d in decisions):
            tier, action = 1, "AUTO_CHALLAN"
        else:
            tier, action = 2, "HUMAN_REVIEW"

        # Generate single violation ID for the whole frame
        vid = decisions[0].violation_id if decisions else f"VIO-BLR-{time.strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:6].upper()}"

        # 1. Save general demo annotated image (will draw plates on vehicles)
        display = frame.copy()
        display = visualizer.draw_detections(display, [d.to_dict() for d in detections])
        if violations:
            display = visualizer.draw_violations(display, [v.to_dict() for v in violations])
        visualizer.draw_stop_line(display, 380)
        visualizer.draw_tier_badge(display, tier, action, (10, 90))
        plate_text_display = plate_info.get("formatted_text") or "UNCLEAR"
        visualizer.draw_plate_result(
            display, plate_text_display,
            plate_info.get("confidence", 0),
            plate_info.get("is_valid", False),
        )
        
        demo_out_path = Path("evidence") / f"demo_{img_path.stem}.jpg"
        cv2.imwrite(str(demo_out_path), display)
        print(f"  Demo annotation saved to: {demo_out_path}")

        # 2. Package collapsed evidence record
        package = packager.create_package(
            frame=frame,
            violations=[d.violation.to_dict() for d in decisions],
            plate_info=plate_info,
            camera_info=DEMO_CAMERA,
            processing_info={"time_ms": round(elapsed_ms, 1), "model": "yolov8m"},
            violation_id=vid,
        )
        print(f"  Collapsed evidence packaged: {vid} (Tier {tier} -> {action})")
        print(f"    Annotated: {package['annotated_image_path']}")
        print(f"    JSON: {package['json_path']}")

        img_report = {
            "filename": img_path.name,
            "resolution": f"{w}x{h}",
            "vehicles": len(vehicles),
            "persons": len(persons),
            "violations_detected": len(violations),
            "routed_tier": tier,
            "routed_action": action,
            "violation_id": vid,
            "demo_annotated_image": str(demo_out_path),
            "evidence_package": {
                "violation_id": vid,
                "tier": tier,
                "action": action,
                "annotated_image": package["annotated_image_path"],
                "json_record": package["json_path"]
            },
            "plates_detected": all_plates,
            "best_plate": plate_text_display,
            "best_plate_confidence": round(plate_info.get("confidence", 0.0), 3)
        }
        results.append(img_report)

        print(f"  Vehicles: {len(vehicles)}, Persons: {len(persons)}")
        if violations:
            for v, d in zip(violations, decisions):
                print(f"  -> Violation: {v.violation_type.value} (conf={v.confidence:.2f}) [Plate: {v.plate_text}]")
        else:
            print("  -> No violations found.")

    # Save summary
    out_path = Path("result/helmet_batch_report.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBatch test finished. Full summary saved to {out_path}")

if __name__ == "__main__":
    main()

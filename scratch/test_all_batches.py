import sys
import json
import time
import uuid
import shutil
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

def clear_evidence_and_results():
    print("Clearing evidence and result directories...")
    # Clear subfolders under evidence
    for sub in ["annotated", "demo", "json", "raw"]:
        d = Path("evidence") / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    
    # Delete any individual files directly in evidence/
    evidence_dir = Path("evidence")
    if evidence_dir.exists():
        for f in evidence_dir.iterdir():
            if f.is_file():
                f.unlink()

    # Clear result folder
    result_dir = Path("result")
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    print("Cleared directories successfully.")

def main():
    # 1. Clear directories
    clear_evidence_and_results()

    # Define directories to test
    test_folders = [
        "helmet",
        "parking",
        "red light",
        "seatbelt",
        "stop line",
        "triple",
        "wrong side"
    ]

    print("Initializing pipeline...")
    preprocessor = ImagePreprocessor()
    detector = VehicleDetector(device="cpu")
    ocr = PlateOCR(plate_detector_weights="ml/models/weights/plate_koushi.pt")
    classifier = ViolationClassifier(stop_line_y=380)
    router = ConfidenceRouter(RepeatOffenderDB())
    packager = EvidencePackager(output_dir="evidence")
    visualizer = FrameVisualizer()

    for folder in test_folders:
        test_dir = Path("test") / folder
        if not test_dir.exists():
            print(f"\nFolder {test_dir} does not exist, skipping.")
            continue

        images = sorted(
            list(test_dir.glob("*.png")) + 
            list(test_dir.glob("*.jpg")) + 
            list(test_dir.glob("*.jpeg")) +
            list(test_dir.glob("*.PNG")) + 
            list(test_dir.glob("*.JPG")) + 
            list(test_dir.glob("*.JPEG"))
        )

        if not images:
            print(f"\nNo images found in {test_dir}, skipping.")
            continue

        print(f"\n==========================================")
        print(f"Running batch test for folder: {folder} ({len(images)} images)")
        print(f"==========================================")

        results = []

        for img_path in images:
            print(f"  Processing {img_path.name}...")
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"    Failed to read {img_path}")
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
                x1, y1, x2, y2 = map(int, vehicle.bbox)
                h_img, w_img = processed.shape[:2]
                veh_crop = processed[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]
                if veh_crop.size > 0:
                    plate_result = ocr.read_plate_from_vehicle(veh_crop)
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
            
            # Clean folder name for filename
            clean_folder_name = folder.replace(" ", "_")
            demo_out_path = Path("evidence") / f"demo_{clean_folder_name}_{img_path.stem}.jpg"
            cv2.imwrite(str(demo_out_path), display)

            # 2. Package collapsed evidence record
            package = packager.create_package(
                frame=frame,
                violations=[d.violation.to_dict() for d in decisions],
                plate_info=plate_info,
                camera_info=DEMO_CAMERA,
                processing_info={"time_ms": round(elapsed_ms, 1), "model": "yolov8m"},
                violation_id=vid,
            )

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

            if violations:
                for v in violations:
                    print(f"    -> Violation: {v.violation_type.value} (conf={v.confidence:.2f}) [Plate: {v.plate_text}]")
            else:
                print("    -> No violations found.")

        # Save summary report for this folder
        clean_folder_name = folder.replace(" ", "_")
        out_path = Path("result") / f"{clean_folder_name}_batch_report.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Finished {folder}. Summary saved to {out_path}")

    print("\nAll batch tests completed successfully.")

if __name__ == "__main__":
    main()

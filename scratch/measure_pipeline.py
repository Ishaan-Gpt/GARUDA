import sys
import time
import os
import cv2
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.pipeline.preprocessor import ImagePreprocessor
from ml.pipeline.detector import VehicleDetector
from ml.pipeline.ocr import PlateOCR
from ml.pipeline.violation_classifier import ViolationClassifier
from ml.pipeline.confidence_router import ConfidenceRouter, RepeatOffenderDB

DEMO_CAMERA = {
    "camera_id":   "BLR-CAM-DEMO-001",
    "location":    "MG Road & Brigade Road Intersection",
    "coordinates": {"lat": 12.9753, "lon": 77.6069},
}

def main():
    print("=" * 70)
    print("GARUDA - Timing Measurement Script")
    print("=" * 70)

    # Resolve weight paths
    weights_dir = Path("ml/models/weights")
    helmet_weights = str(weights_dir / "helmet_cnn.pt")
    plate_weights = str(weights_dir / "plate_yolov8_moin.pt")
    plate_weights_fallback = str(weights_dir / "plate_yolo.pt")

    if not os.path.exists(helmet_weights):
        helmet_weights = None
    if not os.path.exists(plate_weights):
        if os.path.exists(plate_weights_fallback):
            plate_weights = plate_weights_fallback
        else:
            plate_weights = None

    print(f"Helmet weights: {helmet_weights}")
    print(f"Plate weights: {plate_weights}")

    # Measure initialization time
    t_init_start = time.perf_counter()
    preprocessor = ImagePreprocessor()
    detector = VehicleDetector(device="cpu")
    ocr = PlateOCR(plate_detector_weights=plate_weights)
    classifier = ViolationClassifier(stop_line_y=380, helmet_weights_path=helmet_weights)
    repeat_db = RepeatOffenderDB()
    router = ConfidenceRouter(repeat_db)
    t_init_end = time.perf_counter()
    init_time_ms = (t_init_end - t_init_start) * 1000
    print(f"Pipeline Initialization Time: {init_time_ms:.2f} ms")
    print("=" * 70)

    images = [
        "WhatsApp Image 2026-06-20 at 12.24.39 PM.jpeg",
        "WhatsApp Image 2026-06-20 at 12.24.40 PM.jpeg",
        "WhatsApp Image 2026-06-20 at 12.24.41 PM.jpeg",
        "WhatsApp Image 2026-06-20 at 12.24.46 PM.jpeg",
        "WhatsApp Image 2026-06-20 at 12.24.51 PM.jpeg"
    ]

    results = []

    for img_name in images:
        img_path = os.path.join("test", img_name)
        print(f"Processing: {img_name}")
        
        # Load Image
        t_load_start = time.perf_counter()
        frame = cv2.imread(img_path)
        t_load_end = time.perf_counter()
        load_time_ms = (t_load_end - t_load_start) * 1000

        if frame is None:
            print(f"Error: Cannot read {img_path}")
            continue

        h, w = frame.shape[:2]

        # Phase 1: Preprocessing (optimized downscaling + brightness-based bypass)
        t1 = time.perf_counter()
        processed = preprocessor.preprocess(frame, enhance=True, is_video=False)
        t2 = time.perf_counter()
        prep_time_ms = (t2 - t1) * 1000

        # Phase 2: Detection
        t3 = time.perf_counter()
        detections = detector.detect(processed)
        vehicles = detector.get_vehicles(detections)
        persons = detector.get_persons(detections)
        phones = detector.get_phones(detections)
        t4 = time.perf_counter()
        det_time_ms = (t4 - t3) * 1000

        # Phase 3: Violation Classification
        t5 = time.perf_counter()
        all_violations = classifier.check_all(processed, vehicles, persons, phone_detections=phones)
        t6 = time.perf_counter()
        class_time_ms = (t6 - t5) * 1000

        # Phase 4: OCR (Plate Detection + Read)
        t7 = time.perf_counter()
        plate_info = {"formatted_text": "", "confidence": 0.0, "is_valid": False, "state": "Unknown"}
        for vehicle in vehicles:
            plate_region = ocr.detect_plate_region(processed, vehicle.bbox)
            if plate_region is not None and plate_region.size > 0:
                result = ocr.read_plate(plate_region)
                if result.confidence > plate_info.get("confidence", 0):
                    plate_info = result.to_dict()
        t8 = time.perf_counter()
        ocr_time_ms = (t8 - t7) * 1000

        # Phase 5: Routing
        t9 = time.perf_counter()
        decisions = router.route_batch(all_violations, plate_info, DEMO_CAMERA)
        t10 = time.perf_counter()
        route_time_ms = (t10 - t9) * 1000

        total_inference_ms = prep_time_ms + det_time_ms + class_time_ms + ocr_time_ms + route_time_ms

        res = {
            "image": img_name,
            "resolution": f"{w}x{h}",
            "load_time_ms": load_time_ms,
            "prep_time_ms": prep_time_ms,
            "det_time_ms": det_time_ms,
            "class_time_ms": class_time_ms,
            "ocr_time_ms": ocr_time_ms,
            "route_time_ms": route_time_ms,
            "total_inference_ms": total_inference_ms,
            "vehicles_count": len(vehicles),
            "persons_count": len(persons),
            "violations_count": len(all_violations),
            "violations_list": [v.violation_type.value for v in all_violations],
            "plate_text": plate_info.get("formatted_text") or "UNCLEAR",
            "plate_conf": plate_info.get("confidence", 0) * 100,
            "plate_valid": plate_info.get("is_valid")
        }
        results.append(res)
        
        print(f"  Resolution: {res['resolution']}")
        print(f"  Detections: Vehicles={res['vehicles_count']}, Persons={res['persons_count']}")
        print(f"  Violations: {res['violations_count']} {res['violations_list']}")
        print(f"  Plate: {res['plate_text']} (conf={res['plate_conf']:.1f}%, valid={res['plate_valid']})")
        print(f"  Timings (ms): Prep={prep_time_ms:.1f}, Det={det_time_ms:.1f}, Class={class_time_ms:.1f}, OCR={ocr_time_ms:.1f}, Route={route_time_ms:.1f}")
        print(f"  Total Inference: {total_inference_ms:.1f} ms")
        print("-" * 50)

    # Print summary markdown report
    print("\n" + "=" * 70)
    print("FINAL SUMMARY REPORT")
    print("=" * 70)
    print(f"Pipeline Initialization: {init_time_ms:.1f} ms\n")
    
    print("| Image | Resolution | Detections | Violations | Plate (Conf) | Prep (ms) | Det (ms) | Class (ms) | OCR (ms) | Route (ms) | Total Inf (ms) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        v_str = f"{r['violations_count']}"
        if r['violations_list']:
            v_str += f" ({', '.join(r['violations_list'])})"
        p_str = f"{r['plate_text']} ({r['plate_conf']:.0f}%)"
        det_str = f"V:{r['vehicles_count']} P:{r['persons_count']}"
        print(f"| {r['image']} | {r['resolution']} | {det_str} | {v_str} | {p_str} | {r['prep_time_ms']:.1f} | {r['det_time_ms']:.1f} | {r['class_time_ms']:.1f} | {r['ocr_time_ms']:.1f} | {r['route_time_ms']:.1f} | {r['total_inference_ms']:.1f} |")

if __name__ == "__main__":
    main()

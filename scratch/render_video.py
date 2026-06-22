import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.pipeline.detector import VehicleDetector
from ml.pipeline.ocr import PlateOCR
from ml.utils.visualizer import FrameVisualizer

def main():
    input_video = "test/videos/License Plate Detection Test - Dev Drone Bhowmik (720p, h264).mp4"
    output_dir = "evidence/video"
    os.makedirs(output_dir, exist_ok=True)
    output_video = os.path.join(output_dir, "rendered_output.mp4")

    print("Initializing GARUDA ML Pipeline (Koushi-YasirFaiz Plate OCR)...")
    detector = VehicleDetector(device="cpu")
    ocr = PlateOCR()
    visualizer = FrameVisualizer()

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {input_video}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing Video: {input_video}")
    print(f"Resolution: {width}x{height} | FPS: {fps} | Total Frames: {total_frames}")

    # Subsample frames to run much faster on CPU (e.g., target 10 FPS instead of 30 FPS)
    output_fps = 10.0
    sample_interval = max(1, round(fps / output_fps))
    print(f"Subsampling video at 10 FPS (every {sample_interval} frames) to optimize CPU processing")

    # Using mp4v codec for standard mp4 compatibility on Windows
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, output_fps, (width, height))

    frame_idx = 0
    processed_count = 0
    start_time = time.time()

    # Track-based license plate crop & OCR caching to avoid redundant heavy model evaluations
    cached_plates = {} # track_id -> { "plate_crop": np.ndarray, "text": str }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Skip frames to achieve target output FPS
        if frame_idx % sample_interval != 0:
            continue

        processed_count += 1
        
        # Preprocess / Inference with lower resolution imgsz=384 and tracking enabled
        detections = detector.detect_with_tracking(frame, persist=True, imgsz=384)
        vehicles = detector.get_vehicles(detections)

        # Draw default green bounding boxes
        annotated = frame.copy()
        annotated = visualizer.draw_detections(annotated, [d.to_dict() for d in detections], show_conf=False)

        # Process each vehicle for plate detection and crop overlays
        for vehicle in vehicles:
            track_id = vehicle.track_id
            
            # Check tracking cache first to see if we've already done plate detection and OCR on this vehicle
            cached = None
            if track_id is not None and track_id in cached_plates:
                cached = cached_plates[track_id]
                
            vx1, vy1, vx2, vy2 = map(int, vehicle.bbox)
            bbox_area = (vx2 - vx1) * (vy2 - vy1)
            
            # Decide if we need to run OCR:
            # We only run plate detection and OCR if:
            # 1. We have no cached data for this vehicle track yet.
            # 2. Or, the vehicle has moved significantly closer (bounding box area grew by 15% or more).
            # If the vehicle has NOT moved closer, we reuse the cached result (even if it's empty/failed)
            # because the resolution/distance is the same, so retrying would just waste CPU.
            need_ocr = True
            if cached is not None:
                moved_closer = bbox_area > cached["bbox_area"] * 1.15
                if not moved_closer:
                    need_ocr = False

            if not need_ocr and cached is not None:
                plate_crop = cached["plate_crop"]
                formatted_text = cached["text"]
            else:
                # Crop vehicle area
                crop = frame[max(0, vy1):min(height, vy2), max(0, vx1):min(width, vx2)]
                plate_crop = None
                formatted_text = ""
                confidence = 0.0
                
                if crop.size > 0:
                    # Detect license plate region (uses YOLO or Morphological fallback)
                    plate_crop = ocr.detect_plate_region(crop, [0, 0, crop.shape[1], crop.shape[0]])
                    if plate_crop is not None and plate_crop.size > 0:
                        # Perform OCR to read plate text
                        ocr_result = ocr.read_plate_from_vehicle(crop)
                        formatted_text = ocr_result.formatted_text or ""
                        confidence = ocr_result.confidence
                    
                    # Update or store in tracking cache
                    if track_id is not None:
                        if (cached is None or 
                            confidence > cached["confidence"] or 
                            (formatted_text != "" and cached["text"] == "") or
                            bbox_area > cached["bbox_area"] * 1.15):
                            
                            # Keep whichever plate crop is not None
                            final_crop = plate_crop if plate_crop is not None else (cached["plate_crop"] if cached else None)
                            final_text = formatted_text if formatted_text != "" else (cached["text"] if cached else "")
                            final_conf = max(confidence, cached["confidence"]) if cached else confidence
                            
                            cached_plates[track_id] = {
                                "plate_crop": final_crop,
                                "text": final_text,
                                "confidence": final_conf,
                                "bbox_area": bbox_area
                            }
                            
                # Use current or cached fallback if OCR failed this frame but succeeded before
                if (plate_crop is None or formatted_text == "") and cached is not None:
                    plate_crop = plate_crop if plate_crop is not None else cached["plate_crop"]
                    formatted_text = formatted_text if formatted_text != "" else cached["text"]

            if plate_crop is not None and plate_crop.size > 0:
                vx1, vy1, vx2, vy2 = map(int, vehicle.bbox)
                # Draw white bordered plate crop overlay just above the vehicle
                annotated = visualizer.draw_plate_crop(annotated, plate_crop, (vx1, vy1 - 65))
                
                # If readable, write the plate text above the vehicle bounding box
                if formatted_text:
                    cv2.putText(
                        annotated,
                        formatted_text,
                        (vx1, vy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )

        # Write frame to output video
        out.write(annotated)

        # Print progress
        if processed_count % 10 == 0 or frame_idx == total_frames:
            elapsed = time.time() - start_time
            rate = processed_count / elapsed if elapsed > 0 else 0
            eta = ((total_frames / sample_interval) - processed_count) / rate if rate > 0 else 0
            print(f"Processed Frame {frame_idx}/{total_frames} | Sampled {processed_count} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", flush=True)

    cap.release()
    out.release()
    print(f"\n[SUCCESS] Rendered video saved to: {output_video}", flush=True)

if __name__ == '__main__':
    main()

import os
import sys
import time
from pathlib import Path
from collections import defaultdict
import cv2

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.ml_registry import get_ml_registry
from backend.api.stream import _render_frame_full, _reencode_to_browser_h264
from ml.pipeline.tracker import VehicleTracker

def main():
    input_video = "test/videos/License Plate Detection Test - Dev Drone Bhowmik (720p, h264).mp4"
    output_dir = "evidence/video"
    os.makedirs(output_dir, exist_ok=True)
    
    output_video_demo = os.path.join(output_dir, "rendered_output_demo.mp4")
    output_video_annotated = os.path.join(output_dir, "rendered_output_annotated.mp4")

    print("Initializing GARUDA ML Pipeline (detector, OCR, classifier, visualizer)...")
    ml = get_ml_registry()
    if not ml.available:
        print(f"[FATAL] ML pipeline failed to load: {ml.error}")
        sys.exit(1)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {input_video}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing Video: {input_video}")
    print(f"Resolution: {width}x{height} | FPS: {fps} | Total Frames: {total_frames}")

    # Subsample frames to run much faster on CPU (e.g., target 10 FPS instead of 30 FPS)
    output_fps = 10.0
    sample_interval = max(1, round(fps / output_fps))
    print(f"Subsampling video at 10 FPS (every {sample_interval} frames) to optimize CPU processing")

    # Using mp4v codec for standard mp4 compatibility on Windows
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_demo = cv2.VideoWriter(output_video_demo, fourcc, output_fps, (width, height))
    out_annotated = cv2.VideoWriter(output_video_annotated, fourcc, output_fps, (width, height))

    # One tracker.update() per sampled output frame
    STOP_LINE_Y = 400  # Default stop line for this video
    tracker = VehicleTracker(stop_line_y=STOP_LINE_Y)
    ml.classifier.stop_line_y = STOP_LINE_Y
    ml.classifier.fps = output_fps
    ml.classifier.reset_signal_smoothing()

    frame_idx = 0
    processed_count = 0
    first_sampled = True
    start_time = time.time()

    track_seq = {}
    next_seq = [1]
    reported = defaultdict(set)
    cached_plates = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Skip frames to achieve target output FPS
        if frame_idx % sample_interval != 0:
            continue

        processed_count += 1
        
        result = _render_frame_full(
            ml, frame, tracker, frame_idx, not first_sampled,
            track_seq, next_seq, reported, cached_plates
        )
        first_sampled = False
        
        out_annotated.write(result["frame"])
        out_demo.write(result["demo_frame"])

        # Print progress
        if processed_count % 10 == 0 or frame_idx == total_frames:
            elapsed = time.time() - start_time
            rate = processed_count / elapsed if elapsed > 0 else 0
            eta = ((total_frames / sample_interval) - processed_count) / rate if rate > 0 else 0
            print(f"Processed Frame {frame_idx}/{total_frames} | Sampled {processed_count} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", flush=True)

    cap.release()
    out_demo.release()
    out_annotated.release()
    print(f"\n[SUCCESS] Rendered videos saved locally.", flush=True)

    # Post-process: convert both videos to web/WhatsApp compatible H.264 format
    print("\nConverting demo video to universally compatible H.264 format (for WhatsApp/Web)...")
    ok1 = _reencode_to_browser_h264(output_video_demo)
    print("  Demo video H.264 re-encode OK" if ok1 else "  [WARNING] Demo video H.264 re-encode failed")

    print("\nConverting annotated video to universally compatible H.264 format (for WhatsApp/Web)...")
    ok2 = _reencode_to_browser_h264(output_video_annotated)
    print("  Annotated video H.264 re-encode OK" if ok2 else "  [WARNING] Annotated video H.264 re-encode failed")

if __name__ == '__main__':
    main()

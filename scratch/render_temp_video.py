"""
Run the FULL GARUDA pipeline (detector + tracker + every violation check in
ViolationClassifier.check_all, including the red-light/stop-line/wrong-side/
illegal-parking fixes) over a video and write an annotated result video.

Reuses backend.api.stream._render_frame_full — the exact same per-frame
logic the live /ws/video-render endpoint uses — so this is genuinely "the
real pipeline", not a re-implementation.
"""
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.services.ml_registry import get_ml_registry
from backend.api.stream import _render_frame_full, _reencode_to_browser_h264
from backend.services.challan_service import display_name
from ml.pipeline.tracker import VehicleTracker

OUTPUT_DIR = "temp/result"
RENDER_OUTPUT_FPS = 6.0
STOP_LINE_Y = 380  # default calibration — no camera registered for these ad-hoc clips

# Ad-hoc per-clip calibration found by manually inspecting each video (no
# registered camera exists for these, so there's nothing to load from the
# DB) — keyed by input filename. wrong_side_zone/traffic_direction here cover
# the narrow bus-stop service lane where a scooter rider was observed riding
# down it, then turning around and riding back up against the direction
# they (and the lane's apparent flow) had been going. The zone also sweeps
# through the bus's legitimate bus-bay merge path from this camera's oblique
# angle, but ViolationClassifier.check_wrong_side() now exempts buses
# (WRONG_SIDE_EXEMPT_CLASSES) so that no longer false-positives.
CALIBRATION_OVERRIDES = {
    "WhatsApp Video 2026-06-23 at 11.51.08.mp4": {
        "wrong_side_zone": [[1000, 300, 1900, 1300]],
        "traffic_direction": "down",
    },
}


def main() -> None:
    input_video = sys.argv[1] if len(sys.argv) > 1 else "temp/WhatsApp Video 2026-06-23 at 11.50.30.mp4"
    output_name = sys.argv[2] if len(sys.argv) > 2 else "annotated_result.mp4"
    # Trim this many trailing source frames — some of these WhatsApp clips are
    # screen recordings and the last frame or two can be UI chrome (e.g. a
    # YouTube "Up next" overlay), not real footage; safer to drop a small
    # tail than feed garbage frames into the detector.
    trim_tail_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    output_fps_override = float(sys.argv[4]) if len(sys.argv) > 4 else RENDER_OUTPUT_FPS
    
    # Separate outputs for demo vs annotated
    base_out_name = os.path.splitext(output_name)[0]
    output_video_annotated = os.path.join(OUTPUT_DIR, f"{base_out_name}_annotated.mp4")
    output_video_demo = os.path.join(OUTPUT_DIR, f"{base_out_name}_demo.mp4")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading GARUDA ML pipeline (detector, OCR, classifier, all violation checks)...")
    ml = get_ml_registry()
    if not ml.available:
        print(f"[FATAL] ML pipeline failed to load: {ml.error}")
        sys.exit(1)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"[FATAL] Could not open video: {input_video}")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    last_usable_frame = max(0, total_src_frames - trim_tail_frames) if total_src_frames else None

    output_fps = min(src_fps, output_fps_override)
    sample_interval = max(1, round(src_fps / output_fps))

    print(f"Source: {input_video}")
    print(f"  {width}x{height} @ {src_fps:.1f}fps, {total_src_frames} frames "
          f"({total_src_frames / src_fps:.1f}s)")
    print(f"  Sampling every {sample_interval} frames -> output @ {output_fps:.1f}fps")
    if trim_tail_frames:
        print(f"  Trimming last {trim_tail_frames} source frames (screen-recording tail)")

    tracker = VehicleTracker(stop_line_y=STOP_LINE_Y)
    ml.classifier.stop_line_y = STOP_LINE_Y
    # One tracker.update() per sampled output frame
    ml.classifier.fps = output_fps
    ml.classifier.reset_signal_smoothing()

    override = CALIBRATION_OVERRIDES.get(os.path.basename(input_video))
    if override:
        for k, v in override.items():
            setattr(ml.classifier, k, v)
        print(f"  Applied calibration override: {override}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_annotated = cv2.VideoWriter(output_video_annotated, fourcc, output_fps, (width, height))
    writer_demo = cv2.VideoWriter(output_video_demo, fourcc, output_fps, (width, height))

    track_seq: dict = {}
    next_seq = [1]
    reported: dict = defaultdict(set)
    cached_plates: dict = {}
    all_violations: list = []

    frame_idx = 0
    processed_count = 0
    first_sampled = True
    t_start = time.time()

    while True:
        if last_usable_frame is not None and frame_idx >= last_usable_frame:
            break
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            result = _render_frame_full(
                ml, frame, tracker, frame_idx, not first_sampled,
                track_seq, next_seq, reported, cached_plates
            )
            first_sampled = False
            writer_annotated.write(result["frame"])
            writer_demo.write(result["demo_frame"])
            processed_count += 1

            for nv in result["new_violations"]:
                names = [
                    display_name(v.violation_type.value if hasattr(v.violation_type, "value") else str(v.violation_type))
                    for v in nv["violations"]
                ]
                all_violations.append({
                    "frame": frame_idx,
                    "time_sec": round(frame_idx / src_fps, 2),
                    "track_seq": nv["seq"],
                    "violations": names,
                })
                print(f"  [t={frame_idx / src_fps:5.2f}s] vehicle #{nv['seq']}: {', '.join(names)}")

        frame_idx += 1

    cap.release()
    writer_annotated.release()
    writer_demo.release()
    elapsed = time.time() - t_start
    print(f"\nProcessed {processed_count} sampled frames in {elapsed:.1f}s")
    print(f"Raw annotated render written to: {output_video_annotated}")
    print(f"Raw demo render written to: {output_video_demo}")

    print("Re-encoding annotated video to browser/WhatsApp-compatible H.264...")
    ok1 = _reencode_to_browser_h264(output_video_annotated)
    print("  Annotated H.264 re-encode OK" if ok1 else "  [WARNING] Annotated H.264 re-encode failed, raw file kept")

    print("Re-encoding demo video to browser/WhatsApp-compatible H.264...")
    ok2 = _reencode_to_browser_h264(output_video_demo)
    print("  Demo H.264 re-encode OK" if ok2 else "  [WARNING] Demo H.264 re-encode failed, raw file kept")

    print(f"\n=== SUMMARY ===")
    print(f"Total violations found: {len(all_violations)}")
    by_type: dict = defaultdict(int)
    for v in all_violations:
        for name in v["violations"]:
            by_type[name] += 1
    for name, count in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print(f"  {name}: {count}")
    if not all_violations:
        print("  (none — vehicle(s) detected but no violation conditions met)")

    print(f"\nAnnotated result video: {os.path.abspath(output_video_annotated)}")
    print(f"Demo result video:      {os.path.abspath(output_video_demo)}")


if __name__ == "__main__":
    main()

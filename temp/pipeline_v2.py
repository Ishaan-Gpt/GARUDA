"""
GARUDA Pipeline v2 - Full image helmet detection + vehicle association
Strategy:
  1. Detect vehicles on full image (YOLOv8m)
  2. Run helmet model on FULL image (not crops) to find heads/helmets
  3. Associate each 'head' (no helmet) detection with the vehicle it sits above
  4. For violating vehicles, run PlateOCR on that vehicle's crop
"""
import sys, os
sys.path.insert(0, r"D:\vignesh\files\Personal\Hackthon\flipkart_Gridlock2\GARUDA")

import cv2
import numpy as np
from ultralytics import YOLO

IMAGE_PATH = r"D:\vignesh\files\Personal\Hackthon\flipkart_Gridlock2\GARUDA\test\WhatsApp Image 2026-06-20 at 12.25.12 PM.jpeg"
VEHICLE_MODEL = "yolov8m.pt"
HELMET_MODEL  = r"D:\vignesh\files\Personal\Hackthon\flipkart_Gridlock2\GARUDA\temp\best.pt"
PLATE_MODEL   = r"D:\vignesh\files\Personal\Hackthon\flipkart_Gridlock2\GARUDA\ml\models\weights\plate_yolov8_moin.pt"

TWO_WHEELER_COCO_IDS = {1, 3}   # bicycle=1, motorcycle=3

# ── colours ────────────────────────────────────────────────────────────────
RED    = (0,   0,   255)
GREEN  = (0,   255, 0  )
ORANGE = (0,   165, 255)
BLUE   = (255, 0,   0  )
WHITE  = (255, 255, 255)
YELLOW = (0,   255, 255)

def iou_above(head_box, veh_box):
    """Return True if the head box centre is horizontally inside the vehicle box
    and the head box bottom overlaps or is above the vehicle box top third."""
    hx1, hy1, hx2, hy2 = head_box
    vx1, vy1, vx2, vy2 = veh_box
    hcx = (hx1 + hx2) / 2
    hcy = (hy1 + hy2) / 2
    # head centre must be horizontally within vehicle box (with 20 px margin)
    if not (vx1 - 20 <= hcx <= vx2 + 20):
        return False
    # head centre must be above the vehicle's vertical centre
    v_mid_y = (vy1 + vy2) / 2
    if hcy > v_mid_y:
        return False
    return True

def associate_heads_to_vehicles(head_boxes, vehicle_boxes):
    """Return dict: vehicle_idx -> list of associated head boxes."""
    assoc = {i: [] for i in range(len(vehicle_boxes))}
    for hbox in head_boxes:
        best_v = None
        best_area = float('inf')
        for vi, vbox in enumerate(vehicle_boxes):
            if iou_above(hbox, vbox):
                area = (vbox[2]-vbox[0]) * (vbox[3]-vbox[1])
                if area < best_area:
                    best_area = area
                    best_v = vi
        if best_v is not None:
            assoc[best_v].append(hbox)
    return assoc

def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("ERROR: cannot read image"); return
    h, w = img.shape[:2]
    canvas = img.copy()

    print(f"\n{'='*60}")
    print("GARUDA Pipeline v2")
    print(f"Image: {w}x{h}")
    print(f"{'='*60}\n")

    # ── Stage 1: vehicle detection ─────────────────────────────────────────
    print("[Stage 1] Detecting vehicles...")
    veh_model = YOLO(VEHICLE_MODEL)
    veh_results = veh_model(img, conf=0.3, verbose=False)[0]

    vehicles = []
    for box in veh_results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in TWO_WHEELER_COCO_IDS:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            vehicles.append((x1, y1, x2, y2, conf, cls_id))

    print(f"  Found {len(vehicles)} two-wheelers")
    for i, (x1,y1,x2,y2,c,cls) in enumerate(vehicles):
        label = "motorcycle" if cls==3 else "bicycle"
        cv2.rectangle(canvas, (x1,y1), (x2,y2), ORANGE, 2)
        cv2.putText(canvas, f"V{i+1} {label} {c:.2f}", (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, ORANGE, 2)

    # ── Stage 2: helmet detection on FULL IMAGE ────────────────────────────
    print("\n[Stage 2] Running helmet model on full image...")
    helm_model = YOLO(HELMET_MODEL)
    helm_results = helm_model(img, conf=0.25, verbose=False)[0]
    # classes: 0=helmet, 1=head (no helmet), 2=person

    all_detections = []
    head_boxes    = []   # no-helmet riders
    helmet_boxes  = []   # helmeted riders

    for box in helm_results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        name = helm_model.names[cls_id]
        all_detections.append((x1, y1, x2, y2, conf, name))

        if name == "head":         # exposed head = no helmet
            head_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(canvas, (x1,y1), (x2,y2), RED, 2)
            cv2.putText(canvas, f"NO HELMET {conf:.2f}", (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2)
        elif name == "helmet":
            helmet_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(canvas, (x1,y1), (x2,y2), GREEN, 2)
            cv2.putText(canvas, f"HELMET {conf:.2f}", (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2)

    print(f"  Heads (no helmet): {len(head_boxes)}")
    print(f"  Helmets:           {len(helmet_boxes)}")

    # ── Stage 3: associate heads to vehicles ───────────────────────────────
    print("\n[Stage 3] Associating no-helmet detections to vehicles...")
    veh_boxes = [(x1,y1,x2,y2) for x1,y1,x2,y2,c,cls in vehicles]
    assoc = associate_heads_to_vehicles(head_boxes, veh_boxes)

    # ── Stage 4: plate OCR on violating vehicles ───────────────────────────
    print("\n[Stage 4] Reading number plates for violating vehicles...")
    from ml.pipeline.ocr import PlateOCR
    plate_ocr = PlateOCR(plate_detector_weights=PLATE_MODEL)
    print(f"  OCR engine: {plate_ocr._engine_name}")

    violations = []
    for vi, heads in assoc.items():
        if not heads:
            continue
        x1, y1, x2, y2, conf, cls = vehicles[vi]
        print(f"\n  Vehicle V{vi+1} → VIOLATION (exposed heads: {len(heads)})")

        # expand crop slightly for better plate detection
        pad = 20
        cx1 = max(0, x1-pad); cy1 = max(0, y1-pad)
        cx2 = min(w, x2+pad); cy2 = min(h, y2+pad)
        crop = img[cy1:cy2, cx1:cx2]

        plate_text = "UNKNOWN"
        if crop.size > 0:
            result = plate_ocr.read_plate_from_vehicle(crop)
            plate_text = result.formatted_text or result.raw_text or "UNREAD"
            print(f"    Plate raw='{result.raw_text}' formatted='{result.formatted_text}' "
                  f"valid={result.is_valid} engine={result.ocr_engine} conf={result.confidence:.2f}")

        violations.append({
            "vehicle_idx": vi+1,
            "vehicle_box": (x1,y1,x2,y2),
            "plate":       plate_text,
        })

        # mark vehicle box red
        cv2.rectangle(canvas, (x1,y1), (x2,y2), RED, 3)
        label = f"VIOLATION | Plate: {plate_text}"
        cv2.putText(canvas, label, (x1, y2+22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)

    # ── Header ─────────────────────────────────────────────────────────────
    cv2.rectangle(canvas, (0,0), (w, 40), (0,0,0), -1)
    cv2.putText(canvas,
        f"GARUDA v2 | Vehicles: {len(vehicles)} | No-Helmet: {len(head_boxes)} | Violations: {len(violations)}",
        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2)

    out_path = r"D:\vignesh\files\Personal\Hackthon\flipkart_Gridlock2\GARUDA\temp\pipeline_v2_output.jpg"
    cv2.imwrite(out_path, canvas)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Two-wheelers detected : {len(vehicles)}")
    print(f"  No-helmet detections  : {len(head_boxes)}")
    print(f"  Helmet detections     : {len(helmet_boxes)}")
    print(f"  VIOLATIONS            : {len(violations)}")
    for v in violations:
        print(f"    Vehicle V{v['vehicle_idx']} → Plate: {v['plate']}")
    print(f"\n  Output saved: {out_path}")

if __name__ == "__main__":
    main()

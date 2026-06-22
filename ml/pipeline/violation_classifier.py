"""
GARUDA ML Pipeline — Traffic Violation Classifier
===================================================
Detects all 7+ violation types from the problem statement:
  1. Helmet non-compliance
  2. Seatbelt non-compliance
  3. Triple riding
  4. Wrong-side driving
  5. Stop-line violation
  6. Red-light violation
  7. Illegal parking
  +  Phone use while driving
  +  Drowsy driving (see driver_state.py)

Each violation check is an independent method returning
Optional[ViolationResult]. None = no violation found.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .detector import Detection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Violation taxonomy
# ---------------------------------------------------------------------------

class ViolationType(str, Enum):
    HELMET_NON_COMPLIANCE   = "helmet_non_compliance"
    SEATBELT_NON_COMPLIANCE = "seatbelt_non_compliance"
    TRIPLE_RIDING           = "triple_riding"
    WRONG_SIDE_DRIVING      = "wrong_side_driving"
    STOP_LINE_VIOLATION     = "stop_line_violation"
    RED_LIGHT_VIOLATION     = "red_light_violation"
    ILLEGAL_PARKING         = "illegal_parking"
    PHONE_USE               = "phone_use_while_driving"
    DROWSY_DRIVING          = "drowsy_driving"


VIOLATION_SEVERITY: Dict[ViolationType, str] = {
    ViolationType.HELMET_NON_COMPLIANCE:   "high",
    ViolationType.SEATBELT_NON_COMPLIANCE: "medium",
    ViolationType.TRIPLE_RIDING:           "high",
    ViolationType.WRONG_SIDE_DRIVING:      "critical",
    ViolationType.STOP_LINE_VIOLATION:     "medium",
    ViolationType.RED_LIGHT_VIOLATION:     "high",
    ViolationType.ILLEGAL_PARKING:         "low",
    ViolationType.PHONE_USE:               "high",
    ViolationType.DROWSY_DRIVING:          "critical",
}

FINE_AMOUNTS_INR: Dict[ViolationType, int] = {
    ViolationType.HELMET_NON_COMPLIANCE:   1000,
    ViolationType.SEATBELT_NON_COMPLIANCE: 1000,
    ViolationType.TRIPLE_RIDING:           2000,
    ViolationType.WRONG_SIDE_DRIVING:      5000,
    ViolationType.STOP_LINE_VIOLATION:      500,
    ViolationType.RED_LIGHT_VIOLATION:     1000,
    ViolationType.ILLEGAL_PARKING:          500,
    ViolationType.PHONE_USE:               5000,
    ViolationType.DROWSY_DRIVING:          2000,
}

# Violations that need very high confidence before auto-challan (legal sensitivity)
HIGH_SENSITIVITY_VIOLATIONS = {
    ViolationType.HELMET_NON_COMPLIANCE,
    ViolationType.SEATBELT_NON_COMPLIANCE,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ViolationResult:
    violation_type: ViolationType
    confidence: float
    severity: str
    fine_amount: int
    bbox: List[float]
    metadata: Dict = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        vtype: ViolationType,
        confidence: float,
        bbox: List[float],
        metadata: Optional[Dict] = None,
    ) -> "ViolationResult":
        return cls(
            violation_type=vtype,
            confidence=round(confidence, 4),
            severity=VIOLATION_SEVERITY[vtype],
            fine_amount=FINE_AMOUNTS_INR[vtype],
            bbox=bbox,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict:
        res = {
            "type": self.violation_type.value,
            "confidence": self.confidence,
            "severity": self.severity,
            "fine_amount_inr": self.fine_amount,
            "bbox": [round(v, 2) for v in self.bbox],
            "metadata": self.metadata,
        }
        if hasattr(self, "plate_text") and self.plate_text:
            res["plate_text"] = self.plate_text
        return res


# ---------------------------------------------------------------------------
# Helmet binary classifier (lightweight)
# ---------------------------------------------------------------------------

class HelmetClassifier:
    """
    Binary helmet detector using a CNN when model weights are available,
    with a visual heuristic fallback (edge-density + color variance).

    The heuristic intentionally returns confidence 0.50–0.72 so that
    borderline cases go to Tier 2 (human review) rather than auto-challan.
    """

    def __init__(self, weights_path: Optional[str] = None) -> None:
        self._model = None
        self._class_to_idx = {"no_helmet": 0, "helmet": 1}
        if weights_path:
            self._load_weights(weights_path)

    def _load_weights(self, path: str) -> None:
        try:
            import json
            import torch
            from ml.models.helmet_cnn import HelmetCNN

            model = HelmetCNN(pretrained_mobilenet=True)
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            self._model = model

            # Trained class_to_idx may not be alphabetical (helmet=1 by convention,
            # but train_helmet.py records the real mapping in helmet_metrics.json)
            metrics_path = Path(path).with_name("helmet_metrics.json")
            if metrics_path.exists():
                mapping = json.loads(metrics_path.read_text()).get("class_to_idx")
                if mapping:
                    self._class_to_idx = mapping

            logger.info("Helmet classifier loaded (trained weights): %s", path)
        except Exception as e:
            logger.warning("Could not load helmet weights (%s). Using heuristic.", e)

    def classify(self, head_crop: np.ndarray) -> Tuple[bool, float]:
        """
        Returns (helmet_present, confidence).

        Parameters
        ----------
        head_crop : BGR image crop of the head region (~upper 35% of rider bbox)
        """
        if head_crop is None or head_crop.size < 100:
            return False, 0.30

        if self._model is not None:
            return self._nn_classify(head_crop)

        return self._heuristic_classify(head_crop)

    def _nn_classify(self, image: np.ndarray) -> Tuple[bool, float]:
        import torch
        import torchvision.transforms as T  # type: ignore

        tf = T.Compose([
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        helmet_idx = self._class_to_idx.get("helmet", 1)
        tensor = tf(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        with torch.no_grad():
            logits = self._model(tensor)
            prob = float(torch.softmax(logits, dim=1)[0][helmet_idx])
        return prob > 0.50, prob

    def classify_batch(self, head_crops: List[np.ndarray]) -> List[Tuple[bool, float]]:
        """
        Classify a batch of head crops in a single PyTorch forward pass.
        """
        if not head_crops:
            return []

        if self._model is None:
            # Fallback to heuristic sequentially
            return [self.classify(crop) for crop in head_crops]

        import torch
        import torchvision.transforms as T

        tf = T.Compose([
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        tensors = []
        valid_indices = []
        for i, crop in enumerate(head_crops):
            if crop is not None and crop.size >= 100:
                tensors.append(tf(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                valid_indices.append(i)

        if not tensors:
            return [(False, 0.30) for _ in head_crops]

        batched_tensor = torch.stack(tensors)
        helmet_idx = self._class_to_idx.get("helmet", 1)

        with torch.no_grad():
            logits = self._model(batched_tensor)
            probs = torch.softmax(logits, dim=1)[:, helmet_idx].tolist()

        # Map back to original list size and order
        results = [(False, 0.30)] * len(head_crops)
        for idx, prob in zip(valid_indices, probs):
            results[idx] = (prob > 0.50, prob)

        return results

    def _heuristic_classify(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Helmet heuristic: helmets have high edge density (vents, visor rim)
        and low colour variance (solid-colour shell).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / max(edges.size, 1)

        # Colour uniformity (helmets tend to be single-colour)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat_std = float(np.std(hsv[:, :, 1])) / 128.0  # normalise 0-1

        # Score: high edges + low sat spread → more likely helmet
        score = min(0.65, edge_density * 4.0 + (1.0 - sat_std) * 0.25)
        return score > 0.50, score


# ---------------------------------------------------------------------------
# AICity Track 5 — direct 9-class helmet violation detector
# ---------------------------------------------------------------------------

class AIHelmetViolationDetector:
    """
    Helmet violation detector using JarvanLee/yolov8-helmet-violation-detection (mAP@0.5=64.8%).

    3 classes: helmet (0), head (1=no helmet), person (2)
    Runs on the FULL image and associates 'head' detections with vehicle bboxes.
    Falls back to crop-based HelmetClassifier when weights are unavailable.
    """

    _WEIGHTS = Path(__file__).parent.parent / "models" / "weights" / "violations" / "helmet_best.pt"
    _NO_HELMET_CLASS = "head"    # exposed head = no helmet
    _HELMET_CLASS    = "helmet"
    # A "head" (no-helmet) box detected below this confidence is too
    # uncertain to issue a violation from — default to compliant rather
    # than guessing. Riders should not be flagged "No Helmet" off a weak,
    # borderline detection.
    _MIN_VIOLATION_CONF = 0.40

    def __init__(self, helmet_weights_path: Optional[str] = None) -> None:
        self._model = None
        self._class_names: List[str] = []
        self._fallback = HelmetClassifier(helmet_weights_path)
        # Cache full-image results so multiple vehicles share one inference call
        self._cached_image_id: Optional[int] = None
        self._cached_heads:    List[Tuple[List[float], float]] = []   # (box, conf) where head (no helmet) found
        self._cached_helmets:  List[Tuple[List[float], float]] = []   # (box, conf) where helmet found
        self._try_load()

    def _try_load(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
            if self._WEIGHTS.exists():
                self._model = YOLO(str(self._WEIGHTS))
                self._class_names = list(self._model.names.values())
                logger.info("HelmetDetector: loaded %s (%d classes: %s)",
                            self._WEIGHTS.name, len(self._class_names), self._class_names)
            else:
                logger.info("helmet_best.pt not found — using crop-based HelmetClassifier fallback")
        except Exception as e:
            logger.warning("Could not load helmet model (%s), using fallback", e)

    def _run_full_image(self, image: np.ndarray) -> None:
        """Run best.pt on full image and cache head/helmet boxes."""
        img_id = id(image)
        if img_id == self._cached_image_id:
            return  # already cached for this frame

        self._cached_image_id = img_id
        self._cached_heads    = []
        self._cached_helmets  = []

        try:
            results = self._model.predict(image, conf=0.15, iou=0.45, verbose=False)
            for result in results:
                for box in result.boxes:
                    name = result.names[int(box.cls[0])]
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    if name == self._NO_HELMET_CLASS:
                        self._cached_heads.append((xyxy, conf))
                    elif name == self._HELMET_CLASS:
                        self._cached_helmets.append((xyxy, conf))
        except Exception as e:
            logger.warning("Helmet full-image inference failed: %s", e)

    def _head_above_vehicle(self, head_box: List[float], veh_box: List[float]) -> bool:
        """
        True if a head plausibly belongs to a rider of this vehicle: horizontally
        within the bike's own width (scaled margin, not a flat pixel offset — a flat
        +-20px margin is too loose for small/distant vehicles and lets bystanders'
        heads in a crowd get matched), and vertically in the narrow band just above
        the vehicle's top edge where a seated rider's head actually sits — not merely
        "anywhere above the vehicle's midline", which catches pedestrians and signage.
        """
        hx1, hy1, hx2, hy2 = head_box
        vx1, vy1, vx2, vy2 = veh_box
        v_width  = max(vx2 - vx1, 1.0)
        v_height = max(vy2 - vy1, 1.0)

        hcx = (hx1 + hx2) / 2
        hcy = (hy1 + hy2) / 2

        margin = max(0.20 * v_width, 10.0)
        if not (vx1 - margin <= hcx <= vx2 + margin):
            return False

        # Rider head zone: from slightly above the vehicle's top edge down to its midline.
        band_top    = vy1 - 0.9 * v_height
        band_bottom = vy1 + 0.5 * v_height
        return band_top <= hcy <= band_bottom

    def detect(
        self,
        image: np.ndarray,
        vehicle: "Detection",
    ) -> Tuple[bool, float, int]:
        """
        Detect helmet violation on a 2-wheeler.
        Runs best.pt on the full image (once per frame, cached) then associates
        'head' detections with the vehicle bounding box.

        Returns (violation_found, confidence, rider_count).
        rider_count is used for triple-riding detection.
        """
        if not vehicle.is_two_wheeler:
            return False, 0.0, 0

        x1, y1, x2, y2 = map(int, vehicle.bbox)

        if self._model is None:
            head_h = int((y2 - y1) * 0.45)
            head_crop = image[y1 : y1 + head_h, x1:x2]
            has_helmet, conf = self._fallback.classify(head_crop)
            return not has_helmet, conf, 1

        try:
            self._run_full_image(image)
            veh_box = [x1, y1, x2, y2]

            associated_heads = [(h, c) for h, c in self._cached_heads
                                if self._head_above_vehicle(h, veh_box)]
            associated_helmets = [(h, c) for h, c in self._cached_helmets
                                  if self._head_above_vehicle(h, veh_box)]
            rider_count = len(associated_heads) + len(associated_helmets)

            if associated_heads:
                best_conf = max(c for _, c in associated_heads)
                if best_conf < self._MIN_VIOLATION_CONF:
                    # Detected a "head" box, but too unsure to call it a real
                    # no-helmet violation — default to compliant, not a guess.
                    return False, best_conf, max(rider_count, 1)
                return True, best_conf, max(rider_count, 1)

            if rider_count == 0:
                # No head/helmet found near vehicle — use fallback
                head_h = int((y2 - y1) * 0.45)
                head_crop = image[y1 : y1 + head_h, x1:x2]
                has_helmet, fb_conf = self._fallback.classify(head_crop)
                return not has_helmet, fb_conf, 1

            return False, 0.0, rider_count
        except Exception as e:
            logger.warning("Helmet inference failed (%s), using fallback", e)
            head_h = int((y2 - y1) * 0.45)
            head_crop = image[y1 : y1 + head_h, x1:x2]
            has_helmet, conf = self._fallback.classify(head_crop)
            return not has_helmet, conf, 1


# ---------------------------------------------------------------------------
# Signal state detector
# ---------------------------------------------------------------------------

class SignalStateDetector:
    """
    Traffic light state detection using HSV colour blobs.
    Works on a full frame (searches for traffic light region) or a
    pre-cropped signal region.
    """

    def detect(
        self,
        frame: np.ndarray,
        signal_bbox: Optional[List[float]] = None,
    ) -> Tuple[str, float]:
        """
        Returns (state, confidence).
        state: "red" | "yellow" | "green" | "off" | "unknown"
        """
        if signal_bbox:
            x1, y1, x2, y2 = map(int, signal_bbox)
            region = frame[y1:y2, x1:x2]
        else:
            # Restrict to top 40% — traffic lights are never in the lower half
            h = frame.shape[0]
            region = frame[:int(h * 0.4), :]

        if region is None or region.size == 0:
            return "unknown", 0.0

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Red: two HSV ranges (hue wraps at 180)
        red1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
        red2 = cv2.inRange(hsv, (160, 120, 70), (180, 255, 255))
        red_px = int(np.sum((red1 | red2) > 0))

        # Yellow
        yellow_px = int(np.sum(cv2.inRange(hsv, (18, 100, 100), (38, 255, 255)) > 0))

        # Green
        green_px = int(np.sum(cv2.inRange(hsv, (40, 100, 100), (85, 255, 255)) > 0))

        total = red_px + yellow_px + green_px
        if total < 30:
            return "off", 0.40

        dominant = max(
            [("red", red_px), ("yellow", yellow_px), ("green", green_px)],
            key=lambda x: x[1],
        )
        conf = min(0.97, dominant[1] / total + 0.25)
        return dominant[0], conf


# ---------------------------------------------------------------------------
# ML-based signal detector (KASTEL YOLOv8x — 4-dataset pretrained)
# ---------------------------------------------------------------------------

class MLSignalStateDetector:
    """
    Traffic light detector using KASTEL YOLOv8x weights trained on
    DTLD + LISA + BSTLD + HDTLR (4 large datasets, 20 signal classes).
    Falls back to HSV SignalStateDetector if weights are missing.
    """

    _WEIGHTS = Path(__file__).parent.parent / "models" / "weights" / "violations" / "traffic_lights_yolov8x.pt"

    def __init__(self) -> None:
        self._model = None
        self._fallback = SignalStateDetector()
        self._try_load()

    def _try_load(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
            if self._WEIGHTS.exists():
                self._model = YOLO(str(self._WEIGHTS))
                logger.info("MLSignalStateDetector: loaded %s", self._WEIGHTS.name)
            else:
                logger.warning("Signal weights not found (%s), using HSV fallback", self._WEIGHTS)
        except Exception as e:
            logger.warning("Could not load signal model (%s), using HSV fallback", e)

    def detect(
        self,
        frame: np.ndarray,
        signal_bbox: Optional[List[float]] = None,
    ) -> Tuple[str, float]:
        """Returns (state, confidence). state: 'red'|'yellow'|'green'|'off'|'unknown'"""
        if frame is None or frame.size == 0:
            return "unknown", 0.0

        if signal_bbox:
            x1, y1, x2, y2 = map(int, signal_bbox)
            region = frame[y1:y2, x1:x2]
        else:
            h = frame.shape[0]
            region = frame[:int(h * 0.4), :]

        if self._model is None:
            return self._fallback.detect(frame, signal_bbox)

        try:
            results = self._model.predict(region, conf=0.35, iou=0.5, verbose=False)
            best_label, best_conf = "unknown", 0.0
            for result in results:
                for box in result.boxes:
                    name = result.names[int(box.cls[0])].lower()
                    conf = float(box.conf[0])
                    if conf <= best_conf:
                        continue
                    if "green" in name:
                        best_label, best_conf = "green", conf
                    elif "red" in name:
                        best_label, best_conf = "red", conf
                    elif "yellow" in name:
                        best_label, best_conf = "yellow", conf
                    elif name == "off":
                        best_label, best_conf = "off", conf
            if best_label == "unknown":
                return self._fallback.detect(frame, signal_bbox)
            return best_label, best_conf
        except Exception as e:
            logger.warning("Signal ML inference failed (%s), using HSV", e)
            return self._fallback.detect(frame, signal_bbox)

    def detect_with_presence(
        self,
        frame: np.ndarray,
        signal_bbox: Optional[List[float]] = None,
    ) -> Tuple[str, float, bool]:
        """
        Returns (state, confidence, light_detected).

        light_detected is True only when the dedicated traffic-light YOLO
        model is loaded AND actually found a traffic-light box in-frame —
        never set True by the HSV colour-blob fallback, which is too easily
        triggered by unrelated red/green objects (brake lights, signage,
        foliage) to be trusted as proof a signal even exists. Red-light and
        stop-line checks must gate on this: a light has to be confirmed
        present, and confirmed red, before "vehicle past the stop line"
        means anything.
        """
        if frame is None or frame.size == 0 or self._model is None:
            return "unknown", 0.0, False

        if signal_bbox:
            x1, y1, x2, y2 = map(int, signal_bbox)
            region = frame[y1:y2, x1:x2]
        else:
            h = frame.shape[0]
            region = frame[:int(h * 0.4), :]

        try:
            results = self._model.predict(region, conf=0.35, iou=0.5, verbose=False)
            best_label, best_conf = "unknown", 0.0
            for result in results:
                for box in result.boxes:
                    name = result.names[int(box.cls[0])].lower()
                    conf = float(box.conf[0])
                    if conf <= best_conf:
                        continue
                    if "green" in name:
                        best_label, best_conf = "green", conf
                    elif "red" in name:
                        best_label, best_conf = "red", conf
                    elif "yellow" in name:
                        best_label, best_conf = "yellow", conf
                    elif name == "off":
                        best_label, best_conf = "off", conf
            if best_label == "unknown":
                return "unknown", 0.0, False
            return best_label, best_conf, True
        except Exception as e:
            logger.warning("Signal ML inference failed (%s)", e)
            return "unknown", 0.0, False


# ---------------------------------------------------------------------------
# Master violation classifier
# ---------------------------------------------------------------------------

class ViolationClassifier:
    """
    Runs all violation checks against a single frame's detections.

    Parameters
    ----------
    stop_line_y        : Y-pixel coordinate of the stop line (must be calibrated)
    parking_zones      : List of (x1,y1,x2,y2) no-parking zone rectangles
    traffic_direction  : Legal direction of travel as seen by this camera —
                          one of "down"|"up"|"left"|"right" (must be calibrated)
    wrong_side_zone     : List of (x1,y1,x2,y2) zones reserved for oncoming
                          traffic only (must be calibrated)
    helmet_weights_path: Optional path to trained helmet CNN weights
    wrong_side_lane     : Which half of the frame is the "wrong" lane for oncoming
                          traffic at this camera — "left" or "right". Depends on
                          which side of the road the camera is mounted on, not a
                          universal constant; must be calibrated per camera like
                          stop_line_y. Defaults to "left" (the common case for a
                          camera facing oncoming near-lane traffic on a left-hand
                          drive road).

    stop_line_y, parking_zones, traffic_direction, and wrong_side_zone are
    plain mutable attributes (not just constructor args) so a caller serving
    multiple cameras from one shared classifier instance (e.g. backend/api/
    jobs.py) can re-point calibration per request without reloading models.
    """

    _SEATBELT_WEIGHTS = Path(__file__).parent.parent / "models" / "weights" / "violations" / "seatbelt_classifier.pt"
    VALID_DIRECTIONS = ("down", "up", "left", "right")
    # Global floor applied to every violation in check_all() — below this,
    # a flagged violation is too uncertain to cite and is dropped (treated
    # as compliant/green) rather than reported. Does not apply to license
    # plate OCR confidence, which is a separate, unrelated signal.
    MIN_VIOLATION_CONFIDENCE = 0.40

    def __init__(
        self,
        stop_line_y: int = 400,
        parking_zones: Optional[List[List[int]]] = None,
        traffic_direction: str = "down",
        wrong_side_zone: Optional[List[List[int]]] = None,
        helmet_weights_path: Optional[str] = None,
        wrong_side_lane: str = "left",
    ) -> None:
        self.stop_line_y = stop_line_y
        self.parking_zones: List[List[int]] = parking_zones or []
        self.wrong_side_lane = wrong_side_lane if wrong_side_lane in ("left", "right") else "left"
        self.traffic_direction = traffic_direction if traffic_direction in self.VALID_DIRECTIONS else "down"
        self.wrong_side_zone: List[List[int]] = wrong_side_zone or []
        self.helmet_clf = HelmetClassifier(helmet_weights_path)
        self.ai_helmet = AIHelmetViolationDetector(helmet_weights_path)
        self.signal_det = MLSignalStateDetector()
        self._parked_since: Dict[int, float] = {}
        self._seatbelt_model = None
        self._load_seatbelt_model()

    def _load_seatbelt_model(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
            if self._SEATBELT_WEIGHTS.exists():
                self._seatbelt_model = YOLO(str(self._SEATBELT_WEIGHTS))
                logger.info("SeatbeltClassifier loaded: %s", self._SEATBELT_WEIGHTS.name)
            else:
                logger.info("seatbelt_classifier.pt not found — using Hough fallback")
        except Exception as e:
            logger.warning("Could not load seatbelt model: %s", e)

    # ------------------------------------------------------------------
    # 1. Helmet non-compliance
    # ------------------------------------------------------------------

    def check_helmet(
        self,
        image: np.ndarray,
        vehicle: Detection,
        ai_result: Optional[Tuple[bool, float, int]] = None,
    ) -> Optional[ViolationResult]:
        """
        Check helmet on rider of a 2-wheeler.
        Uses YOLOv8n helmet_violation.pt when available (mAP@0.5=0.881),
        otherwise falls back to crop-based CNN + heuristic.

        ai_result : pre-computed self.ai_helmet.detect(image, vehicle) output.
                    check_all() shares this with check_triple_riding so the
                    helmet model only runs once per 2-wheeler instead of twice.
        """
        if not vehicle.is_two_wheeler:
            return None

        violation_found, conf, _ = ai_result if ai_result is not None else self.ai_helmet.detect(image, vehicle)
        if violation_found:
            return ViolationResult.create(
                ViolationType.HELMET_NON_COMPLIANCE,
                confidence=conf,
                bbox=vehicle.bbox,
                metadata={"helmet_score": round(conf, 3),
                          "method": "aicity_9cls" if self.ai_helmet._model else "cnn_crop"},
            )
        return None

    # ------------------------------------------------------------------
    # 2. Triple riding
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_iou(box1: List[float], box2: List[float]) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return intersection / (area1 + area2 - intersection + 1e-6)

    @classmethod
    def associate_riders_with_vehicles(
        cls,
        vehicles: List[Detection],
        persons: List[Detection],
        iou_threshold: float = 0.05,
        max_riders: int = 4,
    ) -> Dict[int, List[Detection]]:
        """
        Assign each person to at most one nearby two-wheeler, ranked by distance.

        Ported from the rider-vehicle association heuristic in
        temp/AI_Traffic_Violation_Detection_triple_riding_detection/detect_triple_riding.py
        (group_riders_with_vehicles). A person is a rider candidate for a vehicle if
        either their box overlaps the vehicle (IoU) or they're horizontally aligned
        with the vehicle AND their feet (bbox bottom) fall within the vehicle's
        vertical span — this catches riders whose torso/head extends above the bike's
        own bbox, which a simple "center inside vehicle bbox" check misses.

        Returns {id(vehicle): [rider Detection, ...]}, keyed by Python object identity
        since vehicles/persons are rebuilt fresh each frame.
        """
        rider_map: Dict[int, List[Detection]] = {}
        used_person_idx: set = set()

        for vehicle in vehicles:
            vx1, vy1, vx2, vy2 = vehicle.bbox
            v_cx, v_cy = vehicle.center
            v_width = max(vx2 - vx1, 1.0)
            v_height = max(vy2 - vy1, 1.0)

            candidates: List[Tuple[Detection, int, float]] = []
            for idx, person in enumerate(persons):
                if idx in used_person_idx:
                    continue

                p_cx, p_cy = person.center
                dist = ((v_cx - p_cx) ** 2 + (v_cy - p_cy) ** 2) ** 0.5 / max(v_width, v_height)
                horiz_dist = abs(p_cx - v_cx) / v_width

                iou = cls._compute_iou(vehicle.bbox, person.bbox)

                p_bottom = person.bbox[3]
                top_margin = 0.05 * v_height
                bottom_margin = 0.30 * v_height
                bottom_overlap = (p_bottom >= (vy1 - top_margin)) and (p_bottom <= (vy2 + bottom_margin))

                if iou > iou_threshold or (horiz_dist < 1.2 and bottom_overlap):
                    candidates.append((person, idx, dist))

            candidates.sort(key=lambda c: c[2])
            riders: List[Detection] = []
            for person, idx, _ in candidates[:max_riders]:
                riders.append(person)
                used_person_idx.add(idx)

            rider_map[id(vehicle)] = riders

        return rider_map

    def check_triple_riding(
        self,
        vehicle: Detection,
        persons: List[Detection],
        rider_map: Optional[Dict[int, List[Detection]]] = None,
        ai_result: Optional[Tuple[bool, float, int]] = None,
    ) -> Optional[ViolationResult]:
        """
        Count persons on a 2-wheeler.
        Prioritizes the IoU + position association heuristic (Vignesh's).
        Falls back to the AI helmet violation detector model (helmet_best.pt)
        if it is loaded and rider_map/heuristic count is not available.
        """
        if not vehicle.is_two_wheeler:
            return None

        if rider_map is not None:
            rider_count = len(rider_map.get(id(vehicle), []))
        elif ai_result is not None and self.ai_helmet._model is not None:
            rider_count = ai_result[2]
        else:
            # Last-resort fallback if no rider_map was precomputed
            rider_count = len(self.associate_riders_with_vehicles([vehicle], persons).get(id(vehicle), []))

        if rider_count >= 3:
            conf = min(0.95, 0.60 + 0.10 * rider_count)
            return ViolationResult.create(
                ViolationType.TRIPLE_RIDING,
                confidence=conf,
                bbox=vehicle.bbox,
                metadata={"person_count": rider_count},
            )
        return None

    # ------------------------------------------------------------------
    # 3. Red-light violation (track-history based)
    # ------------------------------------------------------------------

    def check_red_light(
        self,
        track_bboxes: List[List[float]],
        signal_state: str,
        signal_conf: float,
        latest_bbox: List[float],
        light_detected: bool = True,
    ) -> Optional[ViolationResult]:
        """
        Check if vehicle crossed stop line while signal is red.

        Sequence enforced: (1) a traffic light must actually be detected in
        frame, (2) it must be red, (3) the vehicle's track history must show
        it past the calibrated stop line. Each step gates the next — no
        guessing a red light into existence from track history alone.

        Parameters
        ----------
        track_bboxes   : Last N bounding boxes from tracker history
        signal_state   : Current signal colour
        signal_conf    : Confidence of signal detection
        latest_bbox    : Most recent bbox for the violation record
        light_detected : True only if a traffic light was actually confirmed
                          present in-frame (see MLSignalStateDetector.
                          detect_with_presence) — without this, signal_state
                          cannot be trusted enough to issue a violation.
        """
        if not light_detected:
            return None
        if signal_state.lower() != "red":
            return None
        if signal_conf < 0.60:
            return None

        for bbox in track_bboxes[-15:]:
            if bbox[3] > self.stop_line_y:   # y2 > stop line
                return ViolationResult.create(
                    ViolationType.RED_LIGHT_VIOLATION,
                    confidence=min(0.97, signal_conf * 0.98),
                    bbox=latest_bbox or bbox,
                    metadata={
                        "signal_state": "red",
                        "stop_line_y": self.stop_line_y,
                        "signal_confidence": round(signal_conf, 3),
                    },
                )
        return None

    # ------------------------------------------------------------------
    # 4. Stop-line violation (stationary over line)
    # ------------------------------------------------------------------

    def check_stop_line(
        self,
        track_bboxes: List[List[float]],
        signal_state: str,
        latest_bbox: List[float],
        light_detected: bool = True,
    ) -> Optional[ViolationResult]:
        """
        Detect vehicle stopped over the stop line (not crossing, just encroaching).

        Same sequence as check_red_light: a traffic light must be confirmed
        present before its colour means anything.
        """
        if not light_detected:
            return None
        if signal_state.lower() not in ("red", "yellow"):
            return None

        if not track_bboxes or not latest_bbox:
            return None

        # Vehicle front (y2) is over stop line
        if latest_bbox[3] <= self.stop_line_y:
            return None

        # Check vehicle is stationary (small velocity)
        if len(track_bboxes) >= 5:
            prev = track_bboxes[-5]
            velocity_y = abs(prev[3] - latest_bbox[3])
            if velocity_y > 8:
                return None  # Still moving

        return ViolationResult.create(
            ViolationType.STOP_LINE_VIOLATION,
            confidence=0.85,
            bbox=latest_bbox,
            metadata={"stop_line_y": self.stop_line_y, "signal": signal_state},
        )

    # ------------------------------------------------------------------
    # 5. Wrong-side driving (velocity vector based)
    # ------------------------------------------------------------------

    # Unit vectors for each calibrated legal-travel direction, in image
    # coordinates (y increases downward).
    _DIRECTION_VECTORS: Dict[str, Tuple[float, float]] = {
        "down":  (0.0, 1.0),
        "up":    (0.0, -1.0),
        "right": (1.0, 0.0),
        "left":  (-1.0, 0.0),
    }

    def check_wrong_side(
        self,
        velocity: Tuple[float, float],
        latest_bbox: List[float],
        direction: Optional[str] = None,
    ) -> Optional[ViolationResult]:
        """
        Detect wrong-side driving: a vehicle moving against this camera's
        calibrated legal direction of travel inside the wrong_side_zone.

        Parameters
        ----------
        velocity  : (vx, vy) from tracker
        direction : "down"|"up"|"left"|"right" — legal direction of travel as
                    seen by this camera. Falls back to self.traffic_direction
                    (set per-camera by the caller) when not given explicitly.
        """
        if not self.wrong_side_zone:
            return None

        # Check vehicle center lies in calibrated wrong_side_zone
        cx = (latest_bbox[0] + latest_bbox[2]) / 2
        cy = (latest_bbox[1] + latest_bbox[3]) / 2
        in_zone = any(
            zx1 <= cx <= zx2 and zy1 <= cy <= zy2
            for zx1, zy1, zx2, zy2 in self.wrong_side_zone
        )
        if not in_zone:
            return None

        direction = direction if direction in self.VALID_DIRECTIONS else self.traffic_direction
        vx, vy = velocity
        speed = (vx ** 2 + vy ** 2) ** 0.5

        if speed < 3.0:  # Too slow / stationary
            return None

        dx, dy = self._DIRECTION_VECTORS[direction]
        # Velocity component along the legal direction; negative = moving
        # against it (generalizes the old "vy < -5 when traffic flows down"
        # check to all 4 cardinal directions via dot product).
        along_expected = vx * dx + vy * dy

        if along_expected < -5:
            conf = min(0.90, 0.55 + speed / 80)
            return ViolationResult.create(
                ViolationType.WRONG_SIDE_DRIVING,
                confidence=conf,
                bbox=latest_bbox,
                metadata={
                    "velocity_vx": round(vx, 2),
                    "velocity_vy": round(vy, 2),
                    "expected_direction": direction,
                },
            )
        return None

    # ------------------------------------------------------------------
    # 6. Illegal parking
    # ------------------------------------------------------------------

    def check_illegal_parking(
        self,
        vehicle: Detection,
        track_id: int,
        is_stationary: bool,
        in_no_parking_zone: bool,
        parking_threshold_sec: float = 300.0,
    ) -> Optional[ViolationResult]:
        """
        Detect vehicle parked in a no-parking zone for > 5 minutes.
        """
        if not in_no_parking_zone or not is_stationary:
            if track_id in self._parked_since:
                del self._parked_since[track_id]
            return None

        now = time.monotonic()
        if track_id not in self._parked_since:
            self._parked_since[track_id] = now
            return None

        duration = now - self._parked_since[track_id]
        if duration >= parking_threshold_sec:
            return ViolationResult.create(
                ViolationType.ILLEGAL_PARKING,
                confidence=0.92,
                bbox=vehicle.bbox,
                metadata={"parked_duration_sec": int(duration)},
            )
        return None

    # ------------------------------------------------------------------
    # 7. Seatbelt non-compliance
    # ------------------------------------------------------------------

    def check_seatbelt(
        self,
        image: np.ndarray,
        vehicle: Detection,
    ) -> Optional[ViolationResult]:
        """
        Detect missing seatbelt using RISEFyolov11s-seatbelt classifier.

        Flow:
          1. Skip non-four-wheelers, tiny/square bboxes (autos, bikes), and
             frame-filling bboxes (trucks/buses shot close-up where the box
             covers most of the image — there's no recognisable windshield
             ROI in that crop, just background/cab/cargo, and feeding that
             into the classifier produces a meaningless but overconfident result)
          2. Crop windshield + driver-seat ROI (upper 60%, driver side 70%)
          3. CLAHE + gamma correction for glare/reflection
          4. Feed crop to YOLOv11s classifier → seat_belt | no_seatbelt
          5. Fallback to Hough-diagonal heuristic if model unavailable
        """
        if not vehicle.is_four_wheeler:
            return None

        x1, y1, x2, y2 = map(int, vehicle.bbox)
        vh, vw = y2 - y1, x2 - x1

        if vw < 110 or vh < 80:
            return None
        # Skip frame-filling boxes — no usable windshield ROI in a crop that's
        # mostly background/cab/cargo (common on trucks/buses shot up close)
        img_h, img_w = image.shape[:2]
        if (vw * vh) > 0.35 * (img_w * img_h):
            return None
        # Skip square/tall bboxes — likely autos or misclassified bikes
        if vw < vh * 0.75:
            return None

        # ── Windshield / driver-seat ROI ─────────────────────────────────────
        # India is RHD → driver on right; take left 70% to capture windshield fully
        roi_y1 = y1
        roi_y2 = y1 + int(vh * 0.60)
        roi_x1 = x1
        roi_x2 = x1 + int(vw * 0.70)
        roi = image[max(0, roi_y1):roi_y2, max(0, roi_x1):roi_x2]
        if roi.size < 100:
            return None

        # ── CLAHE + gamma correction (fix glare / windshield reflection) ─────
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        l_ch = clahe.apply(l_ch)
        roi_enhanced = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
        gamma_lut = np.array([((i / 255.0) ** 1.2) * 255 for i in range(256)], dtype=np.uint8)
        roi_enhanced = cv2.LUT(roi_enhanced, gamma_lut)

        # ── YOLOv11s classifier ───────────────────────────────────────────────
        if self._seatbelt_model is not None:
            try:
                results = self._seatbelt_model.predict(roi_enhanced, verbose=False)
                cls_idx  = int(results[0].probs.top1)
                cls_name = results[0].names[cls_idx]
                conf     = float(results[0].probs.top1conf)
                if cls_name == "no_seatbelt" and conf >= 0.55:
                    return ViolationResult.create(
                        ViolationType.SEATBELT_NON_COMPLIANCE,
                        confidence=round(conf, 4),
                        bbox=vehicle.bbox,
                        metadata={"method": "yolov11s_classifier", "cls": cls_name},
                    )
                return None   # seat_belt detected or low confidence
            except Exception as e:
                logger.warning("Seatbelt classifier inference failed: %s", e)

        # ── Hough-diagonal fallback (original method) ─────────────────────────
        gray  = cv2.cvtColor(roi_enhanced, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 25, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=20, maxLineGap=10)
        diagonal_count = 0
        if lines is not None:
            for line in lines:
                xa, ya, xb, yb = line[0]
                if xa == xb:
                    continue
                angle = abs(np.degrees(np.arctan2(yb - ya, xb - xa)))
                if 25 < angle < 65 or 115 < angle < 155:
                    diagonal_count += 1
        if diagonal_count >= 1:
            return None
        return ViolationResult.create(
            ViolationType.SEATBELT_NON_COMPLIANCE,
            confidence=0.60,
            bbox=vehicle.bbox,
            metadata={"method": "hough_fallback", "diagonal_lines_found": diagonal_count},
        )

    # ------------------------------------------------------------------
    # 8. Phone use while driving
    # ------------------------------------------------------------------

    def check_phone(
        self,
        vehicle: Detection,
        phone_detections: List[Detection],
    ) -> Optional[ViolationResult]:
        """
        Flag phone use when a cell-phone bbox overlaps the driver region of a 4-wheeler.
        Uses COCO class 67 detections from the main YOLOv8m run.
        Only applies to 4-wheelers (driver visible through windshield).
        """
        if not vehicle.is_four_wheeler:
            return None
        x1, y1, x2, y2 = map(int, vehicle.bbox)
        vh, vw = y2 - y1, x2 - x1
        # Driver region bbox: upper 60%, right 55% (India RHD)
        dr_x1 = x1 + int(vw * 0.45)
        dr_y1 = y1
        dr_x2 = x2
        dr_y2 = y1 + int(vh * 0.60)

        for phone in phone_detections:
            px1, py1, px2, py2 = phone.bbox
            # Check overlap with driver region
            if px2 > dr_x1 and px1 < dr_x2 and py2 > dr_y1 and py1 < dr_y2:
                return ViolationResult.create(
                    ViolationType.PHONE_USE,
                    confidence=min(0.95, phone.confidence),
                    bbox=vehicle.bbox,
                    metadata={"phone_bbox": [round(v, 1) for v in phone.bbox],
                              "phone_conf": round(phone.confidence, 3)},
                )
        return None

    # ------------------------------------------------------------------
    # 9. In-zone helper
    # ------------------------------------------------------------------

    def is_in_no_parking_zone(self, bbox: List[float]) -> bool:
        """Check if vehicle center falls within any configured no-parking zone"""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        for zone in self.parking_zones:
            if zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]:
                return True
        return False

    # ------------------------------------------------------------------
    # Batch check — convenience method
    # ------------------------------------------------------------------

    def check_all(
        self,
        image: np.ndarray,
        vehicles: List[Detection],
        persons: List[Detection],
        signal_frame: Optional[np.ndarray] = None,
        tracker_states: Optional[Dict] = None,
        phone_detections: Optional[List[Detection]] = None,
        signal_bbox: Optional[List[float]] = None,
    ) -> List[ViolationResult]:
        """
        Run all applicable violation checks for all vehicles in a frame.

        Parameters
        ----------
        image          : Preprocessed BGR frame
        vehicles       : Detected vehicle Detection objects
        persons        : Detected person Detection objects
        signal_frame   : Frame region containing traffic light (for signal detection)
        tracker_states : Dict of {track_id: TrackState} from VehicleTracker
        signal_bbox    : Optional calibrated [x1,y1,x2,y2] ROI for the traffic
                          light, in signal_frame coordinates. When omitted,
                          MLSignalStateDetector falls back to scanning the
                          generic top-40%-of-frame guess.

        Returns
        -------
        List of ViolationResult (may be empty)
        """
        results: List[ViolationResult] = []
        signal_state, signal_conf, light_detected = ("unknown", 0.0, False)
        phones = phone_detections or []

        if signal_frame is not None:
            signal_state, signal_conf, light_detected = self.signal_det.detect_with_presence(signal_frame, signal_bbox)

        # Pre-compute helmet results for all two-wheelers in batch
        two_wheelers = [v for v in vehicles if v.is_two_wheeler]
        helmet_results = {}
        if two_wheelers:
            if self.ai_helmet._model is None:
                # Fallback crop-based path: batch them!
                head_crops = []
                for vehicle in two_wheelers:
                    x1, y1, x2, y2 = map(int, vehicle.bbox)
                    head_h = int((y2 - y1) * 0.45)
                    head_crop = image[y1 : y1 + head_h, x1:x2]
                    head_crops.append(head_crop)
                
                batch_results = self.helmet_clf.classify_batch(head_crops)
                for vehicle, (has_helmet, conf) in zip(two_wheelers, batch_results):
                    # For fallback crop path, rider_count defaults to 1
                    helmet_results[id(vehicle)] = (not has_helmet, conf, 1)
            else:
                # Full-image model path: let the cached predict handle it
                # Pre-run full image prediction once to cache results
                self.ai_helmet._run_full_image(image)
                for vehicle in two_wheelers:
                    helmet_results[id(vehicle)] = self.ai_helmet.detect(image, vehicle)

        two_wheelers = [v for v in vehicles if v.is_two_wheeler]
        rider_map = self.associate_riders_with_vehicles(two_wheelers, persons)

        for vehicle in vehicles:
            tid = vehicle.track_id

            # --- 2-wheeler violations ---
            if vehicle.is_two_wheeler:
                ai_result = helmet_results.get(id(vehicle))

                v = self.check_helmet(image, vehicle, ai_result=ai_result)
                if v:
                    results.append(v)

                v = self.check_triple_riding(vehicle, persons, rider_map=rider_map, ai_result=ai_result)
                if v:
                    results.append(v)

            # --- 4-wheeler violations ---
            if vehicle.is_four_wheeler:
                v = self.check_seatbelt(image, vehicle)
                if v:
                    results.append(v)

                if phones:
                    v = self.check_phone(vehicle, phones)
                    if v:
                        results.append(v)

            # --- Track-based violations ---
            if tid is not None and tracker_states and tid in tracker_states:
                state = tracker_states[tid]
                history_bboxes = state.bboxes_in_window(20)
                vx, vy = state.velocity()
                is_stat = state.is_stationary()

                v = self.check_red_light(
                    history_bboxes, signal_state, signal_conf, vehicle.bbox, light_detected
                )
                if v:
                    results.append(v)

                v = self.check_stop_line(history_bboxes, signal_state, vehicle.bbox, light_detected)
                if v:
                    results.append(v)

                v = self.check_wrong_side((vx, vy), vehicle.bbox)
                if v:
                    results.append(v)

                in_zone = self.is_in_no_parking_zone(vehicle.bbox)
                v = self.check_illegal_parking(vehicle, tid, is_stat, in_zone)
                if v:
                    results.append(v)

            else:
                # --- Image-only (no tracker) fallbacks for PS compliance ---
                v = self._check_red_light_static(vehicle.bbox, signal_state, signal_conf, light_detected)
                if v:
                    results.append(v)

                v = self._check_stop_line_static(vehicle.bbox, signal_state, light_detected)
                if v:
                    results.append(v)

                v = self._check_wrong_side_static(vehicle.bbox)
                if v:
                    results.append(v)

                v = self._check_illegal_parking_static(vehicle.bbox)
                if v:
                    results.append(v)

        # Global confidence gate: a flagged violation only counts as a real
        # violation above 40% confidence. Below that, it's too uncertain to
        # cite — treat it as compliant (green), not a violation. License
        # plate OCR confidence is a separate concern and is untouched by
        # this gate.
        results = [r for r in results if r.confidence > self.MIN_VIOLATION_CONFIDENCE]

        return results

    # ------------------------------------------------------------------
    # Image-only (static frame) counterparts for track-based violations
    # ------------------------------------------------------------------

    def _check_red_light_static(
        self,
        bbox: List[float],
        signal_state: str,
        signal_conf: float,
        light_detected: bool = True,
    ) -> Optional[ViolationResult]:
        """Red-light: vehicle front past stop line while signal is definitively red."""
        if not light_detected:
            return None
        if signal_state.lower() != "red" or signal_conf < 0.65:
            return None
        if bbox[3] <= self.stop_line_y:
            return None
        return ViolationResult.create(
            ViolationType.RED_LIGHT_VIOLATION,
            confidence=min(0.92, signal_conf * 0.95),
            bbox=bbox,
            metadata={
                "signal_state": "red",
                "stop_line_y": self.stop_line_y,
                "signal_confidence": round(signal_conf, 3),
                "method": "static_position",
            },
        )

    def _check_stop_line_static(
        self,
        bbox: List[float],
        signal_state: str,
        light_detected: bool = True,
    ) -> Optional[ViolationResult]:
        """Stop-line: vehicle front past stop line while signal is red or yellow."""
        if not light_detected:
            return None
        if signal_state.lower() not in ("red", "yellow"):
            return None
        if bbox[3] <= self.stop_line_y:
            return None
        return ViolationResult.create(
            ViolationType.STOP_LINE_VIOLATION,
            confidence=0.80,
            bbox=bbox,
            metadata={
                "stop_line_y": self.stop_line_y,
                "signal": signal_state,
                "method": "static_position",
            },
        )

    def _check_wrong_side_static(
        self,
        bbox: List[float],
        frame_width: Optional[int] = None,
    ) -> Optional[ViolationResult]:
        """
        Wrong-side check for static images (no track/velocity available).

        Presence inside the camera's calibrated `wrong_side_zone` — the lane(s)
        reserved for oncoming traffic only — is itself the violation evidence
        in a single photo; no motion is needed to confirm it.
        """
        if not self.wrong_side_zone:
            return None

        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        in_zone = any(
            zx1 <= cx <= zx2 and zy1 <= cy <= zy2
            for zx1, zy1, zx2, zy2 in self.wrong_side_zone
        )
        if not in_zone:
            return None
        return ViolationResult.create(
            ViolationType.WRONG_SIDE_DRIVING,
            confidence=0.75,  # Calibrated zone presence — stronger evidence
            bbox=bbox,
            metadata={
                "centre_x": round(cx, 1),
                "centre_y": round(cy, 1),
                "method": "calibrated_zone",
            },
        )

    def _check_illegal_parking_static(
        self,
        bbox: List[float],
    ) -> Optional[ViolationResult]:
        """
        Illegal parking from a single image (no track history → no duration evidence).

        Zone presence alone can't distinguish a parked vehicle from one
        merely driving through a no-parking zone, so this is deliberately
        capped below the Tier-2 human-review threshold (0.60) — logged for
        audit, never auto-challaned or escalated, unlike the track-based
        check_illegal_parking() which confirms real stationary duration.
        """
        if not self.parking_zones:
            return None
        if not self.is_in_no_parking_zone(bbox):
            return None
        return ViolationResult.create(
            ViolationType.ILLEGAL_PARKING,
            confidence=0.55,
            bbox=bbox,
            metadata={"method": "static_zone_check", "unconfirmed_single_frame": True},
        )

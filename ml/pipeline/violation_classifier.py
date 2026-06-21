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
        return {
            "type": self.violation_type.value,
            "confidence": self.confidence,
            "severity": self.severity,
            "fine_amount_inr": self.fine_amount,
            "bbox": [round(v, 2) for v in self.bbox],
            "metadata": self.metadata,
        }


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

    _WEIGHTS = Path(__file__).parent.parent / "models" / "weights" / "helmet_best.pt"
    _NO_HELMET_CLASS = "head"    # exposed head = no helmet
    _HELMET_CLASS    = "helmet"

    def __init__(self, helmet_weights_path: Optional[str] = None) -> None:
        self._model = None
        self._class_names: List[str] = []
        self._fallback = HelmetClassifier(helmet_weights_path)
        # Cache full-image results so multiple vehicles share one inference call
        self._cached_image_id: Optional[int] = None
        self._cached_heads:    List[List[float]] = []   # boxes where head (no helmet) found
        self._cached_helmets:  List[List[float]] = []   # boxes where helmet found
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
            results = self._model.predict(image, conf=0.25, iou=0.45, verbose=False)
            for result in results:
                for box in result.boxes:
                    name = result.names[int(box.cls[0])]
                    xyxy = box.xyxy[0].tolist()
                    if name == self._NO_HELMET_CLASS:
                        self._cached_heads.append(xyxy)
                    elif name == self._HELMET_CLASS:
                        self._cached_helmets.append(xyxy)
        except Exception as e:
            logger.warning("Helmet full-image inference failed: %s", e)

    def _head_above_vehicle(self, head_box: List[float], veh_box: List[float]) -> bool:
        """True if head centre is horizontally inside vehicle and above vehicle's mid-line."""
        hx1, hy1, hx2, hy2 = head_box
        vx1, vy1, vx2, vy2 = veh_box
        hcx = (hx1 + hx2) / 2
        hcy = (hy1 + hy2) / 2
        if not (vx1 - 20 <= hcx <= vx2 + 20):
            return False
        return hcy <= (vy1 + vy2) / 2

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

            associated_heads = [h for h in self._cached_heads
                                if self._head_above_vehicle(h, veh_box)]
            rider_count = len(associated_heads) + len(
                [h for h in self._cached_helmets if self._head_above_vehicle(h, veh_box)]
            )

            if associated_heads:
                best_conf = 0.50
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

    _WEIGHTS = Path(__file__).parent.parent / "models" / "weights" / "traffic_lights_yolov8x.pt"

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
    helmet_weights_path: Optional path to trained helmet CNN weights
    """

    def __init__(
        self,
        stop_line_y: int = 400,
        parking_zones: Optional[List[List[int]]] = None,
        helmet_weights_path: Optional[str] = None,
    ) -> None:
        self.stop_line_y = stop_line_y
        self.parking_zones: List[List[int]] = parking_zones or []
        self.helmet_clf = HelmetClassifier(helmet_weights_path)
        self.ai_helmet = AIHelmetViolationDetector(helmet_weights_path)  # fallback uses same trained CNN
        self.signal_det = MLSignalStateDetector()
        self._parked_since: Dict[int, float] = {}

    # ------------------------------------------------------------------
    # 1. Helmet non-compliance
    # ------------------------------------------------------------------

    def check_helmet(
        self,
        image: np.ndarray,
        vehicle: Detection,
    ) -> Optional[ViolationResult]:
        """
        Check helmet on rider of a 2-wheeler.
        Uses YOLOv8n helmet_violation.pt when available (mAP@0.5=0.881),
        otherwise falls back to crop-based CNN + heuristic.
        """
        if not vehicle.is_two_wheeler:
            return None

        violation_found, conf, _ = self.ai_helmet.detect(image, vehicle)
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

    def check_triple_riding(
        self,
        vehicle: Detection,
        persons: List[Detection],
        image: Optional[np.ndarray] = None,
    ) -> Optional[ViolationResult]:
        """
        Count persons on a 2-wheeler.
        When AICity model is active, uses per-rider position detection (more accurate).
        Otherwise counts person bboxes overlapping the vehicle.
        """
        if not vehicle.is_two_wheeler:
            return None

        # AICity model path: rider_count comes from 9-class detection
        if image is not None and self.ai_helmet._model is not None:
            _, _, rider_count = self.ai_helmet.detect(image, vehicle)
        else:
            # Fallback: count persons whose center is inside vehicle bbox
            rider_count = sum(
                1 for p in persons if vehicle.contains_point(*p.center)
            )

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
    ) -> Optional[ViolationResult]:
        """
        Check if vehicle crossed stop line while signal is red.

        Parameters
        ----------
        track_bboxes : Last N bounding boxes from tracker history
        signal_state : Current signal colour
        signal_conf  : Confidence of signal detection
        latest_bbox  : Most recent bbox for the violation record
        """
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
    ) -> Optional[ViolationResult]:
        """
        Detect vehicle stopped over the stop line (not crossing, just encroaching).
        """
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

    def check_wrong_side(
        self,
        velocity: Tuple[float, float],
        latest_bbox: List[float],
        expected_vy_positive: bool = True,  # True = traffic flows top→bottom
    ) -> Optional[ViolationResult]:
        """
        Detect wrong-side driving from velocity direction.

        Parameters
        ----------
        velocity           : (vx, vy) from tracker
        expected_vy_positive: If True, normal traffic should move downward (vy > 0)
        """
        vx, vy = velocity
        speed = (vx ** 2 + vy ** 2) ** 0.5

        if speed < 3.0:  # Too slow / stationary
            return None

        is_wrong = (expected_vy_positive and vy < -5) or (
            not expected_vy_positive and vy > 5
        )

        if is_wrong:
            conf = min(0.90, 0.55 + speed / 80)
            return ViolationResult.create(
                ViolationType.WRONG_SIDE_DRIVING,
                confidence=conf,
                bbox=latest_bbox,
                metadata={"velocity_vx": round(vx, 2), "velocity_vy": round(vy, 2)},
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
        Detect missing seatbelt via Hough diagonal line detection in driver region.
        Seatbelts appear as diagonal lines (30-60°) crossing the driver's torso.

        NOTE: This is inherently imprecise from a distance camera.
        Returns conservative confidence (≤0.68) → always Tier 2.
        """
        if not vehicle.is_four_wheeler:
            return None

        x1, y1, x2, y2 = map(int, vehicle.bbox)
        vh, vw = y2 - y1, x2 - x1

        # Minimum size: small detections are likely misclassified 3-wheelers or background vehicles
        if vw < 110 or vh < 80:
            return None

        # Aspect ratio: enclosed cars/SUVs are wider than tall.
        # Auto-rickshaws and misclassified 3-wheelers tend to be square/tall.
        if vw < vh * 0.75:
            return None

        # Driver region: upper 70%, right 55% of car (India RHD — driver sits on right)
        driver_region = image[y1 : y1 + int(vh * 0.70), x1 + int(vw * 0.45) : x2]
        if driver_region.size < 100:
            return None

        gray = cv2.cvtColor(driver_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 25, 100)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=20, minLineLength=20, maxLineGap=10
        )

        diagonal_count = 0
        if lines is not None:
            for line in lines:
                xa, ya, xb, yb = line[0]
                if xa == xb:
                    continue
                angle = abs(np.degrees(np.arctan2(yb - ya, xb - xa)))
                # Seatbelt diagonal: 25°-65° or 115°-155°
                if 25 < angle < 65 or 115 < angle < 155:
                    diagonal_count += 1

        # Need zero diagonals to confidently flag absence — even 1 diagonal line means belt present
        if diagonal_count >= 1:
            return None  # Seatbelt likely present

        # No diagonal found → possible no seatbelt
        return ViolationResult.create(
            ViolationType.SEATBELT_NON_COMPLIANCE,
            confidence=0.62,   # Conservative → always Tier 2
            bbox=vehicle.bbox,
            metadata={
                "method": "hough_diagonal",
                "diagonal_lines_found": diagonal_count,
                "always_tier2": True,
            },
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

        Returns
        -------
        List of ViolationResult (may be empty)
        """
        results: List[ViolationResult] = []
        signal_state, signal_conf = ("unknown", 0.0)
        phones = phone_detections or []

        if signal_frame is not None:
            signal_state, signal_conf = self.signal_det.detect(signal_frame)

        for vehicle in vehicles:
            tid = vehicle.track_id

            # --- 2-wheeler violations ---
            if vehicle.is_two_wheeler:
                v = self.check_helmet(image, vehicle)
                if v:
                    results.append(v)

                v = self.check_triple_riding(vehicle, persons, image)
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
                    history_bboxes, signal_state, signal_conf, vehicle.bbox
                )
                if v:
                    results.append(v)

                v = self.check_stop_line(history_bboxes, signal_state, vehicle.bbox)
                if v:
                    results.append(v)

                v = self.check_wrong_side((vx, vy), vehicle.bbox)
                if v:
                    results.append(v)

                in_zone = self.is_in_no_parking_zone(vehicle.bbox)
                v = self.check_illegal_parking(vehicle, tid, is_stat, in_zone)
                if v:
                    results.append(v)

        return results

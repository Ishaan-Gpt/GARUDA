"""
GARUDA ML Pipeline — Driver State Detector
============================================
Pre-violation predictive detection using:
  - MediaPipe FaceMesh (468 landmarks) → Eye Aspect Ratio (drowsiness)
  - MediaPipe FaceMesh → Mouth Aspect Ratio (yawning)
  - Phone use is detected from the shared yolov8m class-67 pass
    (VehicleDetector), not a second model — see enable_phone_detection.

Alerts fire BEFORE a violation occurs, enabling:
  → Flash roadside LED board
  → Dispatch nearby patrol officer
  → Log fatigue event for insurance/analytics
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FaceMesh landmark indices (MediaPipe 468-point model)
# ---------------------------------------------------------------------------

# 6-point EAR landmarks per eye
LEFT_EYE_UPPER  = [386, 387, 388]
LEFT_EYE_LOWER  = [374, 373, 390]
LEFT_EYE_CORNER = [362, 263]

RIGHT_EYE_UPPER  = [159, 158, 157]
RIGHT_EYE_LOWER  = [145, 144, 153]
RIGHT_EYE_CORNER = [33, 133]

# Mouth aspect ratio landmarks
MOUTH_TOP    = [13]
MOUTH_BOTTOM = [14]
MOUTH_LEFT   = [61]
MOUTH_RIGHT  = [291]

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

EAR_DROWSY_THRESHOLD = 0.25   # Below = eye closing / heavy blinking
MAR_YAWN_THRESHOLD   = 0.65   # Above = yawning
DROWSY_FRAMES        = 45     # ~1.5 seconds at 30 fps
YAWN_FRAMES          = 20     # ~0.67 seconds at 30 fps


# ---------------------------------------------------------------------------
# Alert result
# ---------------------------------------------------------------------------

@dataclass
class DriverAlert:
    alert_type: str        # "DROWSY_DRIVER" | "YAWNING_DETECTED" | "PHONE_USE"
    severity: str          # "critical" | "high" | "medium"
    action: str            # "FLASH_ROADSIDE_BOARD+ALERT_PATROL" | etc.
    confidence: float
    track_id: Optional[int] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "alert_type": self.alert_type,
            "severity": self.severity,
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "track_id": self.track_id,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# EAR / MAR helpers
# ---------------------------------------------------------------------------

def _ear(landmarks: np.ndarray,
         upper_idx: List[int],
         lower_idx: List[int],
         corner_idx: List[int]) -> float:
    """
    Eye Aspect Ratio = (|p1-p5| + |p2-p4|) / (2 * |p0-p3|)
    Falls below ~0.25 when eye closes.
    """
    if len(landmarks) == 0:
        return 0.30

    try:
        upper = np.mean([landmarks[i] for i in upper_idx], axis=0)
        lower = np.mean([landmarks[i] for i in lower_idx], axis=0)
        left  = landmarks[corner_idx[0]]
        right = landmarks[corner_idx[1]]

        vertical   = np.linalg.norm(upper - lower)
        horizontal = np.linalg.norm(left  - right)

        if horizontal < 1e-6:
            return 0.30
        return float(vertical / horizontal)

    except (IndexError, ValueError):
        return 0.30


def _mar(landmarks: np.ndarray) -> float:
    """
    Mouth Aspect Ratio = vertical / horizontal mouth opening.
    Above ~0.65 indicates yawning.
    """
    try:
        top    = landmarks[MOUTH_TOP[0]]
        bottom = landmarks[MOUTH_BOTTOM[0]]
        left   = landmarks[MOUTH_LEFT[0]]
        right  = landmarks[MOUTH_RIGHT[0]]

        vertical   = np.linalg.norm(top - bottom)
        horizontal = np.linalg.norm(left - right)

        if horizontal < 1e-6:
            return 0.0
        return float(vertical / horizontal)

    except (IndexError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class DriverStateDetector:
    """
    Real-time driver state analysis.

    Parameters
    ----------
    phone_model_path : Path to fine-tuned phone detector (.pt / .engine).
                       Falls back to general YOLO class 67 if not provided.
    max_num_faces    : Maximum faces to analyse per frame (keep low for speed)
    """

    def __init__(
        self,
        phone_model_path: Optional[str] = None,
        max_num_faces: int = 3,
        enable_phone_detection: bool = False,
    ) -> None:
        self.max_num_faces = max_num_faces
        self._face_mesh   = None
        self._phone_model = None
        self._mp_ok       = False
        self._ph_ok       = False

        self._drowsy_ctr: Dict[int, int] = defaultdict(int)
        self._yawn_ctr:   Dict[int, int] = defaultdict(int)

        self._init_face_mesh()
        if enable_phone_detection:
            # Skip this when the caller already runs a traffic-class YOLO model
            # that includes COCO class 67 (cell phone) — e.g. VehicleDetector
            # with its default class set — to avoid loading a second redundant
            # detector. Pass those detections straight to analyze_frame's
            # phone_detections argument instead.
            self._init_phone_detector(phone_model_path)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_face_mesh(self) -> None:
        # mediapipe >=0.10.x dropped the legacy `mp.solutions.face_mesh` API in
        # favour of the Tasks API (FaceLandmarker + a downloadable .task model
        # bundle). Landmark indexing (468-point topology) is unchanged, so the
        # EAR/MAR index constants below still apply.
        try:
            import mediapipe as mp  # type: ignore
            from mediapipe.tasks.python import vision as mp_vision  # type: ignore
            from mediapipe.tasks.python.core.base_options import BaseOptions  # type: ignore

            model_path = Path(__file__).parent.parent / "models" / "weights" / "face_landmarker.task"
            if not model_path.exists():
                logger.warning(
                    "face_landmarker.task not found at %s — driver state detection disabled. "
                    "Download from https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                    "face_landmarker/float16/1/face_landmarker.task",
                    model_path,
                )
                return

            options = mp_vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=self.max_num_faces,
                min_face_detection_confidence=0.60,
                min_tracking_confidence=0.55,
            )
            self._mp = mp
            self._face_mesh = mp_vision.FaceLandmarker.create_from_options(options)
            self._mp_ok = True
            logger.info("MediaPipe FaceLandmarker initialised (faces=%d)", self.max_num_faces)

        except ImportError:
            logger.warning(
                "MediaPipe not installed — driver state detection disabled. "
                "Install with: pip install mediapipe"
            )
        except Exception as e:
            logger.warning("FaceLandmarker init failed (%s) — driver state detection disabled.", e)

    def _init_phone_detector(self, model_path: Optional[str]) -> None:
        try:
            from ultralytics import YOLO  # type: ignore

            if model_path:
                self._phone_model = YOLO(model_path)
                logger.info("Phone detector loaded: %s", model_path)
            else:
                self._phone_model = YOLO("yolo11n.pt")
                logger.info("Phone detector: using general YOLO (class 67 = cell phone)")
            self._ph_ok = True

        except Exception as e:
            logger.warning("Phone detector unavailable: %s", e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_frame(
        self,
        frame: np.ndarray,
        track_id: int = 0,
        driver_region: Optional[np.ndarray] = None,
    ) -> List[DriverAlert]:
        """
        Analyse a full frame (or optional driver crop) for state alerts.

        Parameters
        ----------
        frame         : Full BGR camera frame
        track_id      : Vehicle track ID for persistent counter state
        driver_region : Optional pre-cropped driver/windshield region

        Returns
        -------
        List of DriverAlert (empty = driver state OK)
        """
        alerts: List[DriverAlert] = []
        analysis_region = driver_region if driver_region is not None else frame

        # --- Face mesh (drowsiness + yawn) ---
        if self._mp_ok and analysis_region.size > 0:
            face_alerts = self._analyse_face(analysis_region, track_id)
            alerts.extend(face_alerts)

        # --- Phone detection ---
        if self._ph_ok:
            phone_alerts = self._detect_phone(frame, track_id)
            alerts.extend(phone_alerts)

        return alerts

    def analyze_batch(
        self,
        frame: np.ndarray,
        driver_crops: List[Tuple[int, np.ndarray]],
    ) -> Dict[int, List[DriverAlert]]:
        """
        Analyse multiple driver crops in one call.

        Parameters
        ----------
        driver_crops : List of (track_id, crop_image) tuples

        Returns
        -------
        Dict {track_id: [DriverAlert, ...]}
        """
        results: Dict[int, List[DriverAlert]] = {}
        for tid, crop in driver_crops:
            results[tid] = self.analyze_frame(frame, track_id=tid, driver_region=crop)
        return results

    def reset_track(self, track_id: int) -> None:
        """Reset counters for a track that has left the scene"""
        self._drowsy_ctr.pop(track_id, None)
        self._yawn_ctr.pop(track_id, None)

    # ------------------------------------------------------------------
    # Internal: face mesh analysis
    # ------------------------------------------------------------------

    def _analyse_face(
        self,
        image: np.ndarray,
        track_id: int,
    ) -> List[DriverAlert]:
        alerts: List[DriverAlert] = []
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        results = self._face_mesh.detect(mp_image)

        if not results.face_landmarks:
            # No face visible — reset counters
            self._drowsy_ctr[track_id] = max(0, self._drowsy_ctr[track_id] - 2)
            self._yawn_ctr[track_id]   = max(0, self._yawn_ctr[track_id]   - 2)
            return alerts

        h, w = image.shape[:2]

        for face_lm in results.face_landmarks:
            landmarks = np.array(
                [(lm.x * w, lm.y * h) for lm in face_lm],
                dtype=np.float32,
            )

            # --- EAR (drowsiness) ---
            left_ear  = _ear(landmarks, LEFT_EYE_UPPER,  LEFT_EYE_LOWER,  LEFT_EYE_CORNER)
            right_ear = _ear(landmarks, RIGHT_EYE_UPPER, RIGHT_EYE_LOWER, RIGHT_EYE_CORNER)
            avg_ear   = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_DROWSY_THRESHOLD:
                self._drowsy_ctr[track_id] += 1
                if self._drowsy_ctr[track_id] >= DROWSY_FRAMES:
                    conf = min(0.96, 0.70 + self._drowsy_ctr[track_id] / 300)
                    alerts.append(DriverAlert(
                        alert_type="DROWSY_DRIVER",
                        severity="critical",
                        action="FLASH_ROADSIDE_BOARD+ALERT_PATROL",
                        confidence=conf,
                        track_id=track_id,
                        metadata={
                            "ear": round(avg_ear, 3),
                            "frames_closed": self._drowsy_ctr[track_id],
                            "duration_sec": round(self._drowsy_ctr[track_id] / 30, 1),
                        },
                    ))
            else:
                self._drowsy_ctr[track_id] = max(0, self._drowsy_ctr[track_id] - 1)

            # --- MAR (yawning) ---
            mar = _mar(landmarks)
            if mar > MAR_YAWN_THRESHOLD:
                self._yawn_ctr[track_id] += 1
                if self._yawn_ctr[track_id] >= YAWN_FRAMES:
                    conf = min(0.88, 0.60 + mar * 0.35)
                    alerts.append(DriverAlert(
                        alert_type="YAWNING_DETECTED",
                        severity="medium",
                        action="LOG_FATIGUE_EVENT",
                        confidence=conf,
                        track_id=track_id,
                        metadata={"mar": round(mar, 3)},
                    ))
            else:
                self._yawn_ctr[track_id] = max(0, self._yawn_ctr[track_id] - 1)

            break  # Analyse only the first / most prominent face

        return alerts

    # ------------------------------------------------------------------
    # Internal: phone detection
    # ------------------------------------------------------------------

    def _detect_phone(
        self,
        frame: np.ndarray,
        track_id: int,
    ) -> List[DriverAlert]:
        alerts: List[DriverAlert] = []
        try:
            results = self._phone_model.predict(
                frame,
                conf=0.55,
                classes=[67],   # COCO class 67 = cell phone
                verbose=False,
            )
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()
                    alerts.append(DriverAlert(
                        alert_type="PHONE_USE_WHILE_DRIVING",
                        severity="high",
                        action="AUTO_CHALLAN+LOG",
                        confidence=conf,
                        track_id=track_id,
                        metadata={"phone_bbox": [round(v, 1) for v in bbox]},
                    ))
        except Exception as e:
            logger.debug("Phone detection error: %s", e)

        return alerts

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> dict:
        return {
            "face_mesh_available": self._mp_ok,
            "phone_detection_available": self._ph_ok,
            "drowsiness_threshold_ear": EAR_DROWSY_THRESHOLD,
            "yawn_threshold_mar": MAR_YAWN_THRESHOLD,
            "alert_after_frames": DROWSY_FRAMES,
        }

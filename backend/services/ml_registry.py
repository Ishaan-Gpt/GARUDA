"""GARUDA ML Registry — single shared ML pipeline singleton.

Import ``get_ml_registry()`` anywhere in the backend and you always get the
same pre-loaded detector, OCR, classifier, and preprocessor instances.
This eliminates the previous bug where ``_routers.py`` and ``jobs.py`` each
created their own independent model instances.

Usage::

    from backend.services.ml_registry import get_ml_registry

    registry = get_ml_registry()
    if registry.available:
        detections = registry.detector.detect(frame)
        violations = registry.classifier.check_all(...)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolved weight paths — all in one authoritative place
# ---------------------------------------------------------------------------

_WEIGHTS_ROOT = Path(__file__).parent.parent.parent / "ml" / "models" / "weights"

WEIGHTS = {
    # primary vehicle / person detector
    "detector":        _WEIGHTS_ROOT / "detection"   / "yolov8m.pt",
    # helmet violation classifier (YOLO-based best checkpoint)
    "helmet_yolo":     _WEIGHTS_ROOT / "violations"  / "helmet_best.pt",
    # helmet CNN classifier (lighter, used by ViolationClassifier)
    "helmet_cnn":      _WEIGHTS_ROOT / "violations"  / "helmet_cnn.pt",
    # seatbelt binary classifier
    "seatbelt":        _WEIGHTS_ROOT / "violations"  / "seatbelt_classifier.pt",
    # traffic-light state detector
    "traffic_lights":  _WEIGHTS_ROOT / "violations"  / "traffic_lights_yolov8x.pt",
    # plate detector — stage-1 (Koushi)
    "plate_stage1":    _WEIGHTS_ROOT / "ocr"         / "plate_koushi.pt",
    # plate detector — stage-2 fallback (YasirFaiz), auto-loaded by PlateOCR
    "plate_stage2":    _WEIGHTS_ROOT / "ocr"         / "plate_yasir.pt",
    # plate detector — moin's YOLOv8 (second fallback)
    "plate_moin":      _WEIGHTS_ROOT / "ocr"         / "plate_yolov8_moin.pt",
}


def _best_plate_weight() -> Optional[str]:
    """Return the first existing plate-detector weight path, in priority order."""
    for key in ("plate_stage1", "plate_moin"):
        p = WEIGHTS[key]
        if p.exists():
            return str(p)
    return None


def _resolve(key: str) -> Optional[str]:
    p = WEIGHTS.get(key)
    return str(p) if p and p.exists() else None


# ---------------------------------------------------------------------------
# Registry dataclass
# ---------------------------------------------------------------------------

@dataclass
class MLRegistry:
    """Container for all loaded ML pipeline components.

    Attributes
    ----------
    available : bool
        ``True`` when all components loaded without error.
    preprocessor : Any
        ``ImagePreprocessor`` instance, or ``None``.
    detector : Any
        ``VehicleDetector`` instance, or ``None``.
    ocr : Any
        ``PlateOCR`` instance, or ``None``.
    classifier : Any
        ``ViolationClassifier`` instance (default calibration).
        Calibration values are OVERWRITTEN per-job by ``CalibrationService``
        before every inference call — do NOT hold a reference to the values
        set here.
    driver_state : Any
        ``DriverStateDetector`` instance, or ``None``.
    error : str
        Error message if loading failed.
    """

    available: bool = False
    preprocessor: Any = None
    detector: Any = None
    ocr: Any = None
    classifier: Any = None
    driver_state: Any = None
    visualizer: Any = None
    error: str = ""


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: Optional[MLRegistry] = None


def get_ml_registry() -> MLRegistry:
    """Return the module-level ``MLRegistry`` singleton, initialising on first call."""
    global _registry
    if _registry is None:
        _registry = _load_registry()
    return _registry


def _load_registry() -> MLRegistry:
    reg = MLRegistry()
    try:
        from ml.pipeline.preprocessor import ImagePreprocessor
        from ml.pipeline.detector import VehicleDetector
        from ml.pipeline.ocr import PlateOCR
        from ml.pipeline.violation_classifier import ViolationClassifier
        from ml.pipeline.driver_state import DriverStateDetector

        helmet_path = _resolve("helmet_cnn")
        plate_path  = _best_plate_weight()

        logger.info(
            "MLRegistry: loading detector=%s | helmet=%s | plate=%s",
            _resolve("detector") or "bundled",
            helmet_path,
            plate_path,
        )

        from ml.utils.visualizer import FrameVisualizer

        reg.preprocessor  = ImagePreprocessor()
        reg.detector      = VehicleDetector(model_path=None, device="cpu")
        reg.ocr           = PlateOCR(plate_detector_weights=plate_path)
        reg.classifier    = ViolationClassifier(
            stop_line_y=380,
            helmet_weights_path=helmet_path,
        )
        reg.driver_state  = DriverStateDetector()
        reg.visualizer    = FrameVisualizer()
        reg.available     = True
        logger.info("MLRegistry: all components loaded successfully.")

    except Exception as exc:
        reg.available = False
        reg.error     = str(exc)
        logger.error("MLRegistry: failed to initialise — %s", exc, exc_info=True)

    return reg

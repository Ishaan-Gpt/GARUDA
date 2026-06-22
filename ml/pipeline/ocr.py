"""
GARUDA ML Pipeline — License Plate OCR
========================================
Primary  : PaddleOCR (best accuracy for Indian plates)
Fallback : EasyOCR  (if paddle not installed)
Emergency: Tesseract via pytesseract (last resort)

Handles Indian plate formats:
  Standard KA : KA-01-AB-1234
  Old format  : KA-01-1234
  BH Series   : 22-BH-1234-AA
  Military    : 11-A-1234
"""
from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Indian state codes
# ---------------------------------------------------------------------------

STATE_CODES: Dict[str, str] = {
    "AN": "Andaman & Nicobar", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh",
    "AS": "Assam", "BR": "Bihar", "CG": "Chhattisgarh", "CH": "Chandigarh",
    "DD": "Daman & Diu", "DL": "Delhi", "DN": "Dadra & Nagar Haveli",
    "GA": "Goa", "GJ": "Gujarat", "HP": "Himachal Pradesh", "HR": "Haryana",
    "JH": "Jharkhand", "JK": "Jammu & Kashmir", "KA": "Karnataka",
    "KL": "Kerala", "LA": "Ladakh", "MH": "Maharashtra", "ML": "Meghalaya",
    "MN": "Manipur", "MP": "Madhya Pradesh", "MZ": "Mizoram", "NL": "Nagaland",
    "OD": "Odisha", "PB": "Punjab", "PY": "Puducherry", "RJ": "Rajasthan",
    "SK": "Sikkim", "TN": "Tamil Nadu", "TR": "Tripura", "TS": "Telangana",
    "UK": "Uttarakhand", "UP": "Uttar Pradesh", "WB": "West Bengal",
}

# Regex patterns for Indian plates (most specific first)
PLATE_PATTERNS: List[re.Pattern] = [
    # Standard: KA-01-AB-1234
    re.compile(r'[A-Z]{2}[\s\-]?\d{2}[\s\-]?[A-Z]{1,3}[\s\-]?\d{4}'),
    # Old two-letter: KA-01-1234
    re.compile(r'[A-Z]{2}[\s\-]?\d{2}[\s\-]?\d{4}'),
    # BH series: 22-BH-1234-AA
    re.compile(r'\d{2}[\s\-]?BH[\s\-]?\d{4}[\s\-]?[A-Z]{2}'),
]

# Common OCR character substitutions for license plates
OCR_CORRECTIONS: Dict[str, str] = {
    "O": "0", "o": "0",   # Letter O/o → digit 0
    "I": "1", "i": "1",   # Letter I/i → digit 1
    "l": "1", "L": "1",   # Letter l/L → digit 1
    "B": "8", "b": "8",   # Letter B/b → digit 8
    "S": "5", "s": "5",   # Letter S/s → digit 5
    "Z": "2", "z": "2",   # Letter Z/z → digit 2
    "G": "6", "g": "6",   # Letter G/g → digit 6
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class PlateResult:
    """Structured license plate OCR result"""

    def __init__(
        self,
        raw_text: str = "",
        formatted_text: str = "",
        confidence: float = 0.0,
        state_code: str = "",
        state_name: str = "Unknown",
        is_valid: bool = False,
        ocr_engine: str = "none",
    ) -> None:
        self.raw_text = raw_text
        self.formatted_text = formatted_text
        self.confidence = confidence
        self.state_code = state_code
        self.state_name = state_name
        self.is_valid = is_valid
        self.ocr_engine = ocr_engine

    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "formatted_text": self.formatted_text,
            "confidence": round(self.confidence, 4),
            "state": self.state_name,
            "state_code": self.state_code,
            "is_valid": self.is_valid,
            "ocr_engine": self.ocr_engine,
        }

    def __bool__(self) -> bool:
        return bool(self.formatted_text or self.raw_text)

    def __repr__(self) -> str:
        return f"PlateResult(text='{self.formatted_text}' conf={self.confidence:.2f} valid={self.is_valid})"


# ---------------------------------------------------------------------------
# Main OCR class
# ---------------------------------------------------------------------------

class PlateOCR:
    """
    License plate OCR with automatic engine fallback.

    Engine priority: PaddleOCR → EasyOCR → Tesseract → Heuristic only

    Plate detection uses a 2-stage YOLO pipeline:
      Stage 1 — Koushi (plate_koushi.pt): locates plate bbox in vehicle crop
      Stage 2 — YasirFaiz (plate_yasir.pt): confirms on the Stage 1 crop
      Detections below MIN_PLATE_CONF are dropped (eliminates headlight FP)

    Parameters
    ----------
    use_gpu : Enable GPU acceleration (auto-disabled if CUDA unavailable)
    plate_detector_weights : Optional override for Stage 1 model path
    """

    # 2-stage model paths (auto-resolved relative to this file)
    _WEIGHTS_DIR   = Path(__file__).parent.parent / "models" / "weights" / "ocr"
    _STAGE1_WEIGHTS = _WEIGHTS_DIR / "plate_koushi.pt"   # Koushi — better spatial coverage
    _STAGE2_WEIGHTS = _WEIGHTS_DIR / "plate_yasir.pt"    # YasirFaiz — refines on crop
    _STAGE1_CONF   = 0.05    # low to catch all candidates
    _STAGE2_CONF   = 0.05    # low — we take max(stage1, stage2)
    _MIN_PLATE_CONF = 0.15   # final gate — drops headlight / windshield false positives

    def __init__(self, use_gpu: bool = False, plate_detector_weights: Optional[str] = None) -> None:
        self.use_gpu = use_gpu
        self._engine_name: str = "none"
        self._ocr = None
        self._plate_detector  = None   # Stage 1
        self._stage2_detector = None   # Stage 2
        self._init_engine()
        self._load_plate_detectors(plate_detector_weights)

    def _load_plate_detectors(self, stage1_override: Optional[str] = None) -> None:
        """Load Stage-1 (Koushi) and Stage-2 (YasirFaiz) plate detectors."""
        try:
            from ultralytics import YOLO  # type: ignore

            # Stage 1 — Koushi
            s1_path = stage1_override or (str(self._STAGE1_WEIGHTS) if self._STAGE1_WEIGHTS.exists() else None)
            if s1_path:
                self._plate_detector = YOLO(s1_path)
                logger.info("Plate Stage-1 (Koushi) loaded: %s", Path(s1_path).name)
            else:
                logger.warning("plate_koushi.pt not found — falling back to contour heuristic")

            # Stage 2 — YasirFaiz (optional, used for crop confirmation)
            if self._STAGE2_WEIGHTS.exists():
                self._stage2_detector = YOLO(str(self._STAGE2_WEIGHTS))
                logger.info("Plate Stage-2 (YasirFaiz) loaded: %s", self._STAGE2_WEIGHTS.name)
        except Exception as e:
            logger.warning("Could not load plate detector weights (%s). Using contour heuristic.", e)

    def _load_plate_detector(self, weights_path: str) -> None:
        """Legacy single-model loader kept for backwards compatibility."""
        self._load_plate_detectors(stage1_override=weights_path)

    # ------------------------------------------------------------------
    # Engine initialisation
    # ------------------------------------------------------------------

    def _init_engine(self) -> None:
        if self._try_fast_plate_ocr():
            return
        if self._try_paddle():
            return
        if self._try_easyocr():
            return
        if self._try_tesseract():
            return
        logger.error(
            "No OCR engine found. Install: pip install 'fast-plate-ocr[onnx]'"
        )

    def _try_fast_plate_ocr(self) -> bool:
        try:
            from fast_plate_ocr import LicensePlateRecognizer  # type: ignore
            self._ocr = LicensePlateRecognizer("cct-s-v2-global-model")
            self._engine_name = "fast_plate_ocr"
            logger.info("OCR engine: fast-plate-ocr (cct-s-v2-global-model)")
            return True
        except (ImportError, Exception) as e:
            logger.debug("fast-plate-ocr unavailable: %s", e)
            return False

    def _try_paddle(self) -> bool:
        try:
            from paddleocr import PaddleOCR  # type: ignore
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                use_gpu=self.use_gpu,
                show_log=False,
                rec_algorithm="CRNN",
                det_db_thresh=0.3,
                det_db_box_thresh=0.5,
                rec_image_shape="3, 32, 320",
            )
            self._engine_name = "paddle"
            logger.info("OCR engine: PaddleOCR")
            return True
        except (ImportError, Exception) as e:
            logger.debug("PaddleOCR unavailable: %s", e)
            return False

    def _try_easyocr(self) -> bool:
        try:
            import easyocr  # type: ignore
            self._ocr = easyocr.Reader(["en"], gpu=self.use_gpu, verbose=False)
            self._engine_name = "easyocr"
            logger.info("OCR engine: EasyOCR (PaddleOCR recommended for better accuracy)")
            return True
        except (ImportError, Exception) as e:
            logger.debug("EasyOCR unavailable: %s", e)
            return False

    def _try_tesseract(self) -> bool:
        try:
            import pytesseract  # type: ignore
            import os
            tess_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(tess_path):
                pytesseract.pytesseract.tesseract_cmd = tess_path
            pytesseract.get_tesseract_version()
            self._ocr = pytesseract
            self._engine_name = "tesseract"
            logger.info("OCR engine: Tesseract (configured with local path)")
            return True
        except Exception as e:
            logger.debug("Tesseract unavailable: %s", e)
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_plate(self, plate_image: np.ndarray) -> PlateResult:
        """
        Extract license plate text from a cropped plate image.

        Parameters
        ----------
        plate_image : BGR numpy array of the plate region

        Returns
        -------
        PlateResult with raw text, formatted text, confidence, state info
        """
        if plate_image is None or plate_image.size == 0:
            return PlateResult()

        raw_text = ""
        confidence = 0.0

        if self._ocr is not None:
            if self._engine_name == "fast_plate_ocr":
                # Try raw color crop first (modern OCR works best without binarization)
                raw_text, confidence = self._fast_plate_ocr_read(plate_image)
                if not raw_text or confidence < 0.35:
                    enhanced = self._enhance_for_ocr(plate_image)
                    raw_text2, confidence2 = self._fast_plate_ocr_read(enhanced)
                    if confidence2 > confidence:
                        raw_text, confidence = raw_text2, confidence2
            elif self._engine_name == "easyocr":
                # Try raw color crop first
                raw_text, confidence = self._easyocr_ocr(plate_image)
                if not raw_text or confidence < 0.35:
                    enhanced = self._enhance_for_ocr(plate_image)
                    raw_text2, confidence2 = self._easyocr_ocr(enhanced)
                    if confidence2 > confidence:
                        raw_text, confidence = raw_text2, confidence2
            else:
                # PaddleOCR/Tesseract: try raw color first
                raw_text, confidence = self._extract_text(plate_image)
                if not raw_text or confidence < 0.35:
                    enhanced = self._enhance_for_ocr(plate_image)
                    raw_text2, confidence2 = self._extract_text(enhanced)
                    if confidence2 > confidence:
                        raw_text, confidence = raw_text2, confidence2
                    if not raw_text or confidence < 0.35:
                        raw_text3, confidence3 = self._extract_text(cv2.bitwise_not(enhanced))
                        if confidence3 > confidence:
                            raw_text, confidence = raw_text3, confidence3

        # Parse text using robust Indian plate rules (sliding window state search)
        formatted, is_valid = self._parse_plate(raw_text)
        state_code = formatted[:2] if len(formatted) >= 2 else ""
        state_name = STATE_CODES.get(state_code, "Unknown")

        return PlateResult(
            raw_text=raw_text,
            formatted_text=formatted,
            confidence=confidence,
            state_code=state_code,
            state_name=state_name,
            is_valid=is_valid,
            ocr_engine=self._engine_name if self._ocr is not None else "none",
        )

    def read_plate_from_vehicle(
        self,
        vehicle_crop: np.ndarray,
    ) -> "PlateResult":
        """
        Read license plate from a full vehicle crop.

        For EasyOCR: scans the entire vehicle crop for ALL text regions,
        then picks the one that best matches Indian plate patterns.
        Falls back to read_plate(detect_plate_region(...)) otherwise.

        Parameters
        ----------
        vehicle_crop : BGR numpy array of the entire vehicle bounding box crop
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return PlateResult()

        if self._engine_name == "easyocr" and self._ocr is not None:
            try:
                h, w = vehicle_crop.shape[:2]

                # Build list of image candidates to try — upscale small crops aggressively
                candidates = []

                # Full vehicle crop (upscaled to min 150px height)
                if h < 150:
                    scale = max(150.0 / h, 200.0 / max(w, 1))
                    up = cv2.resize(vehicle_crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
                    candidates.append(up)
                else:
                    candidates.append(vehicle_crop)

                # Bottom 45% of vehicle crop (where plate is) — upscaled separately
                plate_strip = vehicle_crop[int(h * 0.55):, :]
                if plate_strip.size > 0:
                    ph, pw = plate_strip.shape[:2]
                    if ph > 5 and pw > 10:
                        ps = max(100.0 / max(ph, 1), 3.0)
                        plate_up = cv2.resize(plate_strip, (int(pw * ps), int(ph * ps)), interpolation=cv2.INTER_CUBIC)
                        candidates.append(plate_up)

                best_text = ""
                best_conf = 0.0

                for img_candidate in candidates:
                    results = self._ocr.readtext(img_candidate)
                    for (bbox_pts, text, conf) in results:
                        text_clean = re.sub(r"[^A-Z0-9\s\-]", "", text.upper().strip())
                        if len(text_clean) < 4:
                            continue
                        # Priority: exact plate pattern match
                        for pattern in PLATE_PATTERNS:
                            if pattern.search(text_clean):
                                if conf > best_conf:
                                    best_conf = conf
                                    best_text = text_clean
                                break
                        else:
                            # Accept 6-12 char alphanumeric at lower confidence threshold
                            no_sp = re.sub(r"[\s\-]", "", text_clean)
                            if 6 <= len(no_sp) <= 12 and conf > best_conf and conf > 0.15:
                                best_conf = conf
                                best_text = text_clean

                if best_text and best_conf > 0.15:
                    formatted, is_valid = self._parse_plate(best_text)
                    state_code = formatted[:2] if len(formatted) >= 2 else ""
                    state_name = STATE_CODES.get(state_code, "Unknown")
                    logger.info("EasyOCR plate found: '%s' conf=%.2f", formatted or best_text, best_conf)
                    return PlateResult(
                        raw_text=best_text,
                        formatted_text=formatted,
                        confidence=best_conf,
                        state_code=state_code,
                        state_name=state_name,
                        is_valid=is_valid,
                        ocr_engine="easyocr_vehicle",
                    )
            except Exception as e:
                logger.warning("EasyOCR vehicle scan error: %s", e)

        # Fallback: extract plate region contour and run standard read_plate
        plate_region = self.detect_plate_region(
            vehicle_crop,
            [0, 0, vehicle_crop.shape[1], vehicle_crop.shape[0]],
        )
        if plate_region is not None and plate_region.size > 0:
            return self.read_plate(plate_region)
        return PlateResult()

    def detect_plate_region(
        self,
        image: np.ndarray,
        vehicle_bbox: List[float],
    ) -> Optional[np.ndarray]:
        """
        Locate license plate within vehicle bounding box.

        Uses the trained YOLO plate detector when available (much more
        reliable than contours), falling back to contour-based morphological
        extraction otherwise.
        """
        if self._plate_detector is not None:
            crop = self._detect_plate_region_yolo(image, vehicle_bbox)
            if crop is not None:
                return crop

        # Fallback to morphological plate extraction pipeline
        return self.detect_plate_region_morphological(image, vehicle_bbox)

    def detect_plate_region_morphological(
        self,
        image: np.ndarray,
        vehicle_bbox: List[float],
    ) -> Optional[np.ndarray]:
        """
        Locate license plate using standard morphological image processing routines
        (Grayscale -> Gaussian Blur -> Otsu Threshold -> Canny Edge -> Dilation Loop -> Contour Filtering).
        As detailed in the SOD real-time plate isolation routines.
        """
        x1, y1, x2, y2 = map(int, vehicle_bbox)
        vh = y2 - y1

        # Search a wider region because drone/aerial angles can place plates anywhere
        # Take lower 75% of the vehicle crop
        plate_roi_y1 = y1 + int(vh * 0.25)
        roi = image[max(0, plate_roi_y1):y2, max(0, x1):x2]

        if roi.size == 0:
            return None

        # 1. Grayscale Colorspace conversion
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 2. Noise reduction via Gaussian Blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. Image thresholding (Otsu's binarization to obtain binary image)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. Canny edge detection
        edges = cv2.Canny(thresh, 30, 200)

        # 5. Dilation loop (dilate 10 times to merge characters & plate boundaries)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=10)

        # 6. Connected component labeling (find contours)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        plate_candidates = []
        roi_h, roi_w = roi.shape[:2]
        roi_area = roi_w * roi_h

        # 7. Discard regions of non-interest using shape & ratio filters
        for cnt in contours:
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            if rh < 8 or rw < 15:
                continue
            aspect = rw / rh
            area = rw * rh
            
            # Standard Indian plate ratio is roughly 3.0 to 4.5
            # We filter for aspect ratio between 2.0 and 6.5, and reasonable size relative to vehicle
            if 2.0 <= aspect <= 6.5 and (0.005 * roi_area <= area <= 0.20 * roi_area):
                score = abs(aspect - 3.5) # Prefer aspect ratios closer to 3.5
                plate_candidates.append((score, (rx, ry, rw, rh)))

        if plate_candidates:
            # Pick best candidate closest to ideal plate aspect ratio
            plate_candidates.sort(key=lambda x: x[0])
            rx, ry, rw, rh = plate_candidates[0][1]
            
            # Crop with small padding (2px)
            pad = 2
            return roi[max(0, ry - pad):min(roi_h, ry + rh + pad), max(0, rx - pad):min(roi_w, rx + rw + pad)].copy()

        # Fallback: return bottom strip of the vehicle
        return roi[int(roi_h * 0.4) :, :].copy()

    def _detect_plate_region_yolo(
        self,
        image: np.ndarray,
        vehicle_bbox: List[float],
    ) -> Optional[np.ndarray]:
        """
        2-Stage plate detection pipeline:
          Stage 1 — Koushi runs on lower 50% of vehicle crop → finds candidate plate bboxes
          Stage 2 — YasirFaiz runs on each Stage-1 crop → confirms or rejects
          Confidence gate: drop anything below _MIN_PLATE_CONF to eliminate FP (headlights etc.)
          Returns the best enhanced plate crop ready for OCR.
        """
        x1, y1, x2, y2 = map(int, vehicle_bbox)
        # Plate is always in lower half of vehicle bbox
        crop_y1 = y1 + int((y2 - y1) * 0.50)
        vehicle_lower = image[max(0, crop_y1):y2, max(0, x1):x2]
        if vehicle_lower.size == 0:
            return None

        # ── Stage 1: Koushi → candidate plate regions ────────────────────────
        s1_results = self._plate_detector.predict(
            source=vehicle_lower, conf=self._STAGE1_CONF, verbose=False
        )
        candidates: List[Tuple[float, np.ndarray]] = []  # (final_conf, crop)

        for result in s1_results:
            for box in result.boxes:
                s1_conf = float(box.conf[0])
                px1, py1c, px2, py2c = map(int, box.xyxy[0])

                # Absolute coords in original image
                abs_x1 = max(0, x1 + px1)
                abs_y1 = max(0, crop_y1 + py1c)
                abs_x2 = x1 + px2
                abs_y2 = crop_y1 + py2c

                pad = 4
                plate_crop = image[
                    max(0, abs_y1 - pad): abs_y2 + pad,
                    max(0, abs_x1 - pad): abs_x2 + pad,
                ]
                if plate_crop.size == 0:
                    continue

                # ── Stage 2: YasirFaiz on Stage-1 crop → confirm ─────────────
                final_conf = s1_conf
                refined_crop = plate_crop
                if self._stage2_detector is not None:
                    s2_results = self._stage2_detector.predict(
                        source=plate_crop, conf=self._STAGE2_CONF, verbose=False
                    )
                    for s2r in s2_results:
                        for s2box in s2r.boxes:
                            s2_conf = float(s2box.conf[0])
                            final_conf = max(s1_conf, s2_conf)
                            # Tighten crop using Stage-2 bbox
                            sx1, sy1, sx2, sy2 = map(int, s2box.xyxy[0])
                            h_c, w_c = plate_crop.shape[:2]
                            inner = plate_crop[
                                max(0, sy1): min(h_c, sy2),
                                max(0, sx1): min(w_c, sx2),
                            ]
                            if inner.size > 0:
                                refined_crop = inner

                # Confidence gate — drop headlights, windshields, etc.
                if final_conf >= self._MIN_PLATE_CONF:
                    candidates.append((final_conf, refined_crop))
                    logger.debug("Plate candidate: conf=%.2f size=%s", final_conf, refined_crop.shape)

        if not candidates:
            return None

        # Return the highest-confidence raw color crop (do NOT enhance it here)
        candidates.sort(key=lambda t: t[0], reverse=True)
        best_crop = candidates[0][1]
        return best_crop

    # ------------------------------------------------------------------
    # Internal: text extraction per engine
    # ------------------------------------------------------------------

    def _extract_text(self, image: np.ndarray) -> Tuple[str, float]:
        """Run OCR and return (uppercase_text, confidence)"""
        if self._ocr is None:
            return "", 0.0

        try:
            if self._engine_name == "fast_plate_ocr":
                return self._fast_plate_ocr_read(image)
            elif self._engine_name == "paddle":
                return self._paddle_ocr(image)
            elif self._engine_name == "easyocr":
                return self._easyocr_ocr(image)
            elif self._engine_name == "tesseract":
                return self._tesseract_ocr(image)
        except Exception as e:
            logger.warning("OCR extraction error: %s", e)

        return "", 0.0

    def _fast_plate_ocr_read(self, image: np.ndarray) -> Tuple[str, float]:
        """fast-plate-ocr: specialized plate recognition, no general OCR overhead."""
        try:
            import numpy as _np
            results = self._ocr.run(image, return_confidence=True)
            if not results:
                return "", 0.0
            pred = results[0]
            text = pred.plate.upper().strip()
            if not text or len(text) < 3:
                return "", 0.0
            # Use mean char confidence as overall score
            conf = float(_np.mean(pred.char_probs)) if hasattr(pred, "char_probs") and len(pred.char_probs) else 0.5
            return text, conf
        except Exception as e:
            logger.debug("fast-plate-ocr read failed: %s", e)
            return "", 0.0

    def _paddle_ocr(self, image: np.ndarray) -> Tuple[str, float]:
        result = self._ocr.ocr(image, cls=True)
        if not result or not result[0]:
            return "", 0.0
        lines = [(line[1][0], float(line[1][1])) for line in result[0] if line[1]]
        if not lines:
            return "", 0.0
        text = " ".join(t for t, _ in lines).upper().strip()
        avg_conf = sum(c for _, c in lines) / len(lines)
        return text, avg_conf

    def _easyocr_ocr(self, image: np.ndarray) -> Tuple[str, float]:
        result = self._ocr.readtext(image)
        if not result:
            return "", 0.0
        text = " ".join(r[1] for r in result).upper().strip()
        avg_conf = sum(r[2] for r in result) / len(result)
        return text, float(avg_conf)

    def _tesseract_ocr(self, image: np.ndarray) -> Tuple[str, float]:
        import pytesseract  # type: ignore
        config = r"--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
        text = pytesseract.image_to_string(image, config=config).strip().upper()
        # Tesseract doesn't return confidence easily; use 0.6 as default
        return text, 0.6 if text else 0.0

    # ------------------------------------------------------------------
    # Internal: image enhancement for OCR
    # ------------------------------------------------------------------

    def _enhance_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Multi-stage plate enhancement pipeline"""
        h, w = image.shape[:2]

        # 1. Upscale small plates to minimum 80px height for better OCR accuracy
        if h < 80:
            scale = 80.0 / h
            new_h = 80
            new_w = max(int(w * scale), 200)  # Also ensure minimum width
            image = cv2.resize(
                image, (new_w, new_h), interpolation=cv2.INTER_CUBIC
            )

        # 2. Convert to LAB → CLAHE on L channel → back to BGR
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        image = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        # 3. Sharpen
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image = cv2.filter2D(image, -1, kernel)

        # 4. Threshold selection based on contrast
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if gray.std() < 35:
            # Low contrast → adaptive threshold
            thr = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        else:
            _, thr = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)

    # ------------------------------------------------------------------
    # Internal: plate format parsing
    # ------------------------------------------------------------------

    def _parse_plate(self, raw: str) -> Tuple[str, bool]:
        """
        Parse raw OCR text into standard Indian plate format KA-01-AB-1234.

        Returns (formatted_text, is_valid).
        is_valid=True only if text matches a known state code pattern.
        """
        if not raw:
            return "", False

        # Clean: keep only alphanumeric characters
        cleaned = re.sub(r"[^A-Z0-9]", "", raw.upper().strip())

        # Sliding window search for 2-letter state codes
        matched_state = None
        state_idx = -1
        for state in STATE_CODES.keys():
            idx = cleaned.find(state)
            if idx != -1:
                # Prioritize state codes that appear earlier in the text
                if state_idx == -1 or idx < state_idx:
                    state_idx = idx
                    matched_state = state

        # If a state code is found, align and parse from there
        if matched_state is not None and state_idx != -1:
            plate_candidate = cleaned[state_idx:]
            if len(plate_candidate) >= 6:
                # Extract first 10 alphanumeric characters (max Indian plate length)
                plate_candidate = plate_candidate[:10]
                chars = list(plate_candidate)
                n = len(chars)
                
                # Correction rules:
                # District code (chars at index 2, 3) must be digits (e.g. MH12...)
                for i in range(2, min(4, n)):
                    if chars[i] in OCR_CORRECTIONS:
                        chars[i] = OCR_CORRECTIONS[chars[i]]
                        
                # End serial number (last 4 chars) must be digits (e.g. ...1234)
                for i in range(max(4, n - 4), n):
                    if chars[i] in OCR_CORRECTIONS:
                        chars[i] = OCR_CORRECTIONS[chars[i]]
                        
                corrected_plate = "".join(chars)
                
                # Check standard regex patterns on the corrected candidate
                for pattern in PLATE_PATTERNS:
                    match = pattern.search(corrected_plate)
                    if match:
                        plate_raw = match.group()
                        formatted = self._format_plate(plate_raw)
                        if formatted:
                            return formatted, True
                            
                # Best-effort formatting if we have a state code alignment
                formatted = self._format_plate(corrected_plate)
                if formatted:
                    return formatted, True

        # Fallback to regex check on the entire cleaned string
        for pattern in PLATE_PATTERNS:
            match = pattern.search(cleaned)
            if match:
                plate_raw = match.group()
                plate_clean = re.sub(r"[\s\-]", "", plate_raw)
                formatted = self._format_plate(plate_clean)
                if formatted:
                    return formatted, True

        # Best-effort: return cleaned text if of acceptable length
        if 6 <= len(cleaned) <= 12:
            return cleaned, False

        return "", False

    def _format_plate(self, plate: str) -> str:
        """
        Format cleaned plate string into KA-01-AB-1234 style.
        Handles various length combinations.
        """
        p = plate.upper()
        n = len(p)

        if n == 10:  # e.g. KA01AB1234
            return f"{p[0:2]}-{p[2:4]}-{p[4:6]}-{p[6:10]}"
        elif n == 9:  # e.g. KA01A1234
            return f"{p[0:2]}-{p[2:4]}-{p[4:5]}-{p[5:9]}"
        elif n == 8:  # e.g. KA011234 (old format)
            return f"{p[0:2]}-{p[2:4]}-{p[4:8]}"
        elif 6 <= n <= 12:
            # Just add hyphens after state + district
            if n >= 4:
                return f"{p[0:2]}-{p[2:4]}-{p[4:]}"
            return p
        return ""

    @property
    def engine_name(self) -> str:
        return self._engine_name

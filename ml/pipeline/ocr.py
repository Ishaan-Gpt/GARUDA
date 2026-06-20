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
    "O": "0",   # Letter O → digit 0 (in numeric positions)
    "I": "1",   # Letter I → digit 1 (in numeric positions)
    "l": "1",   # Lowercase l → 1
    "B": "8",   # B → 8 (context-dependent, handled carefully)
    "S": "5",   # S → 5
    "Z": "2",   # Z → 2
    "G": "6",
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

    Parameters
    ----------
    use_gpu : Enable GPU acceleration (auto-disabled if CUDA unavailable)
    """

    def __init__(self, use_gpu: bool = False, plate_detector_weights: Optional[str] = None) -> None:
        self.use_gpu = use_gpu
        self._engine_name: str = "none"
        self._ocr = None
        self._plate_detector = None
        self._init_engine()
        if plate_detector_weights:
            self._load_plate_detector(plate_detector_weights)

    def _load_plate_detector(self, weights_path: str) -> None:
        """Load a fine-tuned YOLO11n plate-bbox detector (see ml/training/train_plate_yolo.py)."""
        try:
            from ultralytics import YOLO  # type: ignore
            self._plate_detector = YOLO(weights_path)
            logger.info("Plate detector loaded (trained YOLO): %s", weights_path)
        except Exception as e:
            logger.warning("Could not load plate detector weights (%s). Using contour heuristic.", e)

    # ------------------------------------------------------------------
    # Engine initialisation
    # ------------------------------------------------------------------

    def _init_engine(self) -> None:
        if self._try_paddle():
            return
        if self._try_easyocr():
            return
        if self._try_tesseract():
            return
        logger.error(
            "No OCR engine found. Install: pip install paddleocr  OR  pip install easyocr"
        )

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

        # For EasyOCR: run on raw colour image — neural net works best without pre-binarisation
        raw_text = ""
        confidence = 0.0

        if self._ocr is not None:
            if self._engine_name == "easyocr":
                # EasyOCR: read raw colour, also try enhanced
                raw_text, confidence = self._easyocr_ocr(plate_image)
                if not raw_text or confidence < 0.25:
                    enhanced = self._enhance_for_ocr(plate_image)
                    raw_text2, confidence2 = self._easyocr_ocr(enhanced)
                    if confidence2 > confidence:
                        raw_text, confidence = raw_text2, confidence2
            else:
                enhanced = self._enhance_for_ocr(plate_image)
                raw_text, confidence = self._extract_text(enhanced)
                if not raw_text or confidence < 0.3:
                    raw_text2, confidence2 = self._extract_text(cv2.bitwise_not(enhanced))
                    if confidence2 > confidence:
                        raw_text, confidence = raw_text2, confidence2

        # No fake/mock plate generation — return real OCR result or empty
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
                results = self._ocr.readtext(vehicle_crop)
                best_text = ""
                best_conf = 0.0

                for (bbox_pts, text, conf) in results:
                    text_clean = re.sub(r"[^A-Z0-9\s\-]", "", text.upper().strip())
                    if len(text_clean) < 4:
                        continue
                    # Try to match any plate pattern first
                    matched = False
                    for pattern in PLATE_PATTERNS:
                        if pattern.search(text_clean):
                            if conf > best_conf:
                                best_conf = conf
                                best_text = text_clean
                            matched = True
                            break
                    if not matched:
                        # Accept 6-12 char alphanumeric if better confidence
                        no_sp = re.sub(r"[\s\-]", "", text_clean)
                        if 6 <= len(no_sp) <= 12 and conf > best_conf:
                            best_conf = conf
                            best_text = text_clean

                if best_text and best_conf > 0.2:
                    formatted, is_valid = self._parse_plate(best_text)
                    state_code = formatted[:2] if len(formatted) >= 2 else ""
                    state_name = STATE_CODES.get(state_code, "Unknown")
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

        # Fallback: extract plate region and run standard read_plate
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
        reliable than contours), falling back to contour detection +
        aspect-ratio filtering otherwise.

        Plate typical aspect ratio: 4:1 to 7:1 (width:height)
        Plate location: lower 35% of vehicle bbox
        """
        if self._plate_detector is not None:
            crop = self._detect_plate_region_yolo(image, vehicle_bbox)
            if crop is not None:
                return crop
            # fall through to heuristic if YOLO found nothing in this crop

        x1, y1, x2, y2 = map(int, vehicle_bbox)
        vh = y2 - y1

        # Focus on lower portion of vehicle where plate sits
        plate_roi_y1 = y1 + int(vh * 0.55)
        roi = image[plate_roi_y1:y2, x1:x2]

        if roi.size == 0:
            return None

        # Morphological approach for rectangular plate detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 200)

        # Dilate to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        plate_candidates: List[Tuple[int, int, int, int]] = []
        for cnt in contours:
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            if rh < 5:
                continue
            aspect = rw / rh
            area = rw * rh
            # Standard Indian plate: ~330mm × 110mm → aspect ≈ 3
            if 2.5 <= aspect <= 8.0 and area > 400:
                plate_candidates.append((rx, ry, rw, rh))

        if plate_candidates:
            # Best: largest area candidate
            rx, ry, rw, rh = max(plate_candidates, key=lambda r: r[2] * r[3])
            return roi[ry : ry + rh, rx : rx + rw]

        # Fallback: return bottom strip of vehicle (simple heuristic)
        return roi[int(roi.shape[0] * 0.3) :, :]

    def _detect_plate_region_yolo(
        self,
        image: np.ndarray,
        vehicle_bbox: List[float],
    ) -> Optional[np.ndarray]:
        """Run the fine-tuned plate detector on the vehicle crop, return the highest-confidence plate crop."""
        x1, y1, x2, y2 = map(int, vehicle_bbox)
        vehicle_crop = image[max(0, y1):y2, max(0, x1):x2]
        if vehicle_crop.size == 0:
            return None

        results = self._plate_detector.predict(source=vehicle_crop, conf=0.35, verbose=False)
        best_box, best_conf = None, 0.0
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_box = box.xyxy[0].tolist()

        if best_box is None:
            return None

        px1, py1, px2, py2 = map(int, best_box)
        crop = vehicle_crop[py1:py2, px1:px2]
        return crop if crop.size > 0 else None

    # ------------------------------------------------------------------
    # Internal: text extraction per engine
    # ------------------------------------------------------------------

    def _extract_text(self, image: np.ndarray) -> Tuple[str, float]:
        """Run OCR and return (uppercase_text, confidence)"""
        if self._ocr is None:
            return "", 0.0

        try:
            if self._engine_name == "paddle":
                return self._paddle_ocr(image)
            elif self._engine_name == "easyocr":
                return self._easyocr_ocr(image)
            elif self._engine_name == "tesseract":
                return self._tesseract_ocr(image)
        except Exception as e:
            logger.warning("OCR extraction error: %s", e)

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

        # Clean: keep only alphanumeric + spaces/hyphens
        cleaned = re.sub(r"[^A-Z0-9\s\-]", "", raw.upper().strip())
        cleaned = re.sub(r"[\s\-]+", " ", cleaned).strip()

        # Try regex patterns
        for pattern in PLATE_PATTERNS:
            match = pattern.search(cleaned)
            if match:
                plate_raw = match.group()
                plate_clean = re.sub(r"[\s\-]", "", plate_raw)
                formatted = self._format_plate(plate_clean)
                if formatted:
                    return formatted, True

        # Best-effort: return cleaned text even if not standard
        no_spaces = re.sub(r"[\s\-]", "", cleaned)
        if 6 <= len(no_spaces) <= 12:
            return no_spaces, False

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

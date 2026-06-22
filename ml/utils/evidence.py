"""
GARUDA ML Utils — Evidence Package Generator
=============================================
For every confirmed or escalated violation, generates:
  1. Annotated JPEG image (coloured boxes, labels, plate overlay, watermark)
  2. JSON violation record (full metadata, evidence paths)
  3. WhatsApp / SMS alert text (for Tier 2 escalations)

Output directory layout:
  evidence/
    annotated/   VIO-xxx.jpg   (watermarked + annotated frame)
    raw/         VIO-xxx_raw.jpg
    json/        VIO-xxx.json
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette (BGR)
# ---------------------------------------------------------------------------

SEVERITY_COLORS: Dict[str, Tuple[int, int, int]] = {
    "critical": (0,   0, 255),
    "high":     (0,  50, 255),
    "medium":   (0, 140, 255),
    "low":      (0, 210, 255),
}
DEFAULT_COLOR = (0, 140, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Evidence packager
# ---------------------------------------------------------------------------

class EvidencePackager:
    """
    Generates complete, audit-grade evidence packages.

    Parameters
    ----------
    output_dir : Root directory for evidence storage
    jpeg_quality : JPEG compression quality (0-100). 95 = high quality.
    """

    def __init__(
        self,
        output_dir: str = "evidence",
        jpeg_quality: int = 95,
    ) -> None:
        self.jpeg_quality = jpeg_quality
        self.root = Path(output_dir)
        for sub in ("annotated", "raw", "json"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_package(
        self,
        frame: np.ndarray,
        violations: List[Dict],
        plate_info: Dict,
        camera_info: Dict,
        driver_alerts: Optional[List[Dict]] = None,
        track_info: Optional[Dict] = None,
        processing_info: Optional[Dict] = None,
        violation_id: Optional[str] = None,
    ) -> Dict:
        """
        Generate full evidence package for one or more violations in a frame.

        Parameters
        ----------
        violation_id : Reuse an ID already assigned upstream (e.g. by
                       ConfidenceRouter), so evidence files and DB records
                       share the same ID. Generates a fresh one if omitted.

        Returns
        -------
        {
          "violation_id"          : str,
          "annotated_image_path"  : str,
          "raw_image_path"        : str,
          "json_path"             : str,
          "record"                : dict   # full JSON record
        }
        """
        ts = datetime.utcnow()
        vid = violation_id or self._gen_id(ts)

        # --- Save raw frame ---
        raw_path = self.root / "raw" / f"{vid}_raw.jpg"
        cv2.imwrite(str(raw_path), frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])

        # --- Generate annotated image ---
        annotated = self._annotate(frame.copy(), violations, plate_info, camera_info, ts, vid)
        ann_path = self.root / "annotated" / f"{vid}.jpg"
        cv2.imwrite(str(ann_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])

        # --- Build JSON record ---
        record = self._build_record(
            vid, violations, plate_info, camera_info,
            driver_alerts, track_info, processing_info,
            ts, str(ann_path), str(raw_path),
        )
        json_path = self.root / "json" / f"{vid}.json"
        json_path.write_text(
            json.dumps(record, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info("Evidence package created: %s", vid)
        return {
            "violation_id": vid,
            "annotated_image_path": str(ann_path),
            "raw_image_path": str(raw_path),
            "json_path": str(json_path),
            "record": record,
        }

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    def _annotate(
        self,
        frame: np.ndarray,
        violations: List[Dict],
        plate_info: Dict,
        camera_info: Dict,
        ts: datetime,
        vid: str,
    ) -> np.ndarray:
        h, w = frame.shape[:2]

        # --- Header bar ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 52), (12, 12, 22), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        cv2.putText(
            frame, "GARUDA — AUTOMATED TRAFFIC ENFORCEMENT",
            (10, 20), FONT, 0.52, (0, 212, 255), 2,
        )
        cv2.putText(
            frame,
            f"{camera_info.get('location','Unknown')}  |  {ts.strftime('%Y-%m-%d %H:%M:%S')} UTC",
            (10, 42), FONT, 0.38, (170, 170, 170), 1,
        )

        # --- Violation boxes ---
        for v in violations:
            bbox = v.get("bbox") or []
            severity = v.get("severity", "medium")
            color = SEVERITY_COLORS.get(severity, DEFAULT_COLOR)

            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Main box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                # Corner accent marks
                cl = 18
                for cx, cy, dx, dy in [
                    (x1, y1,  cl,  cl),
                    (x2, y1, -cl,  cl),
                    (x1, y2,  cl, -cl),
                    (x2, y2, -cl, -cl),
                ]:
                    cv2.line(frame, (cx, cy), (cx + dx, cy), color, 4)
                    cv2.line(frame, (cx, cy), (cx, cy + dy), color, 4)

                # Label
                vtype = v.get("type", "violation").replace("_", " ").upper()
                conf  = v.get("confidence", 0)
                plate_txt = v.get("plate_text")
                label = f"{vtype}  {conf * 100:.0f}%"
                if plate_txt and plate_txt != "UNCLEAR":
                    label += f" [{plate_txt}]"
                lw    = len(label) * 8 + 10
                label_y1 = max(0, y1 - 30)
                cv2.rectangle(frame, (x1, label_y1), (x1 + lw, y1), color, -1)
                cv2.putText(
                    frame, label,
                    (x1 + 4, max(16, y1 - 8)),
                    FONT, 0.46, (255, 255, 255), 1,
                )

        # --- Footer bar ---
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 56), (w, h), (10, 10, 18), -1)
        cv2.addWeighted(overlay2, 0.80, frame, 0.20, 0, frame)

        plate_text = (
            plate_info.get("formatted_text")
            or plate_info.get("raw_text")
            or "PLATE UNCLEAR"
        )
        plate_conf = plate_info.get("confidence", 0)
        cv2.putText(
            frame,
            f"PLATE: {plate_text}   ({plate_conf * 100:.0f}% OCR confidence)",
            (10, h - 32), FONT, 0.60, (0, 240, 100), 2,
        )
        cv2.putText(
            frame,
            f"GARUDA v1.0  |  {camera_info.get('camera_id','N/A')}  |  ID: {vid}",
            (10, h - 10), FONT, 0.36, (130, 130, 130), 1,
        )

        return frame

    # ------------------------------------------------------------------
    # JSON record builder
    # ------------------------------------------------------------------

    def _build_record(
        self,
        vid: str,
        violations: List[Dict],
        plate_info: Dict,
        camera_info: Dict,
        driver_alerts: Optional[List[Dict]],
        track_info: Optional[Dict],
        processing_info: Optional[Dict],
        ts: datetime,
        ann_path: str,
        raw_path: str,
    ) -> Dict:
        return {
            "violation_id":     vid,
            "system_version":   "GARUDA-v1.0",
            "schema_version":   "1.0",
            "timestamp":        ts.isoformat() + "Z",
            "camera": {
                "id":          camera_info.get("camera_id", "unknown"),
                "location":    camera_info.get("location",  "unknown"),
                "coordinates": camera_info.get("coordinates", {}),
            },
            "vehicle": {
                "class":            track_info.get("class", "unknown") if track_info else "unknown",
                "color":            track_info.get("color", "unknown") if track_info else "unknown",
                "track_id":         track_info.get("track_id")         if track_info else None,
                "license_plate":    plate_info.get("formatted_text", ""),
                "plate_raw":        plate_info.get("raw_text",       ""),
                "plate_confidence": round(plate_info.get("confidence", 0.0), 4),
                "plate_valid":      plate_info.get("is_valid", False),
                "plate_state":      plate_info.get("state",   "Unknown"),
                "repeat_offender":  track_info.get("repeat_offender", False) if track_info else False,
                "prior_violations": track_info.get("prior_violations", 0)    if track_info else 0,
            },
            "violations": violations,
            "driver_state": {
                "alerts":       driver_alerts or [],
                "total_alerts": len(driver_alerts) if driver_alerts else 0,
            },
            "evidence": {
                "annotated_image": ann_path,
                "raw_frame":       raw_path,
            },
            "processing": {
                "inference_device":  (processing_info or {}).get("device",    "CPU"),
                "inference_time_ms": (processing_info or {}).get("time_ms",   0),
                "model":             (processing_info or {}).get("model",     "yolo11n"),
                "ocr_engine":        plate_info.get("ocr_engine", "unknown"),
            },
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gen_id(ts: datetime) -> str:
        short = str(uuid.uuid4())[:6].upper()
        return f"VIO-BLR-{ts.strftime('%Y%m%d-%H%M%S')}-{short}"

    def load_package(self, violation_id: str) -> Optional[Dict]:
        """Load a previously generated JSON record by ID"""
        path = self.root / "json" / f"{violation_id}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def list_packages(self, limit: int = 50) -> List[str]:
        """List most recent violation IDs"""
        jsons = sorted(
            (self.root / "json").glob("VIO-*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return [p.stem for p in jsons[:limit]]

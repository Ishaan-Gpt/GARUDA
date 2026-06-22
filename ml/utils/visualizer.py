"""
GARUDA ML Utils — Frame Visualizer
=====================================
Real-time frame annotation for live dashboard display and demo.
Separate from EvidencePackager (which handles audit-grade saved images).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Design palette (BGR)
# ---------------------------------------------------------------------------

class Colors:
    CYAN     = (255, 212, 0)    # primary accent
    GREEN    = (100, 230, 0)    # safe / success
    ORANGE   = (0,   165, 255)  # warning / Tier 2
    RED      = (50,   50, 255)  # danger / violation
    CRIMSON  = (0,     0, 255)  # critical
    WHITE    = (255, 255, 255)
    DARK     = (18,  18,  30)
    GRAY     = (140, 140, 140)

SEVERITY_CLR: Dict[str, Tuple[int, int, int]] = {
    "critical": Colors.CRIMSON,
    "high":     Colors.RED,
    "medium":   Colors.ORANGE,
    "low":      Colors.CYAN,
}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = 0.42
FONT_MED   = 0.55
FONT_LARGE = 0.75
THICK      = 1


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------

class FrameVisualizer:
    """
    Lightweight frame annotator for real-time display.

    Designed to add minimal overhead:
      - Reuses pre-allocated overlay arrays where possible
      - Returns the modified frame (in-place modification)
    """

    # ------------------------------------------------------------------
    # Detection boxes
    # ------------------------------------------------------------------

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        show_conf: bool = True,
    ) -> np.ndarray:
        """Draw bounding boxes for all detections (no violation colour)"""
        for det in detections:
            bbox = det.get("bbox")
            if not bbox or len(bbox) < 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            cls   = det.get("class_name", "vehicle")
            conf  = det.get("confidence", 0.0)
            tid   = det.get("track_id")
            plate_txt = det.get("plate_text")

            label = f"#{tid} {cls}" if tid is not None else cls
            if plate_txt and plate_txt != "UNCLEAR":
                label += f" [{plate_txt}]"
            elif show_conf:
                label += f" {conf * 100:.0f}%"

            cv2.rectangle(frame, (x1, y1), (x2, y2), Colors.GREEN, 2)
            self._label(frame, label, (x1, y1), Colors.GREEN)

        return frame

    # ------------------------------------------------------------------
    # Violation overlay
    # ------------------------------------------------------------------

    def draw_violations(
        self,
        frame: np.ndarray,
        violations: List[Dict],
    ) -> np.ndarray:
        """Draw violation boxes with severity colour coding"""
        for v in violations:
            bbox     = v.get("bbox")
            severity = v.get("severity", "medium")
            color    = SEVERITY_CLR.get(severity, Colors.ORANGE)
            conf     = v.get("confidence", 0.0)
            vtype    = v.get("type", "violation").replace("_", " ").upper()

            if not bbox or len(bbox) < 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            # Double-border effect
            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), color, 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            self._label(frame, f"⚠ {vtype} {conf * 100:.0f}%", (x1, y1), color)

        return frame

    # ------------------------------------------------------------------
    # Stop line
    # ------------------------------------------------------------------

    def draw_stop_line(self, frame: np.ndarray, y: int, label: str = "STOP LINE") -> np.ndarray:
        w = frame.shape[1]
        cv2.line(frame, (0, y), (w, y), Colors.RED, 2)
        cv2.putText(frame, label, (8, y - 6), FONT, FONT_SMALL, Colors.RED, 1)
        return frame

    # ------------------------------------------------------------------
    # HUD (heads-up display)
    # ------------------------------------------------------------------

    def draw_hud(self, frame: np.ndarray, stats: Dict) -> np.ndarray:
        """
        Draw top-right HUD panel with system stats.

        Expected stats keys:
          fps, active_tracks, violations_today, tier1, tier2
        """
        h, w = frame.shape[:2]
        panel_w, panel_h = 210, 140

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - panel_w - 4, 0), (w, panel_h), Colors.DARK, -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        lines = [
            ("FPS",        f"{stats.get('fps', 0):.1f}",       Colors.CYAN),
            ("Tracks",     str(stats.get("active_tracks", 0)),  Colors.GREEN),
            ("Today",      str(stats.get("violations_today", 0)), Colors.ORANGE),
            ("Tier 1 Auto", str(stats.get("tier1", 0)),         Colors.GREEN),
            ("Tier 2 Review", str(stats.get("tier2", 0)),       Colors.ORANGE),
        ]

        for i, (key, val, color) in enumerate(lines):
            y = 20 + i * 24
            cv2.putText(frame, key, (w - panel_w + 4, y), FONT, FONT_SMALL, Colors.GRAY, 1)
            cv2.putText(frame, val, (w - 55, y), FONT, FONT_MED, color, 2)

        return frame

    # ------------------------------------------------------------------
    # Driver alert overlay
    # ------------------------------------------------------------------

    def draw_driver_alert(
        self,
        frame: np.ndarray,
        alert_type: str,
        position: Tuple[int, int] = (10, 80),
    ) -> np.ndarray:
        """Flash an alert banner for driver state events"""
        x, y = position

        # Pulsing-style: just draw a high-visibility rect + text
        text = f"⚠ {alert_type.replace('_', ' ')}"
        tw, _ = cv2.getTextSize(text, FONT, FONT_LARGE, 2)[0], None
        cv2.rectangle(frame, (x - 4, y - 30), (x + tw[0] + 8, y + 8),
                      Colors.CRIMSON, -1)
        cv2.putText(frame, text, (x, y), FONT, FONT_LARGE, Colors.WHITE, 2)
        return frame

    # ------------------------------------------------------------------
    # Tier badge
    # ------------------------------------------------------------------

    def draw_tier_badge(
        self,
        frame: np.ndarray,
        tier: int,
        action: str,
        position: Tuple[int, int] = (10, 100),
    ) -> np.ndarray:
        """Draw routing tier badge in corner of frame"""
        colors = {1: Colors.GREEN, 2: Colors.ORANGE, 3: Colors.GRAY}
        color  = colors.get(tier, Colors.GRAY)
        badge  = f"TIER {tier}: {action.replace('_', ' ')}"

        x, y = position
        (bw, bh), _ = cv2.getTextSize(badge, FONT, FONT_MED, 1)
        cv2.rectangle(frame, (x - 4, y - bh - 4), (x + bw + 6, y + 4), color, -1)
        cv2.putText(frame, badge, (x, y), FONT, FONT_MED, Colors.WHITE, 1)
        return frame

    # ------------------------------------------------------------------
    # Plate overlay
    # ------------------------------------------------------------------

    def draw_plate_result(
        self,
        frame: np.ndarray,
        plate_text: str,
        confidence: float,
        is_valid: bool,
        position: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        x, y = position or (10, h - 20)
        color = Colors.GREEN if is_valid else Colors.ORANGE
        text  = f"PLATE: {plate_text}  ({confidence * 100:.0f}%)"
        cv2.putText(frame, text, (x, y), FONT, FONT_MED, color, 2)
        return frame

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _label(
        self,
        frame: np.ndarray,
        text: str,
        pos: Tuple[int, int],
        color: Tuple[int, int, int],
        alpha: float = 0.75,
    ) -> None:
        x, y = pos
        (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SMALL, 1)
        label_y0 = max(0, y - th - 6)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, label_y0), (x + tw + 6, y), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, text, (x + 3, max(th + 2, y - 3)),
                    FONT, FONT_SMALL, Colors.WHITE, THICK)

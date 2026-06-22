"""GARUDA — Challan Service.

Packages ML inference results into the standardised violation record dict,
dispatches DB writes, and decides tier/action based on confidence thresholds.

Previously this packaging logic was inlined inside ``jobs.py`` (per-frame and
batch paths) with slight differences.  Centralising it here ensures both
paths produce identical record formats.

Usage::

    from backend.services.challan_service import ChallanService
    from backend.services.ml_registry import get_ml_registry

    svc = ChallanService(db_session)
    record = await svc.package_and_save(
        violation_id=vid,
        camera_id=camera_id,
        location=location,
        violations=violations,          # list of ViolationResult from classifier
        vehicle=vehicle,                # detected Detection object
        plate_result=plate_result,      # OCRResult from PlateOCR
        annotated_img_path=path,
        raw_img_path=raw_path,
    )
"""
from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..core.database import (
    VehicleModel,
    save_violation,
    upsert_vehicle,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier thresholds — single source of truth
# ---------------------------------------------------------------------------

TIER_1_THRESHOLD = 0.80   # >= this → AUTO_CHALLAN
TIER_2_THRESHOLD = 0.50   # >= this → HUMAN_REVIEW
# < TIER_2 → DISMISSED (not persisted)

# Fine amounts (INR) per violation display type
FINE_TABLE: Dict[str, int] = {
    "No Helmet":      1000,
    "Seatbelt":       1000,
    "Triple Riding":  2000,
    "Wrong Way":      5000,
    "Stop Line":      500,
    "Red Light":      1000,
    "Illegal Parking":500,
    "Phone Use":      1500,
    "Drowsy":         2000,
}

# Violation display-name mapping (from ML enum → display label)
VIOLATION_DISPLAY: Dict[str, str] = {
    "helmet_non_compliance":   "No Helmet",
    "seatbelt_non_compliance": "Seatbelt",
    "triple_riding":           "Triple Riding",
    "wrong_side_driving":      "Wrong Way",
    "stop_line_violation":     "Stop Line",
    "red_light_violation":     "Red Light",
    "illegal_parking":         "Illegal Parking",
    "phone_use_while_driving": "Phone Use",
    "drowsy_driving":          "Drowsy",
}


def display_name(raw: str) -> str:
    return VIOLATION_DISPLAY.get(raw, raw)


def _determine_tier(confidence: float) -> tuple[int, str]:
    if confidence >= TIER_1_THRESHOLD:
        return 1, "AUTO_CHALLAN"
    return 2, "HUMAN_REVIEW"


class ChallanService:
    """Build and persist standardised violation records."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Core packaging method
    # ------------------------------------------------------------------

    async def package_and_save(
        self,
        *,
        camera_id: str,
        location: str,
        violations: List[Any],          # list of ViolationResult from ViolationClassifier
        vehicle: Optional[Any] = None,  # Detection object (may be None for frame-level)
        plate_result: Optional[Any] = None,  # OCRResult or None
        annotated_img_path: str = "",
        raw_img_path: str = "",
        source: str = "video",
        calibrated: bool = True,
        violation_id: Optional[str] = None,
    ) -> Optional[Dict]:
        """Package a list of violations into one record and persist to DB.

        Returns the saved record dict, or ``None`` if confidence is too low.
        """
        if not violations:
            return None

        # Use the minimum confidence across all violations (most conservative)
        min_confidence = min(getattr(v, "confidence", 0.0) for v in violations)
        if min_confidence < TIER_2_THRESHOLD:
            logger.debug(
                "Violation confidence %.2f < threshold %.2f — dismissed.",
                min_confidence, TIER_2_THRESHOLD,
            )
            return None

        tier, action = _determine_tier(min_confidence)
        vid = violation_id or f"VIO-{source.upper()}-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4].upper()}"

        # Build plate info
        plate_text  = "PLATE-UNREAD"
        plate_conf  = 0.0
        plate_state = "Unknown"
        plate_state_code = ""
        plate_valid = False
        if plate_result is not None:
            plate_text  = plate_result.formatted_text or "PLATE-UNREAD"
            plate_conf  = plate_result.confidence
            plate_state = plate_result.state_name or "Unknown"
            plate_state_code = getattr(plate_result, "state_code", "")
            plate_valid = plate_result.is_valid

        # Prior-offence lookup (before upsert increments the counter)
        existing = (
            await self._session.execute(
                select(VehicleModel).where(VehicleModel.plate == plate_text)
            )
        ).scalar_one_or_none()
        prior_violations  = existing.violation_count    if existing else 0
        repeat_offender   = existing.is_repeat_offender if existing else False

        vehicle_class = getattr(vehicle, "class_name", "unknown") if vehicle else "unknown"

        # Violation list payload
        violations_payload = [
            {
                "type":             display_name(getattr(v, "violation_type", v).value
                                    if hasattr(getattr(v, "violation_type", None), "value")
                                    else str(getattr(v, "violation_type", v))),
                "confidence":       getattr(v, "confidence", 0.0),
                "severity":         getattr(v, "severity", "medium"),
                "fine_amount_inr":  getattr(v, "fine_amount", 0),
                "bbox":             list(getattr(v, "bbox", [])),
                "metadata": {
                    "source":     source,
                    "calibrated": calibrated,
                },
            }
            for v in violations
        ]

        record = {
            "violation_id": vid,
            "tier":         tier,
            "action":       action,
            "timestamp":    datetime.utcnow().isoformat() + "Z",
            "camera": {
                "id":          camera_id,
                "location":    location,
                "coordinates": {},
            },
            "vehicle": {
                "vehicle_class":    vehicle_class,
                "license_plate":    plate_text,
                "plate_confidence": plate_conf,
                "plate_valid":      plate_valid,
                "plate_state":      plate_state,
                "repeat_offender":  repeat_offender,
                "prior_violations": prior_violations,
            },
            "violations":   violations_payload,
            "driver_state": {"alerts": [], "total_alerts": 0},
            "evidence": {
                "annotated_image": annotated_img_path,
                "raw_frame":       raw_img_path,
            },
        }

        await save_violation(self._session, record)
        await upsert_vehicle(
            self._session,
            plate_text,
            violations_payload[0]["type"] if violations_payload else "",
            state_code=plate_state_code,
        )

        logger.info(
            "Challan %s — tier=%d action=%s plate=%s camera=%s",
            vid, tier, action, plate_text, camera_id,
        )
        return record

    # ------------------------------------------------------------------
    # Evidence directory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def evidence_path(vid: str, kind: str = "annotated") -> str:
        """Return the OS path for saving an evidence image.

        ``kind`` is ``'annotated'`` or ``'raw'``.
        """
        directory = f"evidence/{kind}"
        os.makedirs(directory, exist_ok=True)
        return f"{directory}/{vid}.jpg"

    @staticmethod
    def evidence_url(vid: str, kind: str = "annotated") -> str:
        """Return the public URL for serving evidence through FastAPI ``/evidence``."""
        return f"/evidence/{kind}/{vid}.jpg"

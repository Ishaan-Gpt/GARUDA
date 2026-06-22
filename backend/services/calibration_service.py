"""GARUDA — Calibration Service.

Resolves per-camera calibration parameters (stop_line_y, parking_zones,
traffic_direction, wrong_side_zone) from the database and applies them to a
``ViolationClassifier`` instance.

Previously this logic was copy-pasted inside both ``jobs.py`` and
``_routers.py`` (ws_patrol).  Centralised here so both callers are always in
sync.

Usage::

    from backend.services.calibration_service import CalibrationService
    from backend.services.ml_registry import get_ml_registry

    svc = CalibrationService(db_session)
    calibrated = await svc.apply(camera_id, get_ml_registry().classifier)
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import CameraModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default fallback calibration (generic urban intersection)
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "stop_line_y":       380,
    "parking_zones":     [],
    "traffic_direction": "down",
    "wrong_side_zone":   [],
}


class CalibrationService:
    """Resolve and apply per-camera calibration to a ViolationClassifier."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def resolve(self, camera_id: str) -> dict:
        """Return calibration dict for *camera_id*, falling back to defaults.

        Returns
        -------
        dict with keys: stop_line_y, parking_zones, traffic_direction,
        wrong_side_zone, calibrated (bool)
        """
        cam: Optional[CameraModel] = (
            await self._session.execute(
                select(CameraModel).where(CameraModel.id == camera_id)
            )
        ).scalar_one_or_none()

        if cam is None:
            logger.debug("Camera %s not registered — using default calibration.", camera_id)
            return {**_DEFAULTS, "calibrated": False}

        return {
            "stop_line_y":       cam.stop_line_y,
            "parking_zones":     json.loads(cam.parking_zones or "[]"),
            "traffic_direction": cam.traffic_direction or "down",
            "wrong_side_zone":   json.loads(cam.wrong_side_zone or "[]"),
            "calibrated":        True,
        }

    async def apply(self, camera_id: str, classifier: Any) -> bool:
        """Resolve calibration and apply it to *classifier* in place.

        Returns ``True`` if the camera was found and proper calibration applied,
        ``False`` if defaults were used.
        """
        calib = await self.resolve(camera_id)
        classifier.stop_line_y       = calib["stop_line_y"]
        classifier.parking_zones     = calib["parking_zones"]
        classifier.traffic_direction = calib["traffic_direction"]
        classifier.wrong_side_zone   = calib["wrong_side_zone"]
        return calib["calibrated"]

"""GARUDA API — Cameras router.

Endpoints:
  GET    /api/v1/cameras              List all cameras
  POST   /api/v1/cameras              Register a new camera
  GET    /api/v1/cameras/{camera_id}  Get camera details
  PUT    /api/v1/cameras/{camera_id}/config  Update calibration / RTSP config
  DELETE /api/v1/cameras/{camera_id}  Remove a camera
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import CameraModel, get_db
from ..models.schemas import (
    CameraConfigUpdate,
    CameraCreate,
    CameraResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cameras")


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------

@router.get("", response_model=List[CameraResponse])
async def list_cameras(db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(select(CameraModel))).scalars().all()
    return [CameraResponse.model_validate(r) for r in rows]


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

@router.post("", response_model=CameraResponse, status_code=201)
async def register_camera(body: CameraCreate, db: AsyncSession = Depends(get_db)):
    existing = (await db.execute(
        select(CameraModel).where(CameraModel.id == body.id)
    )).scalar_one_or_none()

    if existing:
        raise HTTPException(status_code=409, detail=f"Camera {body.id} already exists")

    cam = CameraModel(
        id=body.id, location=body.location,
        lat=body.lat, lon=body.lon,
        stop_line_y=body.stop_line_y,
        status="active",
        last_seen=datetime.utcnow().isoformat(),
        description=body.description,
        rtsp_url=body.rtsp_url,
        resolution=body.resolution,
    )
    db.add(cam)
    await db.commit()
    await db.refresh(cam)
    logger.info("Camera registered: %s @ %s", body.id, body.location)
    return CameraResponse.model_validate(cam)


# ---------------------------------------------------------------------------
# Get single
# ---------------------------------------------------------------------------

@router.get("/{camera_id}", response_model=CameraResponse)
async def get_camera(camera_id: str, db: AsyncSession = Depends(get_db)):
    cam = (await db.execute(
        select(CameraModel).where(CameraModel.id == camera_id)
    )).scalar_one_or_none()
    if not cam:
        raise HTTPException(404, f"Camera {camera_id} not found")
    return CameraResponse.model_validate(cam)


# ---------------------------------------------------------------------------
# Update config / calibration
# ---------------------------------------------------------------------------

@router.put("/{camera_id}/config", response_model=CameraResponse)
async def update_camera_config(
    camera_id: str, body: CameraConfigUpdate, db: AsyncSession = Depends(get_db)
):
    cam = (await db.execute(
        select(CameraModel).where(CameraModel.id == camera_id)
    )).scalar_one_or_none()
    if not cam:
        raise HTTPException(404, f"Camera {camera_id} not found")

    if body.stop_line_y is not None:
        cam.stop_line_y = body.stop_line_y
    if body.parking_zones is not None:
        cam.parking_zones = json.dumps(body.parking_zones)
    if body.traffic_direction is not None:
        if body.traffic_direction not in ("down", "up", "left", "right"):
            raise HTTPException(422, "traffic_direction must be one of: down, up, left, right")
        cam.traffic_direction = body.traffic_direction
    if body.wrong_side_zone is not None:
        cam.wrong_side_zone = json.dumps(body.wrong_side_zone)
    if body.description is not None:
        cam.description = body.description
    if body.rtsp_url is not None:
        cam.rtsp_url = body.rtsp_url
    if body.resolution is not None:
        cam.resolution = body.resolution

    await db.commit()
    await db.refresh(cam)
    return CameraResponse.model_validate(cam)


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

@router.delete("/{camera_id}", status_code=204)
async def delete_camera(camera_id: str, db: AsyncSession = Depends(get_db)):
    cam = (await db.execute(
        select(CameraModel).where(CameraModel.id == camera_id)
    )).scalar_one_or_none()
    if not cam:
        raise HTTPException(404, f"Camera {camera_id} not found")
    await db.delete(cam)
    await db.commit()

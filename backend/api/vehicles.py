"""GARUDA API — Vehicles router.

Endpoints:
  GET    /api/v1/vehicles/repeat     List repeat offenders
  GET    /api/v1/vehicles/{plate}    Look up vehicle by plate
  DELETE /api/v1/vehicles/{plate}/clear  Admin: reset violation history
"""
from __future__ import annotations

import json
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import VehicleModel, get_db
from ..models.schemas import VehicleResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vehicles")


@router.get("/repeat", response_model=List[VehicleResponse])
async def list_repeat_offenders(
    limit: int = Query(50, le=200),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(
        select(VehicleModel)
        .where(VehicleModel.is_repeat_offender == True)
        .order_by(VehicleModel.violation_count.desc())
        .limit(limit)
    )).scalars().all()

    results = []
    for r in rows:
        v = VehicleResponse.model_validate(r)
        v.violations = json.loads(r.violations_json or "[]")
        results.append(v)
    return results


@router.get("/{plate}", response_model=VehicleResponse)
async def get_vehicle(plate: str, db: AsyncSession = Depends(get_db)):
    plate_upper = plate.upper().strip()
    row = (await db.execute(
        select(VehicleModel).where(VehicleModel.plate == plate_upper)
    )).scalar_one_or_none()

    if not row:
        raise HTTPException(404, f"No record for plate: {plate_upper}")

    resp = VehicleResponse.model_validate(row)
    resp.violations = json.loads(row.violations_json or "[]")
    return resp


@router.delete("/{plate}/clear", status_code=204)
async def clear_vehicle_record(plate: str, db: AsyncSession = Depends(get_db)):
    """Admin endpoint — reset a vehicle's violation history."""
    plate_upper = plate.upper().strip()
    row = (await db.execute(
        select(VehicleModel).where(VehicleModel.plate == plate_upper)
    )).scalar_one_or_none()
    if row:
        await db.delete(row)
        await db.commit()

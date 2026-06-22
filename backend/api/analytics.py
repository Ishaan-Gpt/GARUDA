"""GARUDA API — Analytics router.

Endpoints:
  GET /api/v1/analytics/summary   High-level stats (today, week, breakdown)
  GET /api/v1/analytics/trends    Daily violation counts over N days
  GET /api/v1/analytics/heatmap   Per-camera geo-coords + intensity for Leaflet
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import CameraModel, ViolationModel, get_db
from ..models.schemas import (
    AnalyticsSummary,
    HeatmapPoint,
    HeatmapResponse,
    TrendPoint,
    TrendResponse,
    ViolationTypeStat,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics")


@router.get("/summary", response_model=AnalyticsSummary)
async def analytics_summary(db: AsyncSession = Depends(get_db)):
    today    = datetime.utcnow().date().isoformat()
    week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()

    total_today = (await db.execute(
        select(func.count()).select_from(ViolationModel)
        .where(ViolationModel.timestamp >= today)
    )).scalar_one()

    total_week = (await db.execute(
        select(func.count()).select_from(ViolationModel)
        .where(ViolationModel.timestamp >= week_ago)
    )).scalar_one()

    auto_count = (await db.execute(
        select(func.count()).select_from(ViolationModel)
        .where(ViolationModel.status == "auto_challan")
    )).scalar_one()

    review_count = (await db.execute(
        select(func.count()).select_from(ViolationModel)
        .where(ViolationModel.status == "pending")
    )).scalar_one()

    type_rows = (await db.execute(
        select(ViolationModel.violation_type, func.count().label("cnt"))
        .group_by(ViolationModel.violation_type)
        .order_by(func.count().desc())
    )).all()

    total_all = sum(r.cnt for r in type_rows) or 1
    breakdown = [
        ViolationTypeStat(
            violation_type=r.violation_type,
            count=r.cnt,
            percentage=round(r.cnt / total_all * 100, 1),
        )
        for r in type_rows
    ]

    top_cam_row = (await db.execute(
        select(ViolationModel.camera_id, func.count().label("cnt"))
        .group_by(ViolationModel.camera_id)
        .order_by(func.count().desc())
        .limit(1)
    )).first()

    return AnalyticsSummary(
        total_today=total_today,
        total_this_week=total_week,
        auto_challan_count=auto_count,
        human_review_count=review_count,
        top_violation_type=breakdown[0].violation_type if breakdown else "N/A",
        top_camera=top_cam_row.camera_id if top_cam_row else "N/A",
        violation_type_breakdown=breakdown,
    )


@router.get("/trends", response_model=TrendResponse)
async def violation_trends(
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
):
    since = (datetime.utcnow() - timedelta(days=days)).isoformat()
    rows = (await db.execute(
        select(
            func.substr(ViolationModel.timestamp, 1, 10).label("date"),
            func.count().label("cnt"),
        )
        .where(ViolationModel.timestamp >= since)
        .group_by(func.substr(ViolationModel.timestamp, 1, 10))
        .order_by(func.substr(ViolationModel.timestamp, 1, 10))
    )).all()

    return TrendResponse(
        period=f"last_{days}_days",
        data_points=[TrendPoint(date=r.date, count=r.cnt) for r in rows],
    )


@router.get("/heatmap", response_model=HeatmapResponse)
async def violation_heatmap(db: AsyncSession = Depends(get_db)):
    """Return per-camera violation counts with coordinates for Leaflet heatmap."""
    cam_rows  = (await db.execute(select(CameraModel))).scalars().all()
    cam_map   = {c.id: c for c in cam_rows}

    count_rows = (await db.execute(
        select(ViolationModel.camera_id, func.count().label("cnt"))
        .group_by(ViolationModel.camera_id)
    )).all()

    points: List[HeatmapPoint] = []
    for row in count_rows:
        cam = cam_map.get(row.camera_id)
        if cam and (cam.lat or cam.lon):
            points.append(HeatmapPoint(
                lat=cam.lat, lon=cam.lon,
                intensity=row.cnt,
                camera_id=cam.id,
                location=cam.location,
            ))

    return HeatmapResponse(points=points)

"""
GARUDA Backend — Pydantic Schemas
===================================
All request/response models for the FastAPI API.
Using Pydantic v2.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ViolationStatus(str, Enum):
    PENDING       = "pending"
    AUTO_CHALLAN  = "auto_challan"
    CONFIRMED     = "confirmed"
    REJECTED      = "rejected"
    ESCALATED     = "escalated"
    DISCARDED     = "discarded"

class CameraStatus(str, Enum):
    ACTIVE  = "active"
    OFFLINE = "offline"
    UNKNOWN = "unknown"

class RoutingTier(int, Enum):
    AUTO    = 1
    REVIEW  = 2
    LOW     = 3


# ---------------------------------------------------------------------------
# Violation schemas
# ---------------------------------------------------------------------------

class ViolationDetectionItem(BaseModel):
    type        : str
    confidence  : float = Field(..., ge=0, le=1)
    severity    : str
    fine_amount_inr : int
    bbox        : List[float]
    metadata    : Dict[str, Any] = {}


class PlateInfo(BaseModel):
    raw_text        : str = ""
    formatted_text  : str = ""
    confidence      : float = 0.0
    state           : str = "Unknown"
    state_code      : str = ""
    is_valid        : bool = False
    ocr_engine      : str = "unknown"


class VehicleInfo(BaseModel):
    vehicle_class       : str = ""
    color               : str = ""
    track_id            : Optional[int] = None
    license_plate       : str = ""
    plate_confidence    : float = 0.0
    plate_valid         : bool = False
    plate_state         : str = "Unknown"
    repeat_offender     : bool = False
    prior_violations    : int = 0


class CameraRef(BaseModel):
    id          : str
    location    : str
    coordinates : Dict[str, float] = {}


class DriverStateAlert(BaseModel):
    alert_type  : str
    severity    : str
    action      : str
    confidence  : float
    track_id    : Optional[int] = None
    metadata    : Dict[str, Any] = {}


class EvidencePaths(BaseModel):
    annotated_image : str = ""
    raw_frame       : str = ""


class ProcessingInfo(BaseModel):
    inference_device    : str = "CPU"
    inference_time_ms   : float = 0.0
    model               : str = "yolov8m"
    ocr_engine          : str = "unknown"


# ---------------------------------------------------------------------------
# Ingest request (from ML pipeline → backend)
# ---------------------------------------------------------------------------

class ViolationIngestRequest(BaseModel):
    """POST /api/v1/violations/ingest — submitted by ML demo_pipeline.py"""
    violation_id    : str
    tier            : RoutingTier
    action          : str
    timestamp       : str
    camera          : CameraRef
    vehicle         : VehicleInfo
    violations      : List[ViolationDetectionItem]
    driver_state    : Dict[str, Any] = {}
    evidence        : EvidencePaths = EvidencePaths()
    processing      : ProcessingInfo = ProcessingInfo()
    plate           : Optional[PlateInfo] = None
    escalation_reason : Optional[str] = None

    @field_validator("violations")
    @classmethod
    def at_least_one_violation(cls, v):
        if not v:
            raise ValueError("Must have at least one violation")
        return v


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class ViolationResponse(BaseModel):
    id              : str
    camera_id       : str
    location        : str
    timestamp       : str
    violation_type  : str
    confidence      : float
    severity        : str
    tier            : int
    action          : str
    fine_amount     : int
    plate_text      : str
    plate_conf      : float = 0.0
    vehicle_class   : str
    annotated_img   : str
    raw_img         : str = ""
    status          : str
    officer_id      : Optional[str] = None
    created_at      : str

    model_config = {"from_attributes": True}


class ViolationDetailResponse(ViolationResponse):
    json_record : str = "{}"


class ViolationListResponse(BaseModel):
    total       : int
    page        : int
    page_size   : int
    violations  : List[ViolationResponse]


class OfficerActionRequest(BaseModel):
    officer_id  : str = "officer_001"
    notes       : Optional[str] = None


class OfficerActionResponse(BaseModel):
    violation_id : str
    new_status   : str
    officer_id   : str
    message      : str


# ---------------------------------------------------------------------------
# Camera schemas
# ---------------------------------------------------------------------------

class CameraCreate(BaseModel):
    id          : str
    location    : str
    lat         : float = 0.0
    lon         : float = 0.0
    stop_line_y : int = 380
    description : str = ""
    rtsp_url    : str = ""
    resolution  : str = ""


class CameraResponse(BaseModel):
    id          : str
    location    : str
    lat         : float
    lon         : float
    stop_line_y : int
    status      : str
    last_seen   : str
    description : str
    rtsp_url    : str
    resolution  : str
    parking_zones     : List[List[int]] = []
    traffic_direction  : str = "down"
    wrong_side_zone    : List[List[int]] = []

    model_config = {"from_attributes": True}

    @field_validator("parking_zones", "wrong_side_zone", mode="before")
    @classmethod
    def _parse_zone_json(cls, v):
        if isinstance(v, str):
            import json
            return json.loads(v) if v else []
        return v


class CameraConfigUpdate(BaseModel):
    stop_line_y       : Optional[int] = None
    parking_zones     : Optional[List[List[int]]] = None
    traffic_direction  : Optional[str] = None   # "down"|"up"|"left"|"right"
    wrong_side_zone    : Optional[List[List[int]]] = None
    description       : Optional[str] = None
    rtsp_url          : Optional[str] = None
    resolution        : Optional[str] = None


# ---------------------------------------------------------------------------
# Vehicle schemas
# ---------------------------------------------------------------------------

class VehicleResponse(BaseModel):
    plate               : str
    violation_count     : int
    is_repeat_offender  : bool
    first_seen          : str
    last_seen           : str
    state_code          : str
    violations          : List[Dict[str, Any]] = []

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Analytics schemas
# ---------------------------------------------------------------------------

class ViolationTypeStat(BaseModel):
    violation_type  : str
    count           : int
    percentage      : float


class AnalyticsSummary(BaseModel):
    total_today             : int
    total_this_week         : int
    auto_challan_count      : int
    human_review_count      : int
    top_violation_type      : str
    top_camera              : str
    violation_type_breakdown: List[ViolationTypeStat]


class TrendPoint(BaseModel):
    date    : str
    count   : int


class TrendResponse(BaseModel):
    period      : str
    data_points : List[TrendPoint]


class HeatmapPoint(BaseModel):
    lat         : float
    lon         : float
    intensity   : int
    camera_id   : str
    location    : str


class HeatmapResponse(BaseModel):
    points : List[HeatmapPoint]


# ---------------------------------------------------------------------------
# WebSocket event schemas
# ---------------------------------------------------------------------------

class WSViolationEvent(BaseModel):
    event           : str = "violation_detected"
    violation_id    : str
    violation_type  : str
    confidence      : float
    tier            : int
    plate           : str
    camera_id       : str
    location        : str
    timestamp       : str
    severity        : str
    annotated_image_url : str = ""


class WSSystemStats(BaseModel):
    event               : str = "system_stats"
    fps                 : float = 0.0
    active_cameras      : int = 0
    violations_today    : int = 0
    tier1_count         : int = 0
    tier2_count         : int = 0
    pending_reviews     : int = 0


# ---------------------------------------------------------------------------
# Debug schemas
# ---------------------------------------------------------------------------

class DebugInjectRequest(BaseModel):
    """POST /debug/inject-violation — insert fake violation for testing"""
    violation_type  : str = "helmet_non_compliance"
    confidence      : float = 0.75
    tier            : int = 2
    plate           : str = "KA-01-AB-1234"
    camera_id       : str = "BLR-CAM-DEMO-001"
    location        : str = "MG Road & Brigade Road"


class HealthResponse(BaseModel):
    status      : str = "ok"
    version     : str
    db_ok       : bool
    uptime_sec  : float

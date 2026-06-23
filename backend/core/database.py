"""
GARUDA Backend — Async Database
=================================
SQLite (dev) via aiosqlite / SQLAlchemy async ORM.
Switch to PostgreSQL by changing DATABASE_URL in .env.

Tables created automatically on startup.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Float, Integer, Boolean, Text, DateTime

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Violation type normalization
# ---------------------------------------------------------------------------
# The ML classifier (ml/pipeline/violation_classifier.py) emits snake_case
# enum values (e.g. "red_light_violation"). All other ingestion paths
# (public reports, debug injection) use the human-readable display form.
# Normalize here so every record in the DB uses one consistent label,
# regardless of which path produced it.
VIOLATION_TYPE_DISPLAY_MAP = {
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


def normalize_violation_type(raw_type: str) -> str:
    return VIOLATION_TYPE_DISPLAY_MAP.get(raw_type, raw_type)


# ---------------------------------------------------------------------------
# Engine + session factory
# ---------------------------------------------------------------------------

db_url = settings.DATABASE_URL
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
elif db_url.startswith("postgresql://"):
    db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

engine = create_async_engine(
    db_url,
    echo=settings.DEBUG,
    connect_args={"check_same_thread": False}
    if "sqlite" in db_url
    else {},
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


# ---------------------------------------------------------------------------
# ORM base + models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class ViolationModel(Base):
    __tablename__ = "violations"

    id             : Mapped[str]   = mapped_column(String, primary_key=True)
    camera_id      : Mapped[str]   = mapped_column(String, index=True)
    location       : Mapped[str]   = mapped_column(String, default="")
    timestamp      : Mapped[str]   = mapped_column(String, index=True)
    violation_type : Mapped[str]   = mapped_column(String, index=True)
    confidence     : Mapped[float] = mapped_column(Float)
    severity       : Mapped[str]   = mapped_column(String)
    tier           : Mapped[int]   = mapped_column(Integer)
    action         : Mapped[str]   = mapped_column(String)
    fine_amount    : Mapped[int]   = mapped_column(Integer, default=0)
    plate_text     : Mapped[str]   = mapped_column(String, default="", index=True)
    plate_conf     : Mapped[float] = mapped_column(Float,  default=0.0)
    vehicle_class  : Mapped[str]   = mapped_column(String, default="")
    annotated_img  : Mapped[str]   = mapped_column(String, default="")
    raw_img        : Mapped[str]   = mapped_column(String, default="")
    json_record    : Mapped[str]   = mapped_column(Text,   default="{}")
    status         : Mapped[str]   = mapped_column(String, default="pending", index=True)
    officer_id     : Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at     : Mapped[str]   = mapped_column(String, default=lambda: datetime.utcnow().isoformat())


class CameraModel(Base):
    __tablename__ = "cameras"

    id           : Mapped[str]   = mapped_column(String, primary_key=True)
    location     : Mapped[str]   = mapped_column(String)
    lat          : Mapped[float] = mapped_column(Float, default=0.0)
    lon          : Mapped[float] = mapped_column(Float, default=0.0)
    stop_line_y  : Mapped[int]   = mapped_column(Integer, default=380)
    status       : Mapped[str]   = mapped_column(String, default="active")
    last_seen    : Mapped[str]   = mapped_column(String, default="")
    description  : Mapped[str]   = mapped_column(String, default="")
    rtsp_url     : Mapped[str]   = mapped_column(String, default="")
    resolution   : Mapped[str]   = mapped_column(String, default="")
    # Calibration for stop-line/red-light/wrong-side/illegal-parking detection.
    # JSON-encoded list of [x1,y1,x2,y2] no-parking rectangles.
    parking_zones    : Mapped[str] = mapped_column(Text, default="[]")
    # Legal direction of travel as seen by this camera: "down"|"up"|"left"|"right"
    traffic_direction: Mapped[str] = mapped_column(String, default="down")
    # JSON-encoded list of [x1,y1,x2,y2] zones reserved for oncoming traffic only.
    wrong_side_zone   : Mapped[str] = mapped_column(Text, default="[]")


class VehicleModel(Base):
    __tablename__ = "vehicles"

    plate              : Mapped[str]  = mapped_column(String, primary_key=True)
    violation_count    : Mapped[int]  = mapped_column(Integer, default=0)
    is_repeat_offender : Mapped[bool] = mapped_column(Boolean, default=False)
    first_seen         : Mapped[str]  = mapped_column(String, default="")
    last_seen          : Mapped[str]  = mapped_column(String, default="")
    violations_json    : Mapped[str]  = mapped_column(Text,   default="[]")
    state_code         : Mapped[str]  = mapped_column(String, default="")


class UserModel(Base):
    __tablename__ = "users"

    id                 : Mapped[str]  = mapped_column(String, primary_key=True)
    name               : Mapped[str]  = mapped_column(String)
    role               : Mapped[str]  = mapped_column(String, default="Operator")
    email              : Mapped[str]  = mapped_column(String, default="")
    status             : Mapped[str]  = mapped_column(String, default="Active")
    last_login         : Mapped[str]  = mapped_column(String, default="")
    password_hash      : Mapped[str]  = mapped_column(String, default="")
    is_verified        : Mapped[bool] = mapped_column(Boolean, default=False)
    verification_token : Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)


class JobModel(Base):
    __tablename__ = "processing_jobs"

    id               : Mapped[str]   = mapped_column(String, primary_key=True)
    name             : Mapped[str]   = mapped_column(String)
    source_type      : Mapped[str]   = mapped_column(String, default="Video")
    progress         : Mapped[int]   = mapped_column(Integer, default=0)
    status           : Mapped[str]   = mapped_column(String, default="Queued")
    duration         : Mapped[int]   = mapped_column(Integer, default=0)
    frames_processed : Mapped[int]   = mapped_column(Integer, default=0)
    violations_found : Mapped[int]   = mapped_column(Integer, default=0)
    upload_time      : Mapped[str]   = mapped_column(String, default=lambda: datetime.utcnow().isoformat() + "Z")
    camera_id        : Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # JSON-encoded {"records": [...every per-image/per-frame pipeline record,
    # violation or compliant, never collapsed...]} — the full real-pipeline
    # breakdown for the Evidence page, independent of which records also made
    # it into the violations table (that save path intentionally collapses
    # an all-compliant video to one representative row; this column doesn't).
    result_summary   : Mapped[str]   = mapped_column(Text, default="{}")


class AuditLogModel(Base):
    __tablename__ = "audit_logs"

    id        : Mapped[int]  = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp : Mapped[str]  = mapped_column(String, default=lambda: datetime.utcnow().isoformat() + "Z")
    actor     : Mapped[str]  = mapped_column(String)
    action    : Mapped[str]  = mapped_column(String)
    target    : Mapped[str]  = mapped_column(String)
    details   : Mapped[str]  = mapped_column(String, default="")


# ---------------------------------------------------------------------------
# Init / teardown
# ---------------------------------------------------------------------------

async def _add_missing_columns(conn, table: str, new_columns: Dict[str, str]) -> None:
    """
    SQLAlchemy's create_all() only creates missing tables, never alters
    existing ones. Add columns shipped after a table's initial release to
    already-deployed databases, in place.

    SQLite-only (PRAGMA table_info): fine for dev/this project. A fresh
    Postgres deployment gets new columns straight from create_all() since
    the table won't exist yet there.
    """
    if "sqlite" not in db_url:
        return
    result = await conn.execute(text(f"PRAGMA table_info({table})"))
    existing = {row[1] for row in result.fetchall()}
    for col, ddl in new_columns.items():
        if col not in existing:
            await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}"))
            logger.info("Migrated %s table: added column %s", table, col)


async def init_db() -> None:
    """Create all tables. Called on FastAPI startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _add_missing_columns(conn, "cameras", {
            "parking_zones":     "TEXT DEFAULT '[]'",
            "traffic_direction": "VARCHAR DEFAULT 'down'",
            "wrong_side_zone":   "TEXT DEFAULT '[]'",
        })
        await _add_missing_columns(conn, "processing_jobs", {
            "camera_id":      "VARCHAR",
            "result_summary": "TEXT DEFAULT '{}'",
        })
    logger.info("Database initialised: %s", db_url)

    # Seed default users
    from .auth_utils import hash_password
    async with AsyncSessionLocal() as session:
        from sqlalchemy import select
        # Check if users already exist
        result = await session.execute(select(UserModel))
        users = result.scalars().all()
        if not users:
            logger.info("Seeding default users database...")
            default_users = [
                UserModel(id="USR-001", name="Officer Keshav", role="Admin", email="keshav@enforcement.gov", status="Active", password_hash=hash_password("admin123"), is_verified=True),
                UserModel(id="USR-002", name="Analyst Priya", role="Reviewer", email="priya@enforcement.gov", status="Active", password_hash=hash_password("priya123"), is_verified=True),
                UserModel(id="USR-003", name="Supervisor Sanjay", role="Supervisor", email="sanjay@enforcement.gov", status="Active", password_hash=hash_password("sanjay123"), is_verified=True),
                UserModel(id="USR-004", name="Operator Amit", role="Operator", email="amit@controlroom.gov", status="Active", password_hash=hash_password("amit123"), is_verified=True),
            ]
            session.add_all(default_users)
            await session.commit()
            logger.info("Default users seeded successfully.")


async def close_db() -> None:
    await engine.dispose()
    logger.info("Database connection closed")


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

async def get_db():
    """FastAPI dependency: yields an async DB session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3, "none": -1}


def _club_violations_summary(violations: List[Dict]) -> tuple[str, float, str, int]:
    """
    One image can carry several distinct violations (e.g. helmet + triple
    riding on the same frame). They're saved as ONE clubbed citation row, not
    one row per violation, so an officer reviews the whole image at once.
    Roll the list up into one summary: every distinct type (joined), the
    weakest (minimum) confidence — the figure that actually decides whether
    the case needed review — the worst severity, and the total fine.
    """
    if not violations:
        return "", 0.0, "", 0

    types: List[str] = []
    for v in violations:
        t = normalize_violation_type(v.get("type", ""))
        if t not in types:
            types.append(t)

    confidence = min(v.get("confidence", 0.0) for v in violations)
    severity = max((v.get("severity", "low") for v in violations), key=lambda s: _SEVERITY_RANK.get(s, 0))
    fine_amount = sum(v.get("fine_amount_inr", 0) for v in violations)
    return ", ".join(types), confidence, severity, fine_amount


async def save_violation(session: AsyncSession, record: Dict) -> ViolationModel:
    violation_type, confidence, severity, fine_amount = _club_violations_summary(record.get("violations", []))
    obj = ViolationModel(
        id             = record["violation_id"],
        camera_id      = record.get("camera", {}).get("id", ""),
        location       = record.get("camera", {}).get("location", ""),
        timestamp      = record.get("timestamp", datetime.utcnow().isoformat()),
        violation_type = violation_type,
        confidence     = confidence,
        severity       = severity,
        tier           = record.get("tier", 3),
        action         = record.get("action", ""),
        fine_amount    = fine_amount,
        plate_text     = record.get("vehicle", {}).get("license_plate", ""),
        plate_conf     = record.get("vehicle", {}).get("plate_confidence", 0.0),
        vehicle_class  = record.get("vehicle", {}).get("vehicle_class", ""),
        annotated_img  = record.get("evidence", {}).get("annotated_image", ""),
        raw_img        = record.get("evidence", {}).get("raw_frame", ""),
        json_record    = json.dumps(record, default=str),
        status         = "auto_challan" if record.get("tier") == 1 else "pending",
    )
    session.add(obj)
    await session.commit()
    await session.refresh(obj)
    
    if obj.status == "auto_challan":
        try:
            from ..services.sms_service import send_challan_sms
            await send_challan_sms(obj, force=False)
        except Exception as sms_err:
            logger.error("Failed to automatically send SMS challan on save: %s", sms_err)
            
    return obj


async def update_violation_status(
    session: AsyncSession,
    violation_id: str,
    status: str,
    officer_id: Optional[str] = None,
) -> Optional[ViolationModel]:
    from sqlalchemy import select
    result = await session.execute(
        select(ViolationModel).where(ViolationModel.id == violation_id)
    )
    obj = result.scalar_one_or_none()
    if obj:
        status_changed = obj.status != status
        obj.status     = status
        obj.officer_id = officer_id
        await session.commit()
        await session.refresh(obj)
        
        if status_changed and status == "confirmed":
            try:
                from ..services.sms_service import send_challan_sms
                await send_challan_sms(obj, force=False)
            except Exception as sms_err:
                logger.error("Failed to send confirmed SMS challan: %s", sms_err)
    return obj


async def upsert_vehicle(session: AsyncSession, plate: str, violation_type: str, state_code: str = "") -> None:
    from sqlalchemy import select
    result = await session.execute(
        select(VehicleModel).where(VehicleModel.plate == plate)
    )
    obj = result.scalar_one_or_none()
    now = datetime.utcnow().isoformat()

    if obj:
        obj.violation_count += 1
        obj.last_seen = now
        obj.is_repeat_offender = obj.violation_count >= 3
        violations = json.loads(obj.violations_json or "[]")
        violations.append({"type": violation_type, "time": now})
        obj.violations_json = json.dumps(violations[-50:])  # Keep last 50
    else:
        obj = VehicleModel(
            plate              = plate,
            violation_count    = 1,
            is_repeat_offender = False,
            first_seen         = now,
            last_seen          = now,
            violations_json    = json.dumps([{"type": violation_type, "time": now}]),
            state_code         = state_code,
        )
        session.add(obj)

    await session.commit()

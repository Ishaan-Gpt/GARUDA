"""
GARUDA Backend — Configuration
================================
All settings read from environment variables / .env file.
Defaults are safe for local development.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    APP_NAME: str = "GARUDA"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    SECRET_KEY: str = "CHANGE_ME_IN_PRODUCTION_please_use_a_long_random_string"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_ORIGINS: List[str] = ["*"]

    # Database
    # SQLite for dev: sqlite+aiosqlite:///./garuda.db
    # Postgres for prod: postgresql+asyncpg://user:pass@localhost/garuda
    DATABASE_URL: str = "sqlite+aiosqlite:///./garuda.db"

    # Redis (for vehicle track state — optional)
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_ENABLED: bool = False

    # Evidence storage
    EVIDENCE_DIR: str = "evidence"

    # Twilio (leave empty = mock/log mode)
    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""
    TWILIO_FROM_NUMBER: str = ""       # e.g. +14155238886 (WhatsApp sandbox)
    TWILIO_WHATSAPP_FROM: str = ""     # whatsapp:+14155238886
    ALERTS_ENABLED: bool = False       # Set True only when Twilio creds are set

    # Officer phone numbers (comma-separated for demo)
    OFFICER_PHONES: str = "+919999999999,+918888888888"

    # ML Pipeline settings
    MODEL_PATH: str = ""               # Leave empty = use ml/models/weights/yolov8m.pt
    DEVICE: str = "cpu"                # "cpu" | "cuda:0" | "0"
    STOP_LINE_Y: int = 380             # Pixels — calibrate per camera
    CONFIDENCE_TIER1: float = 0.90
    CONFIDENCE_TIER2: float = 0.60

    # Federated Learning
    FL_SERVER_ADDRESS: str = "localhost:8080"
    FL_ENABLED: bool = False

    # Camera heartbeat (seconds between expected updates)
    CAMERA_TIMEOUT_SEC: int = 30

    # SMTP Email Settings (for real verification)
    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM_EMAIL: str = ""
    SMTP_USE_TLS: bool = True

    @property
    def officer_phone_list(self) -> List[str]:
        return [p.strip() for p in self.OFFICER_PHONES.split(",") if p.strip()]

    @property
    def evidence_path(self) -> Path:
        p = Path(self.EVIDENCE_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache()
def get_settings() -> Settings:
    return Settings()

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from pydantic import BaseModel
from datetime import datetime
import asyncio
import uuid

from ..core.database import JobModel, get_db, AsyncSessionLocal

router = APIRouter(prefix="/jobs")

class JobCreate(BaseModel):
    name: str
    source_type: str  # "Image" or "Video"

class JobResponse(BaseModel):
    id: str
    name: str
    source_type: str
    progress: int
    status: str
    duration: int
    frames_processed: int
    violations_found: int
    upload_time: str

    class Config:
        from_attributes = True

async def run_job_pipeline(job_id: str):
    """Progress tracker for batch upload jobs. Real ML inference on file is wired here in future."""
    await asyncio.sleep(2)

    async with AsyncSessionLocal() as session:
        job = (await session.execute(select(JobModel).where(JobModel.id == job_id))).scalar_one_or_none()
        if job:
            job.status = "Processing"
            job.progress = 25
            job.duration = 2
            await session.commit()

    await asyncio.sleep(2)

    async with AsyncSessionLocal() as session:
        job = (await session.execute(select(JobModel).where(JobModel.id == job_id))).scalar_one_or_none()
        if job:
            job.progress = 65
            job.duration = 4
            job.frames_processed = 120
            await session.commit()

    await asyncio.sleep(2)

    async with AsyncSessionLocal() as session:
        job = (await session.execute(select(JobModel).where(JobModel.id == job_id))).scalar_one_or_none()
        if job:
            job.progress = 100
            job.status = "Completed"
            job.duration = 6
            job.frames_processed = 240 if job.source_type == "Video" else 1
            job.violations_found = 0  # Real count comes from actual ML inference on the uploaded file
            await session.commit()

@router.get("", response_model=List[JobResponse])
async def list_jobs(db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(select(JobModel).order_by(JobModel.upload_time.desc()))).scalars().all()
    return [JobResponse.model_validate(r) for r in rows]

@router.post("", response_model=JobResponse, status_code=201)
async def create_job(body: JobCreate, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)):
    job_id = f"JOB-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4].upper()}"
    job = JobModel(
        id=job_id,
        name=body.name,
        source_type=body.source_type,
        progress=0,
        status="Queued",
        duration=0,
        frames_processed=0,
        violations_found=0,
        upload_time=datetime.utcnow().isoformat() + "Z"
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    background_tasks.add_task(run_job_pipeline, job_id)
    return JobResponse.model_validate(job)

@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, db: AsyncSession = Depends(get_db)):
    job = (await db.execute(select(JobModel).where(JobModel.id == job_id))).scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobResponse.model_validate(job)

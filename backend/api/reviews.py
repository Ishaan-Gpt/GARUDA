from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from ..core.database import AuditLogModel, ViolationModel, get_db, update_violation_status

router = APIRouter(prefix="/reviews")

class ReviewSubmitRequest(BaseModel):
    violation_id: str
    action: str  # "Approved" | "Rejected" | "Escalated"
    reviewer: str
    reason: Optional[str] = None
    corrected_plate_text: Optional[str] = None
    # When set, the action applies to just ONE violation within a clubbed
    # multi-violation record (json_record["violations"][item_index]) instead
    # of the whole row — lets an officer approve one finding in an image
    # and reject another from the same review page. Omitted = whole-row
    # action (the common case: a record with a single violation).
    item_index: Optional[int] = None

class ReviewLogResponse(BaseModel):
    id: int
    timestamp: str
    actor: str
    action: str
    target: str
    details: str

    class Config:
        from_attributes = True

@router.get("", response_model=List[ReviewLogResponse])
async def list_reviews(db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(
        select(AuditLogModel)
        .where(AuditLogModel.action.in_(["CITATION_APPROVED", "CITATION_REJECTED", "CITATION_ESCALATED"]))
        .order_by(AuditLogModel.timestamp.desc())
    )).scalars().all()
    return [ReviewLogResponse.model_validate(r) for r in rows]

@router.post("")
async def submit_review(body: ReviewSubmitRequest, db: AsyncSession = Depends(get_db)):
    # Map status
    db_status = "pending"
    action_type = "CITATION_ESCALATED"
    if body.action == "Approved":
        db_status = "confirmed"
        action_type = "CITATION_APPROVED"
    elif body.action == "Rejected":
        db_status = "rejected"
        action_type = "CITATION_REJECTED"

    violation = (await db.execute(select(ViolationModel).where(ViolationModel.id == body.violation_id))).scalar_one_or_none()
    if not violation:
        raise HTTPException(status_code=404, detail=f"Violation {body.violation_id} not found")

    audit_target = body.violation_id

    if body.item_index is not None:
        # Act on just one violation inside a clubbed multi-violation record —
        # an officer can approve one finding and reject another from the
        # same image without leaving the page. The row's overall status
        # only resolves once every item in it has been decided.
        try:
            record = json.loads(violation.json_record or "{}")
        except json.JSONDecodeError:
            record = {}
        items = record.get("violations", [])
        if not (0 <= body.item_index < len(items)):
            raise HTTPException(status_code=400, detail=f"item_index {body.item_index} out of range")

        item_review_status = (
            "confirmed" if body.action == "Approved"
            else "rejected" if body.action == "Rejected"
            else items[body.item_index].get("review_status", "pending")
        )
        items[body.item_index]["review_status"] = item_review_status
        items[body.item_index]["reviewed_by"] = body.reviewer
        if body.action == "Escalated":
            items[body.item_index]["escalated"] = True
            items[body.item_index]["escalation_reason"] = body.reason

        # Row resolves once no item is left pending: confirmed if anything
        # was approved, rejected only if every item was rejected.
        statuses = [it.get("review_status", "pending") for it in items]
        if any(s == "pending" for s in statuses):
            db_status = "pending"
        elif any(s == "confirmed" for s in statuses):
            db_status = "confirmed"
        else:
            db_status = "rejected"

        record["violations"] = items
        violation.json_record = json.dumps(record, default=str)
        audit_target = f"{body.violation_id}#{body.item_index}"

    await update_violation_status(db, body.violation_id, db_status, body.reviewer)

    plate_corrected = False
    if body.corrected_plate_text and body.corrected_plate_text.strip().upper() != (violation.plate_text or "").strip().upper():
        old_plate = violation.plate_text or "UNCLEAR"
        violation.plate_text = body.corrected_plate_text.strip().upper()
        plate_corrected = True

    # Save audit log
    details = body.reason or f"Violation status changed to {body.action.lower()}"
    if plate_corrected:
        details += f" | OCR plate corrected: {old_plate} -> {violation.plate_text}"
    log = AuditLogModel(
        timestamp=datetime.utcnow().isoformat() + "Z",
        actor=body.reviewer,
        action=action_type,
        target=audit_target,
        details=details
    )
    db.add(log)
    await db.commit()
    return {"status": "ok", "message": f"Review action {body.action} logged successfully", "plate_corrected": plate_corrected}
